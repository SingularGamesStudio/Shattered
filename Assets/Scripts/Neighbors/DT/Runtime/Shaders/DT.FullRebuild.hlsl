// DTRebuild.compute
// Full scratch rebuild path for 2D DT on GPU.
// Assumptions:
// - _VertexCount may include super / guard vertices; _RealVertexCount are the real input vertices.
// - Host dispatches kernels in the order listed at the end of this file.
// - Host copies _RebuildCounters[CTR_TRI_COUNT] -> _TriCount and _RebuildCounters[CTR_HE_USED] -> _HalfEdgeCount
//   before running FixHalfEdges / LegalizeHalfEdges from the existing runtime.
// - Grid/hash capacities must be sized generously.
//
// New resources added here:
//   _GridSeedOwner
//   _VoronoiA / _VoronoiB
//   _TriHash*
//   _TriRaw
//   _SiteSeenInTri
//   _MissingSites
//   _EdgeHash*
//   _RebuildCounters
//
// Existing runtime resources are kept and reused.

struct HalfEdge
{
    int v;
    int next;
    int twin;
    int t;
};

StructuredBuffer<float2> _Positions;
RWStructuredBuffer<HalfEdge> _HalfEdges;
StructuredBuffer<HalfEdge> _HalfEdgesSrc;
RWStructuredBuffer<HalfEdge> _HalfEdgesDst;
RWStructuredBuffer<int> _TriLocks;
RWStructuredBuffer<int> _VToE;
RWStructuredBuffer<int> _Neighbors;
RWStructuredBuffer<int> _NeighborCounts;
RWStructuredBuffer<uint> _FlipCount;
RWStructuredBuffer<int> _TriToHE;
RWStructuredBuffer<int> _TriToHEAll;
RWStructuredBuffer<uint> _DirtyVertexFlags;
RWStructuredBuffer<uint> _TriInternal;
RWStructuredBuffer<uint> _BoundaryEdgeFlags;
RWStructuredBuffer<float2> _BoundaryNormals;
RWStructuredBuffer<int> _OwnerByVertex;
RWStructuredBuffer<float4> _BoundsPartials;
RWStructuredBuffer<float4> _BoundsResult;
RWStructuredBuffer<int> _Debug;

int _VertexCount;
int _RealVertexCount;
int _HalfEdgeCount;
int _TriCount;
int _NeighborCount;
int _UseSupportRadiusFilter;
float _SupportRadius2;
int _BoundsPartialCount;

// -------------------------
// Rebuild resources
// -------------------------

RWStructuredBuffer<int>  _GridSeedOwner;     // size: _GridW * _GridH, init INT_MAX
RWStructuredBuffer<int>  _VoronoiA;          // size: _GridW * _GridH
RWStructuredBuffer<int>  _VoronoiB;          // size: _GridW * _GridH

RWStructuredBuffer<uint> _TriHashA;          // size: _TriHashSize
RWStructuredBuffer<uint> _TriHashB;
RWStructuredBuffer<uint> _TriHashC;
RWStructuredBuffer<uint> _TriHashState;      // 0 empty, 1 writing, 2 occupied

RWStructuredBuffer<uint3> _TriRaw;           // size: _MaxTriangles

RWStructuredBuffer<uint> _SiteSeenInTri;     // size: _VertexCount
RWStructuredBuffer<int>  _MissingSites;      // size: _MaxMissingVertices

RWStructuredBuffer<uint> _EdgeHashSrc;       // size: _EdgeHashSize
RWStructuredBuffer<uint> _EdgeHashDst;
RWStructuredBuffer<int>  _EdgeHashHE;
RWStructuredBuffer<uint> _EdgeHashState;     // 0 empty, 1 writing, 2 occupied

RWStructuredBuffer<uint> _RebuildCounters;   // generic counters

int _GridW;
int _GridH;
int _TriHashSize;
int _EdgeHashSize;
int _MaxTriangles;
int _MaxHalfEdges;
int _MaxMissingVertices;
int _InsertionWalkLimit;
float _RebuildPadding;
float _InsideEps;

// -------------------------
// Counter layout
// -------------------------

static const uint CTR_TRI_COUNT      = 0u; // final triangle count after compaction
static const uint CTR_HE_USED        = 1u; // used halfedges
static const uint CTR_MISSING_COUNT  = 2u; // number of missing sites
static const uint CTR_TRI_USED       = 3u; // allocator for dynamic insertions
static const uint CTR_INSERTED_COUNT = 4u; // number of successful missing-site insertions

// -------------------------
// Shared-edge conflict prune resources
// -------------------------

RWStructuredBuffer<uint>  _TriReject;        // size: _MaxTriangles, 0 keep, 1 reject
RWStructuredBuffer<uint3> _TriTemp;          // size: _MaxTriangles, compacted survivors

RWStructuredBuffer<uint> _EdgeRecHash;       // size: _MaxEdgeRecords
RWStructuredBuffer<uint> _EdgeRecA;          // canonical a=min(u,v)
RWStructuredBuffer<uint> _EdgeRecB;          // canonical b=max(u,v)
RWStructuredBuffer<int>  _EdgeRecTri;        // owner triangle
RWStructuredBuffer<int>  _EdgeRecOpp;        // opposite vertex in that triangle

int _MaxEdgeRecords;
int _SortK;
int _SortJ;

static const uint CTR_TRI_FILTERED = 5u;

// -------------------------
// Geometry helpers
// -------------------------

static float Orient2D(float2 a, float2 b, float2 c)
{
    float2 ab = b - a;
    float2 ac = c - a;
    return ab.x * ac.y - ab.y * ac.x;
}

static float InCircleDet(float2 a, float2 b, float2 c, float2 p)
{
    float2 ap = a - p;
    float2 bp = b - p;
    float2 cp = c - p;

    float a2 = dot(ap, ap);
    float b2 = dot(bp, bp);
    float c2 = dot(cp, cp);

    return ap.x * (bp.y * c2 - b2 * cp.y) -
           ap.y * (bp.x * c2 - b2 * cp.x) +
           a2   * (bp.x * cp.y - bp.y * cp.x);
}

static int Next(int he)          { return _HalfEdges[he].next; }
static int Prev(int he)          { return Next(Next(he)); }
static int Dest(int he)          { return _HalfEdges[Next(he)].v; }
static void UnlockTri(int t)     { _TriLocks[t] = 0; }

static void MarkDirtyVertex(int v)
{
    if ((uint)v >= (uint)_RealVertexCount) return;
    InterlockedOr(_DirtyVertexFlags[v], 1u);
}

static bool IsEdgeLengthValid(int a, int b)
{
    float2 pa = _Positions[a];
    float2 pb = _Positions[b];
    float2 d = pb - pa;
    float len2 = dot(d, d);
    if (len2 <= 1e-12f) return false;
    if (_SupportRadius2 > 0.0f && len2 > _SupportRadius2) return false;
    return true;
}

static bool TryLockTwo(int t0, int t1, int owner)
{
    int prev0;
    InterlockedCompareExchange(_TriLocks[t0], 0, owner, prev0);
    if (prev0 != 0) return false;

    int prev1;
    InterlockedCompareExchange(_TriLocks[t1], 0, owner, prev1);
    if (prev1 != 0)
    {
        UnlockTri(t0);
        return false;
    }
    return true;
}

static void FlipDiagonal(int he0, int he1)
{
    int t0 = _HalfEdges[he0].t;
    int t1 = _HalfEdges[he1].t;

    int he0n = Next(he0);
    int he0p = Next(he0n);
    int he1n = Next(he1);
    int he1p = Next(he1n);

    int c = Dest(he0n);
    int d = Dest(he1n);

    _HalfEdges[he0].v = c;
    _HalfEdges[he1].v = d;

    _HalfEdges[he0].twin = he1;
    _HalfEdges[he1].twin = he0;

    _HalfEdges[he0].t = t0;
    _HalfEdges[he1p].t = t0;
    _HalfEdges[he0n].t = t0;

    _HalfEdges[he1].t = t1;
    _HalfEdges[he0p].t = t1;
    _HalfEdges[he1n].t = t1;

    _HalfEdges[he0].next = he1p;
    _HalfEdges[he1p].next = he0n;
    _HalfEdges[he0n].next = he0;

    _HalfEdges[he1].next = he0p;
    _HalfEdges[he0p].next = he1n;
    _HalfEdges[he1n].next = he1;
}

// -------------------------
// Existing maintenance kernels
// -------------------------

[numthreads(256,1,1)]
void FixHalfEdges(uint3 id : SV_DispatchThreadID)
{
    uint he0u = id.x;
    if (he0u >= (uint)_HalfEdgeCount) return;

    int he0 = (int)he0u;
    int he1 = _HalfEdges[he0].twin;
    if (he1 < 0 || he0 > he1) return;

    int t0 = _HalfEdges[he0].t;
    int t1 = _HalfEdges[he1].t;
    if (t0 < 0 || t1 < 0) return;

    HalfEdge e0 = _HalfEdges[he0];
    int he0n = e0.next;
    HalfEdge e0n = _HalfEdges[he0n];
    int he0p = e0n.next;

    HalfEdge e1 = _HalfEdges[he1];
    int he1n = e1.next;
    HalfEdge e1n = _HalfEdges[he1n];
    int he1p = e1n.next;

    int a = e0.v;
    int b = e0n.v;
    int c = _HalfEdges[he0p].v;
    int d = _HalfEdges[he1p].v;

    if ((uint)a >= (uint)_VertexCount || (uint)b >= (uint)_VertexCount ||
        (uint)c >= (uint)_VertexCount || (uint)d >= (uint)_VertexCount)
        return;

    float2 pa = _Positions[a];
    float2 pb = _Positions[b];
    float2 pc = _Positions[c];
    float2 pd = _Positions[d];

    float o0 = Orient2D(pa, pb, pc);
    float o1 = Orient2D(pb, pa, pd);
    bool inv0 = o0 <= 0.0;
    bool inv1 = o1 <= 0.0;
    if (inv0 == inv1) return;

    float sD = Orient2D(pa, pb, pd);
    if (o0 * sD >= 0.0) return;

    int owner = he0 + 1;
    if (!TryLockTwo(t0, t1, owner)) return;

    if (_HalfEdges[he0].twin != he1 || _HalfEdges[he1].twin != he0)
    {
        UnlockTri(t0);
        UnlockTri(t1);
        return;
    }

    e0 = _HalfEdges[he0];
    he0n = e0.next;
    e0n = _HalfEdges[he0n];
    he0p = e0n.next;

    e1 = _HalfEdges[he1];
    he1n = e1.next;
    e1n = _HalfEdges[he1n];
    he1p = e1n.next;

    a = e0.v;
    b = e0n.v;
    c = _HalfEdges[he0p].v;
    d = _HalfEdges[he1p].v;

    if ((uint)a >= (uint)_VertexCount || (uint)b >= (uint)_VertexCount ||
        (uint)c >= (uint)_VertexCount || (uint)d >= (uint)_VertexCount)
    {
        UnlockTri(t0);
        UnlockTri(t1);
        return;
    }

    pa = _Positions[a];
    pb = _Positions[b];
    pc = _Positions[c];
    pd = _Positions[d];

    o0 = Orient2D(pa, pb, pc);
    o1 = Orient2D(pb, pa, pd);
    inv0 = o0 <= 0.0;
    inv1 = o1 <= 0.0;
    if (inv0 == inv1)
    {
        UnlockTri(t0);
        UnlockTri(t1);
        return;
    }

    sD = Orient2D(pa, pb, pd);
    if (o0 * sD >= 0.0)
    {
        UnlockTri(t0);
        UnlockTri(t1);
        return;
    }

    MarkDirtyVertex(a);
    MarkDirtyVertex(b);
    MarkDirtyVertex(c);
    MarkDirtyVertex(d);

    FlipDiagonal(he0, he1);
    InterlockedAdd(_FlipCount[0], 1);

    UnlockTri(t0);
    UnlockTri(t1);
}

[numthreads(256,1,1)]
void LegalizeHalfEdges(uint3 id : SV_DispatchThreadID)
{
    uint he0u = id.x;
    if (he0u >= (uint)_HalfEdgeCount) return;

    int he0 = (int)he0u;
    int he1 = _HalfEdges[he0].twin;
    if (he1 < 0 || he0 > he1) return;

    int t0 = _HalfEdges[he0].t;
    int t1 = _HalfEdges[he1].t;
    if (t0 < 0 || t1 < 0) return;

    HalfEdge e0 = _HalfEdges[he0];
    int he0n = e0.next;
    HalfEdge e0n = _HalfEdges[he0n];
    int he0p = e0n.next;

    HalfEdge e1 = _HalfEdges[he1];
    int he1n = e1.next;
    HalfEdge e1n = _HalfEdges[he1n];
    int he1p = e1n.next;

    int a = e0.v;
    int b = e0n.v;
    int c = _HalfEdges[he0p].v;
    int d = _HalfEdges[he1p].v;

    if ((uint)a >= (uint)_VertexCount || (uint)b >= (uint)_VertexCount ||
        (uint)c >= (uint)_VertexCount || (uint)d >= (uint)_VertexCount)
        return;

    float2 pa = _Positions[a];
    float2 pb = _Positions[b];
    float2 pc = _Positions[c];
    float2 pd = _Positions[d];

    float o = Orient2D(pa, pb, pc);
    float sD = Orient2D(pa, pb, pd);
    if (abs(o) <= 1e-10 || abs(sD) <= 1e-10) return;
    if (o * sD >= 0.0) return;

    float det = InCircleDet(pa, pb, pc, pd);
    if (o < 0.0) det = -det;
    if (det <= 1e-12) return;

    int owner = he0 + 1;
    if (!TryLockTwo(t0, t1, owner)) return;

    if (_HalfEdges[he0].twin != he1 || _HalfEdges[he1].twin != he0)
    {
        UnlockTri(t0);
        UnlockTri(t1);
        return;
    }

    e0 = _HalfEdges[he0];
    he0n = e0.next;
    e0n = _HalfEdges[he0n];
    he0p = e0n.next;

    e1 = _HalfEdges[he1];
    he1n = e1.next;
    e1n = _HalfEdges[he1n];
    he1p = e1n.next;

    a = e0.v;
    b = e0n.v;
    c = _HalfEdges[he0p].v;
    d = _HalfEdges[he1p].v;

    if ((uint)a >= (uint)_VertexCount || (uint)b >= (uint)_VertexCount ||
        (uint)c >= (uint)_VertexCount || (uint)d >= (uint)_VertexCount)
    {
        UnlockTri(t0);
        UnlockTri(t1);
        return;
    }

    pa = _Positions[a];
    pb = _Positions[b];
    pc = _Positions[c];
    pd = _Positions[d];

    o = Orient2D(pa, pb, pc);
    sD = Orient2D(pa, pb, pd);

    if (abs(o) <= 1e-10 || abs(sD) <= 1e-10)
    {
        UnlockTri(t0);
        UnlockTri(t1);
        return;
    }
    if (o * sD >= 0.0)
    {
        UnlockTri(t0);
        UnlockTri(t1);
        return;
    }

    det = InCircleDet(pa, pb, pc, pd);
    if (o < 0.0) det = -det;

    if (det > 1e-12)
    {
        MarkDirtyVertex(a);
        MarkDirtyVertex(b);
        MarkDirtyVertex(c);
        MarkDirtyVertex(d);

        FlipDiagonal(he0, he1);
        InterlockedAdd(_FlipCount[0], 1);
    }

    UnlockTri(t0);
    UnlockTri(t1);
}

// -------------------------
// Rebuild helpers
// -------------------------

static int GridIndex(int2 c)
{
    return c.y * _GridW + c.x;
}

static bool GridInside(int2 c)
{
    return (uint)c.x < (uint)_GridW && (uint)c.y < (uint)_GridH;
}

static float4 GetRebuildBoundsRaw()
{
    return _BoundsResult[0];
}

static float2 GetRebuildMin()
{
    float4 b = GetRebuildBoundsRaw();
    float2 mn = b.xy;
    float2 mx = b.zw;
    float2 ext = max(mx - mn, float2(1e-4, 1e-4));
    return mn - ext * _RebuildPadding;
}

static float2 GetRebuildMax()
{
    float4 b = GetRebuildBoundsRaw();
    float2 mn = b.xy;
    float2 mx = b.zw;
    float2 ext = max(mx - mn, float2(1e-4, 1e-4));
    return mx + ext * _RebuildPadding;
}

static float2 GetCellCenterWorld(int2 c)
{
    float2 mn = GetRebuildMin();
    float2 mx = GetRebuildMax();
    float2 size = max(mx - mn, float2(1e-4, 1e-4));
    float2 uv = (float2(c) + 0.5) / float2(_GridW, _GridH);
    return mn + uv * size;
}

static int2 WorldToCell(float2 p)
{
    float2 mn = GetRebuildMin();
    float2 mx = GetRebuildMax();
    float2 size = max(mx - mn, float2(1e-4, 1e-4));
    float2 uv = saturate((p - mn) / size);
    int2 c = int2(uv * float2(_GridW, _GridH));
    c = clamp(c, int2(0, 0), int2(_GridW - 1, _GridH - 1));
    return c;
}

static uint Hash32(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

static uint HashEdge(uint a, uint b)
{
    return Hash32(a * 0x9e3779b9u ^ (b + 0x85ebca6bu));
}

static uint HashTri(uint a, uint b, uint c)
{
    return Hash32(a * 73856093u ^ b * 19349663u ^ c * 83492791u);
}

static void Sort3(inout uint a, inout uint b, inout uint c)
{
    if (a > b) { uint t = a; a = b; b = t; }
    if (b > c) { uint t = b; b = c; c = t; }
    if (a > b) { uint t = a; a = b; b = t; }
}

static uint3 RotateToMinCCW(uint a, uint b, uint c)
{
    if (a <= b && a <= c) return uint3(a, b, c);
    if (b <= a && b <= c) return uint3(b, c, a);
    return uint3(c, a, b);
}

static bool TriangleDistinct(uint a, uint b, uint c)
{
    return a != b && b != c && c != a;
}

static bool TriangleEdgesValid(uint a, uint b, uint c)
{
    if (_UseSupportRadiusFilter == 0) return true;
    return IsEdgeLengthValid((int)a, (int)b) &&
           IsEdgeLengthValid((int)b, (int)c) &&
           IsEdgeLengthValid((int)c, (int)a);
}

static bool CanonicalCCWTriangle(int a, int b, int c, out uint3 tri)
{
    tri = uint3(0, 0, 0);

    if ((uint)a >= (uint)_VertexCount || (uint)b >= (uint)_VertexCount || (uint)c >= (uint)_VertexCount)
        return false;
    if (!TriangleDistinct((uint)a, (uint)b, (uint)c))
        return false;
    if (!TriangleEdgesValid((uint)a, (uint)b, (uint)c))
        return false;

    float2 pa = _Positions[a];
    float2 pb = _Positions[b];
    float2 pc = _Positions[c];

    float o = Orient2D(pa, pb, pc);
    if (abs(o) <= 1e-12) return false;
    if (o < 0.0)
    {
        int t = b; b = c; c = t;
    }

    tri = RotateToMinCCW((uint)a, (uint)b, (uint)c);
    return true;
}

static bool PointInTriStrict(float2 p, float2 a, float2 b, float2 c)
{
    float o = Orient2D(a, b, c);
    if (abs(o) <= 1e-14) return false;

    float s0 = Orient2D(a, b, p);
    float s1 = Orient2D(b, c, p);
    float s2 = Orient2D(c, a, p);

    if (o > 0.0)
        return (s0 > _InsideEps && s1 > _InsideEps && s2 > _InsideEps);
    else
        return (s0 < -_InsideEps && s1 < -_InsideEps && s2 < -_InsideEps);
}

static bool TryInsertTriangleHash(uint3 tri)
{
    uint h = HashTri(tri.x, tri.y, tri.z) % (uint)_TriHashSize;

    [loop]
    for (uint probe = 0u; probe < (uint)_TriHashSize; ++probe)
    {
        uint slot = (h + probe) % (uint)_TriHashSize;
        uint state = _TriHashState[slot];

        if (state == 2u)
        {
            if (_TriHashA[slot] == tri.x && _TriHashB[slot] == tri.y && _TriHashC[slot] == tri.z)
                return true;
            continue;
        }

        uint prev;
        InterlockedCompareExchange(_TriHashState[slot], 0u, 1u, prev);
        if (prev == 0u)
        {
            _TriHashA[slot] = tri.x;
            _TriHashB[slot] = tri.y;
            _TriHashC[slot] = tri.z;
            _TriHashState[slot] = 2u;
            return true;
        }

        if (prev == 2u)
        {
            if (_TriHashA[slot] == tri.x && _TriHashB[slot] == tri.y && _TriHashC[slot] == tri.z)
                return true;
        }
    }

    return false;
}

static void EmitTriangleCandidate(int a, int b, int c)
{
    uint3 tri;
    if (!CanonicalCCWTriangle(a, b, c, tri)) return;
    TryInsertTriangleHash(tri);
}

static void EmitQuadAsDelaunay(int s00, int s10, int s11, int s01)
{
    int ring[4];
    ring[0] = s00;
    ring[1] = s10;
    ring[2] = s11;
    ring[3] = s01;

    if ((uint)ring[0] >= (uint)_VertexCount || (uint)ring[1] >= (uint)_VertexCount ||
        (uint)ring[2] >= (uint)_VertexCount || (uint)ring[3] >= (uint)_VertexCount)
        return;

    float2 p0 = _Positions[ring[0]];
    float2 p1 = _Positions[ring[1]];
    float2 p2 = _Positions[ring[2]];
    float2 p3 = _Positions[ring[3]];

    float area = 0.0;
    area += p0.x * p1.y - p0.y * p1.x;
    area += p1.x * p2.y - p1.y * p2.x;
    area += p2.x * p3.y - p2.y * p3.x;
    area += p3.x * p0.y - p3.y * p0.x;

    if (area < 0.0)
    {
        int t = ring[1]; ring[1] = ring[3]; ring[3] = t;
        p1 = _Positions[ring[1]];
        p2 = _Positions[ring[2]];
        p3 = _Positions[ring[3]];
    }

    float o = Orient2D(_Positions[ring[0]], _Positions[ring[1]], _Positions[ring[2]]);
    if (abs(o) <= 1e-12) return;

    float det = InCircleDet(_Positions[ring[0]], _Positions[ring[1]], _Positions[ring[2]], _Positions[ring[3]]);
    if (o < 0.0) det = -det;

    if (det > 0.0)
    {
        EmitTriangleCandidate(ring[0], ring[1], ring[3]);
        EmitTriangleCandidate(ring[1], ring[2], ring[3]);
    }
    else
    {
        EmitTriangleCandidate(ring[0], ring[1], ring[2]);
        EmitTriangleCandidate(ring[0], ring[2], ring[3]);
    }
}

static bool TryInsertEdgeHash(uint src, uint dst, int he)
{
    uint h = HashEdge(src, dst) % (uint)_EdgeHashSize;

    [loop]
    for (uint probe = 0u; probe < (uint)_EdgeHashSize; ++probe)
    {
        uint slot = (h + probe) % (uint)_EdgeHashSize;
        uint state = _EdgeHashState[slot];

        if (state == 2u)
        {
            if (_EdgeHashSrc[slot] == src && _EdgeHashDst[slot] == dst)
                return true;
            continue;
        }

        uint prev;
        InterlockedCompareExchange(_EdgeHashState[slot], 0u, 1u, prev);
        if (prev == 0u)
        {
            _EdgeHashSrc[slot] = src;
            _EdgeHashDst[slot] = dst;
            _EdgeHashHE[slot] = he;
            _EdgeHashState[slot] = 2u;
            return true;
        }
    }

    return false;
}

static int FindEdgeHash(uint src, uint dst)
{
    uint h = HashEdge(src, dst) % (uint)_EdgeHashSize;

    [loop]
    for (uint probe = 0u; probe < (uint)_EdgeHashSize; ++probe)
    {
        uint slot = (h + probe) % (uint)_EdgeHashSize;
        uint state = _EdgeHashState[slot];

        if (state == 0u)
            return -1;

        if (state == 2u && _EdgeHashSrc[slot] == src && _EdgeHashDst[slot] == dst)
            return _EdgeHashHE[slot];
    }

    return -1;
}

// -------------------------
// Bounds reduction
// -------------------------

groupshared float2 gsMin[256];
groupshared float2 gsMax[256];

[numthreads(256,1,1)]
void BoundsReducePartials(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID, uint3 dtid : SV_DispatchThreadID)
{
    uint i = dtid.x;

    float2 mn = float2(1e30, 1e30);
    float2 mx = float2(-1e30, -1e30);

    if (i < (uint)_RealVertexCount)
    {
        float2 p = _Positions[i];
        mn = p;
        mx = p;
    }

    gsMin[gtid.x] = mn;
    gsMax[gtid.x] = mx;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = 128u; stride > 0u; stride >>= 1u)
    {
        if (gtid.x < stride)
        {
            gsMin[gtid.x] = min(gsMin[gtid.x], gsMin[gtid.x + stride]);
            gsMax[gtid.x] = max(gsMax[gtid.x], gsMax[gtid.x + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (gtid.x == 0u)
    {
        _BoundsPartials[gid.x] = float4(gsMin[0], gsMax[0]);
    }
}

[numthreads(256,1,1)]
void BoundsFinalize(uint3 dtid : SV_DispatchThreadID)
{
    if (dtid.x != 0u) return;

    float2 mn = float2(1e30, 1e30);
    float2 mx = float2(-1e30, -1e30);

    [loop]
    for (int i = 0; i < _BoundsPartialCount; ++i)
    {
        float4 b = _BoundsPartials[i];
        mn = min(mn, b.xy);
        mx = max(mx, b.zw);
    }

    _BoundsResult[0] = float4(mn, mx);
}

// -------------------------
// Clear / init kernels
// -------------------------

[numthreads(256,1,1)]
void ClearRebuildGrid(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    uint n = (uint)(_GridW * _GridH);
    if (i >= n) return;

    _GridSeedOwner[i] = 0x7fffffff;
    _VoronoiA[i] = -1;
    _VoronoiB[i] = -1;
}

[numthreads(256,1,1)]
void ClearTriangleHash(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= (uint)_TriHashSize) return;

    _TriHashA[i] = 0u;
    _TriHashB[i] = 0u;
    _TriHashC[i] = 0u;
    _TriHashState[i] = 0u;
}

[numthreads(256,1,1)]
void ClearEdgeHash(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= (uint)_EdgeHashSize) return;

    _EdgeHashSrc[i] = 0u;
    _EdgeHashDst[i] = 0u;
    _EdgeHashHE[i] = -1;
    _EdgeHashState[i] = 0u;
}

[numthreads(256,1,1)]
void ClearMeshState(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;

    if (i < (uint)_VertexCount)
    {
        _VToE[i] = -1;
        _SiteSeenInTri[i] = 0u;
        _OwnerByVertex[i] = -1;
        if (i < (uint)_RealVertexCount)
        {
            _NeighborCounts[i] = 0;
            _DirtyVertexFlags[i] = 0u;
            _BoundaryNormals[i] = float2(0.0, 0.0);
        }
    }

    if (i < (uint)_MaxTriangles)
    {
        _TriToHE[i] = -1;
        _TriToHEAll[i] = -1;
        _TriLocks[i] = 0;
        _TriInternal[i] = 0u;
    }

    if (i < (uint)_MaxHalfEdges)
    {
        _BoundaryEdgeFlags[i] = 0u;
        _HalfEdges[i].v = -1;
        _HalfEdges[i].next = -1;
        _HalfEdges[i].twin = -1;
        _HalfEdges[i].t = -1;
    }

    if (i < (uint)_MaxMissingVertices)
    {
        _MissingSites[i] = -1;
    }

    if (i == 0u)
    {
        _Debug[0]  = 0u;
        _FlipCount[0] = 0u;
        _RebuildCounters[CTR_TRI_COUNT] = 0u;
        _RebuildCounters[CTR_HE_USED] = 0u;
        _RebuildCounters[CTR_MISSING_COUNT] = 0u;
        _RebuildCounters[CTR_TRI_USED] = 0u;
        _RebuildCounters[CTR_INSERTED_COUNT] = 0u;
    }
}

// -------------------------
// Site seeding
// -------------------------

[numthreads(256,1,1)]
void SeedSitesToGrid(uint3 id : SV_DispatchThreadID)
{
    uint v = id.x;
    if (v >= (uint)_VertexCount) return;

    int2 c = WorldToCell(_Positions[v]);
    int gi = GridIndex(c);

    int oldVal;
    InterlockedMin(_GridSeedOwner[gi], (int)v, oldVal);
}

[numthreads(256,1,1)]
void AssignOwnersByCell(uint3 id : SV_DispatchThreadID)
{
    uint v = id.x;
    if (v >= (uint)_VertexCount) return;

    int2 c = WorldToCell(_Positions[v]);
    int gi = GridIndex(c);
    _OwnerByVertex[v] = _GridSeedOwner[gi];
}

[numthreads(256,1,1)]
void InitVoronoiFromSeeds(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    uint n = (uint)(_GridW * _GridH);
    if (i >= n) return;

    int s = _GridSeedOwner[i];
    if (s == 0x7fffffff) s = -1;

    _VoronoiA[i] = s;
    _VoronoiB[i] = s;
}

// -------------------------
// Jump flooding
// Host runs these with steps: maxPow2, ..., 2, 1
// -------------------------

int ChooseBetterSiteAtCell(int2 cell, int curSite, int candSite)
{
    if (candSite < 0) return curSite;
    if ((uint)candSite >= (uint)_VertexCount) return curSite;
    if (curSite < 0) return candSite;

    float2 wp = GetCellCenterWorld(cell);
    float2 pc = _Positions[curSite];
    float2 pn = _Positions[candSite];
    float dc = dot(pc - wp, pc - wp);
    float dn = dot(pn - wp, pn - wp);

    if (dn < dc) return candSite;
    if (dn > dc) return curSite;
    return min(curSite, candSite);
}

[numthreads(8,8,1)]
void JumpFloodAtoB(uint3 id : SV_DispatchThreadID)
{
    int2 c = int2(id.xy);
    if (!GridInside(c)) return;

    int idx = GridIndex(c);
    int best = _VoronoiA[idx];

    [unroll]
    for (int oy = -1; oy <= 1; ++oy)
    {
        [unroll]
        for (int ox = -1; ox <= 1; ++ox)
        {
            int2 n = c + int2(ox, oy) * _NeighborCount;
            if (!GridInside(n)) continue;
            int cand = _VoronoiA[GridIndex(n)];
            best = ChooseBetterSiteAtCell(c, best, cand);
        }
    }

    _VoronoiB[idx] = best;
}

[numthreads(8,8,1)]
void JumpFloodBtoA(uint3 id : SV_DispatchThreadID)
{
    int2 c = int2(id.xy);
    if (!GridInside(c)) return;

    int idx = GridIndex(c);
    int best = _VoronoiB[idx];

    [unroll]
    for (int oy = -1; oy <= 1; ++oy)
    {
        [unroll]
        for (int ox = -1; ox <= 1; ++ox)
        {
            int2 n = c + int2(ox, oy) * _NeighborCount;
            if (!GridInside(n)) continue;
            int cand = _VoronoiB[GridIndex(n)];
            best = ChooseBetterSiteAtCell(c, best, cand);
        }
    }

    _VoronoiA[idx] = best;
}

// -------------------------
// Island cleanup
// Run a few iterations after JFA, alternating A->B and B->A
// -------------------------

[numthreads(8,8,1)]
void RemoveIslandsAtoB(uint3 id : SV_DispatchThreadID)
{
    int2 c = int2(id.xy);
    if (!GridInside(c)) return;

    int idx = GridIndex(c);
    int s = _VoronoiA[idx];
    if (s < 0)
    {
        _VoronoiB[idx] = -1;
        return;
    }

    int2 siteCell = WorldToCell(_Positions[s]);
    int2 d = siteCell - c;

    int2 q0 = int2(0, 0);
    int2 q1 = int2(0, 0);
    int2 q2 = int2(0, 0);

    if (d.x > 0 && d.y > 0)      { q0 = int2(1, 0); q1 = int2(1, 1); q2 = int2(0, 1); }
    else if (d.x < 0 && d.y > 0) { q0 = int2(-1, 0); q1 = int2(-1, 1); q2 = int2(0, 1); }
    else if (d.x < 0 && d.y < 0) { q0 = int2(-1, 0); q1 = int2(-1, -1); q2 = int2(0, -1); }
    else if (d.x > 0 && d.y < 0) { q0 = int2(1, 0); q1 = int2(1, -1); q2 = int2(0, -1); }
    else if (d.x > 0)            { q0 = int2(1, 0); }
    else if (d.x < 0)            { q0 = int2(-1, 0); }
    else if (d.y > 0)            { q0 = int2(0, 1); }
    else if (d.y < 0)            { q0 = int2(0, -1); }

    int best = s;
    bool island = false;

    if (q1.x == 0 && q1.y == 0 && q2.x == 0 && q2.y == 0)
    {
        int2 n = c + q0;
        if (GridInside(n))
        {
            int sn = _VoronoiA[GridIndex(n)];
            island = (sn != s);
            best = ChooseBetterSiteAtCell(c, best, sn);
        }
    }
    else
    {
        int diffCount = 0;
        int2 nn[3] = { c + q0, c + q1, c + q2 };
        [unroll]
        for (int i = 0; i < 3; ++i)
        {
            if (!GridInside(nn[i])) continue;
            int sn = _VoronoiA[GridIndex(nn[i])];
            if (sn != s) diffCount++;
            best = ChooseBetterSiteAtCell(c, best, sn);
        }
        island = (diffCount == 3);
    }

    _VoronoiB[idx] = island ? best : s;
}

[numthreads(8,8,1)]
void RemoveIslandsBtoA(uint3 id : SV_DispatchThreadID)
{
    int2 c = int2(id.xy);
    if (!GridInside(c)) return;

    int idx = GridIndex(c);
    int s = _VoronoiB[idx];
    if (s < 0)
    {
        _VoronoiA[idx] = -1;
        return;
    }

    int2 siteCell = WorldToCell(_Positions[s]);
    int2 d = siteCell - c;

    int2 q0 = int2(0, 0);
    int2 q1 = int2(0, 0);
    int2 q2 = int2(0, 0);

    if (d.x > 0 && d.y > 0)      { q0 = int2(1, 0); q1 = int2(1, 1); q2 = int2(0, 1); }
    else if (d.x < 0 && d.y > 0) { q0 = int2(-1, 0); q1 = int2(-1, 1); q2 = int2(0, 1); }
    else if (d.x < 0 && d.y < 0) { q0 = int2(-1, 0); q1 = int2(-1, -1); q2 = int2(0, -1); }
    else if (d.x > 0 && d.y < 0) { q0 = int2(1, 0); q1 = int2(1, -1); q2 = int2(0, -1); }
    else if (d.x > 0)            { q0 = int2(1, 0); }
    else if (d.x < 0)            { q0 = int2(-1, 0); }
    else if (d.y > 0)            { q0 = int2(0, 1); }
    else if (d.y < 0)            { q0 = int2(0, -1); }

    int best = s;
    bool island = false;

    if (q1.x == 0 && q1.y == 0 && q2.x == 0 && q2.y == 0)
    {
        int2 n = c + q0;
        if (GridInside(n))
        {
            int sn = _VoronoiB[GridIndex(n)];
            island = (sn != s);
            best = ChooseBetterSiteAtCell(c, best, sn);
        }
    }
    else
    {
        int diffCount = 0;
        int2 nn[3] = { c + q0, c + q1, c + q2 };
        [unroll]
        for (int i = 0; i < 3; ++i)
        {
            if (!GridInside(nn[i])) continue;
            int sn = _VoronoiB[GridIndex(nn[i])];
            if (sn != s) diffCount++;
            best = ChooseBetterSiteAtCell(c, best, sn);
        }
        island = (diffCount == 3);
    }

    _VoronoiA[idx] = island ? best : s;
}

// -------------------------
// Triangle extraction from final VoronoiA
// -------------------------

[numthreads(8,8,1)]
void ExtractTrianglesFromVoronoi(uint3 id : SV_DispatchThreadID)
{
    int2 c = int2(id.xy);
    if (c.x + 1 >= _GridW || c.y + 1 >= _GridH) return;

    int s00 = _VoronoiA[GridIndex(c)];
    int s10 = _VoronoiA[GridIndex(c + int2(1, 0))];
    int s01 = _VoronoiA[GridIndex(c + int2(0, 1))];
    int s11 = _VoronoiA[GridIndex(c + int2(1, 1))];

    int labels[4] = { s00, s10, s01, s11 };
    int uniq[4];
    int n = 0;

    [unroll]
    for (int i = 0; i < 4; ++i)
    {
        int s = labels[i];
        if (s < 0) continue;
        bool found = false;
        [unroll]
        for (int j = 0; j < n; ++j)
            found = found | (uniq[j] == s);
        if (!found) uniq[n++] = s;
    }

    if (n < 3) return;

    if (n == 3)
    {
        bool checker =
            (s00 == s11 && s00 != s10 && s00 != s01 && s10 != s01) ||
            (s10 == s01 && s10 != s00 && s10 != s11 && s00 != s11);

        if (!checker)
            EmitTriangleCandidate(uniq[0], uniq[1], uniq[2]);
    }
    else
    {
        EmitQuadAsDelaunay(s00, s10, s11, s01);
    }
}

[numthreads(256,1,1)]
void CompactTrianglesFromHash(uint3 id : SV_DispatchThreadID)
{
    uint slot = id.x;
    if (slot >= (uint)_TriHashSize) return;
    if (_TriHashState[slot] != 2u) return;

    uint outIdx;
    InterlockedAdd(_RebuildCounters[CTR_TRI_COUNT], 1u, outIdx);
    if (outIdx >= (uint)_MaxTriangles) return;

    _TriRaw[outIdx] = uint3(_TriHashA[slot], _TriHashB[slot], _TriHashC[slot]);
}

[numthreads(256,1,1)]
void InitAllocatorsFromTriCount(uint3 id : SV_DispatchThreadID)
{
    if (id.x != 0u) return;

    uint triCount = _RebuildCounters[CTR_TRI_COUNT];
    _RebuildCounters[CTR_TRI_USED] = triCount;
    _RebuildCounters[CTR_HE_USED] = triCount * 3u;
}


// -------------------------
// Shared-edge conflict prune helpers
// -------------------------
static void WriteEdgeRecord(uint recIdx, uint u, uint v, int tri, int opp)
{
    uint a = min(u, v);
    uint b = max(u, v);

    _EdgeRecA[recIdx] = a;
    _EdgeRecB[recIdx] = b;
    _EdgeRecHash[recIdx] = HashEdge(a, b);
    _EdgeRecTri[recIdx] = tri;
    _EdgeRecOpp[recIdx] = opp;
}

static void RejectTri(uint t)
{
    if ((int)t < 0) return;
    _TriReject[t] = 1u;
    InterlockedAdd(_Debug[0], 1u);
}

// -------------------------
// Clear kernels
// -------------------------

[numthreads(256,1,1)]
void ClearTriangleRejectFlags(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= (uint)_MaxTriangles) return;
    _TriReject[i] = 0u;
}

[numthreads(256,1,1)]
void ClearEdgeRecords(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= (uint)_MaxEdgeRecords) return;

    _EdgeRecHash[i] = 0xffffffffu;
    _EdgeRecA[i] = 0u;
    _EdgeRecB[i] = 0u;
    _EdgeRecTri[i] = -1;
    _EdgeRecOpp[i] = -1;
}

[numthreads(256,1,1)]
void ResetFilteredTriCounter(uint3 id : SV_DispatchThreadID)
{
    if (id.x != 0u) return;
    _RebuildCounters[CTR_TRI_FILTERED] = 0u;
}

// -------------------------
// Emit 3 edge records per triangle
// -------------------------

[numthreads(256,1,1)]
void EmitTriangleEdgeRecords(uint3 id : SV_DispatchThreadID)
{
    uint t = id.x;
    uint triCount = _RebuildCounters[CTR_TRI_COUNT];
    if (t >= triCount) return;

    uint3 tri = _TriRaw[t];
    uint a = tri.x;
    uint b = tri.y;
    uint c = tri.z;

    uint baseIdx = 3u * t;
    if (baseIdx + 2u >= (uint)_MaxEdgeRecords) return;

    if (!TriangleDistinct(a, b, c)) { _TriReject[t] = 1u; return; }
    if (a >= (uint)_VertexCount || b >= (uint)_VertexCount || c >= (uint)_VertexCount) { _TriReject[t] = 1u; return; }

    float2 pa = _Positions[a];
    float2 pb = _Positions[b];
    float2 pc = _Positions[c];
    float o = Orient2D(pa, pb, pc);
    if (abs(o) <= 1e-12f) { _TriReject[t] = 1u; return; }

    WriteEdgeRecord(baseIdx + 0u, a, b, (int)t, (int)c);
    WriteEdgeRecord(baseIdx + 1u, b, c, (int)t, (int)a);
    WriteEdgeRecord(baseIdx + 2u, c, a, (int)t, (int)b);
}

// -------------------------
// In-place bitonic sort by edge hash
// -------------------------

[numthreads(256,1,1)]
void SortEdgeRecordsBitonic(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= (uint)_MaxEdgeRecords) return;

    uint j = (uint)_SortJ;
    uint k = (uint)_SortK;
    uint ixj = i ^ j;
    if (ixj <= i) return;

    uint hi = _EdgeRecHash[i];
    uint hj = _EdgeRecHash[ixj];

    bool ascending = ((i & k) == 0u);
    bool swap = ascending ? (hi > hj) : (hi < hj);
    if (!swap) return;

    uint ai = _EdgeRecA[i];
    uint bi = _EdgeRecB[i];
    int triI = _EdgeRecTri[i];
    int oppI = _EdgeRecOpp[i];

    _EdgeRecHash[i] = _EdgeRecHash[ixj];
    _EdgeRecA[i] = _EdgeRecA[ixj];
    _EdgeRecB[i] = _EdgeRecB[ixj];
    _EdgeRecTri[i] = _EdgeRecTri[ixj];
    _EdgeRecOpp[i] = _EdgeRecOpp[ixj];

    _EdgeRecHash[ixj] = hi;
    _EdgeRecA[ixj] = ai;
    _EdgeRecB[ixj] = bi;
    _EdgeRecTri[ixj] = triI;
    _EdgeRecOpp[ixj] = oppI;
}

// -------------------------
// Reject from sorted edge records
// -------------------------

[numthreads(256,1,1)]
void RejectTrianglesFromSortedEdgeRecords(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    uint triCount = _RebuildCounters[CTR_TRI_COUNT];
    uint recCount = triCount * 3u;
    if (i >= recCount) return;

    uint h = _EdgeRecHash[i];
    if (h == 0xffffffffu) return;

    if (i > 0u && _EdgeRecHash[i - 1u] == h) return;

    uint blockBegin = i;
    uint blockEnd = i + 1u;
    while (blockEnd < recCount && _EdgeRecHash[blockEnd] == h)
        ++blockEnd;

    for (uint j = blockBegin; j < blockEnd; ++j)
    {
        uint a = _EdgeRecA[j];
        uint b = _EdgeRecB[j];
        int triJ = _EdgeRecTri[j];
        int oppJ = _EdgeRecOpp[j];
        if (triJ < 0 || oppJ < 0) continue;

        bool firstExact = true;
        for (uint p = blockBegin; p < j; ++p)
        {
            if (_EdgeRecA[p] == a && _EdgeRecB[p] == b)
            {
                firstExact = false;
                break;
            }
        }
        if (!firstExact) continue;

        uint count = 0u;
        int t0 = -1, t1 = -1;
        int o0 = -1, o1 = -1;

        for (uint k = j; k < blockEnd; ++k)
        {
            if (_EdgeRecA[k] != a || _EdgeRecB[k] != b) continue;

            int triK = _EdgeRecTri[k];
            int oppK = _EdgeRecOpp[k];
            if (triK < 0 || oppK < 0) continue;

            if (count == 0u) { t0 = triK; o0 = oppK; }
            else if (count == 1u) { t1 = triK; o1 = oppK; }

            ++count;
        }

        if (count > 2u)
        {
            for (uint k = j; k < blockEnd; ++k)
            {
                if (_EdgeRecA[k] == a && _EdgeRecB[k] == b)
                {
                    int triK = _EdgeRecTri[k];
                    if (triK >= 0) RejectTri((uint)triK);
                }
            }
            continue;
        }

        if (count == 2u)
        {
            if ((uint)o0 >= (uint)_VertexCount || (uint)o1 >= (uint)_VertexCount)
            {
                RejectTri((uint)t0);
                RejectTri((uint)t1);
                continue;
            }

            float2 pa = _Positions[a];
            float2 pb = _Positions[b];
            float2 pc = _Positions[o0];
            float2 pd = _Positions[o1];

            float s0 = Orient2D(pa, pb, pc);
            float s1 = Orient2D(pa, pb, pd);

            if (abs(s0) <= 1e-12f || abs(s1) <= 1e-12f || s0 * s1 >= 0.0f)
            {
                RejectTri((uint)t0);
                RejectTri((uint)t1);
            }
        }
    }
}

// -------------------------
// Compact survivors back into _TriRaw
// -------------------------

[numthreads(256,1,1)]
void CompactValidTrianglesToTemp(uint3 id : SV_DispatchThreadID)
{
    uint t = id.x;
    uint triCount = _RebuildCounters[CTR_TRI_COUNT];
    if (t >= triCount) return;
    
    if (_TriReject[t] != 0u){
         InterlockedAdd(_Debug[0], 100u);
         return;
    }

    uint outIdx;
    InterlockedAdd(_RebuildCounters[CTR_TRI_FILTERED], 1u, outIdx);
    if (outIdx >= (uint)_MaxTriangles) return;

    _TriTemp[outIdx] = _TriRaw[t];
}

[numthreads(256,1,1)]
void CopyFilteredTrianglesBack(uint3 id : SV_DispatchThreadID)
{
    uint t = id.x;
    uint triCount = _RebuildCounters[CTR_TRI_FILTERED];
    if (t >= triCount) return;

    _TriRaw[t] = _TriTemp[t];
}

[numthreads(256,1,1)]
void FinalizeFilteredTriCount(uint3 id : SV_DispatchThreadID)
{
    if (id.x != 0u) return;

    uint triCount = _RebuildCounters[CTR_TRI_FILTERED];
    _RebuildCounters[CTR_TRI_COUNT] = triCount;
    _RebuildCounters[CTR_TRI_USED] = triCount;
    _RebuildCounters[CTR_HE_USED] = triCount * 3u;
}

// -------------------------
// Build halfedges from raw triangles
// -------------------------

[numthreads(256,1,1)]
void BuildHalfEdgesFromTriangles(uint3 id : SV_DispatchThreadID)
{
    uint t = id.x;
    uint triCount = _RebuildCounters[CTR_TRI_COUNT];
    if (t >= triCount) return;

    uint3 tri = _TriRaw[t];
    int baseHe = (int)(3u * t);

    _HalfEdges[baseHe + 0].v = (int)tri.x;
    _HalfEdges[baseHe + 1].v = (int)tri.y;
    _HalfEdges[baseHe + 2].v = (int)tri.z;

    _HalfEdges[baseHe + 0].next = baseHe + 1;
    _HalfEdges[baseHe + 1].next = baseHe + 2;
    _HalfEdges[baseHe + 2].next = baseHe + 0;

    _HalfEdges[baseHe + 0].twin = -1;
    _HalfEdges[baseHe + 1].twin = -1;
    _HalfEdges[baseHe + 2].twin = -1;

    _HalfEdges[baseHe + 0].t = (int)t;
    _HalfEdges[baseHe + 1].t = (int)t;
    _HalfEdges[baseHe + 2].t = (int)t;

    _TriToHE[t] = baseHe;
    _TriToHEAll[t] = baseHe;

    uint a = tri.x;
    uint b = tri.y;
    uint c = tri.z;
    _TriInternal[t] = (a < (uint)_RealVertexCount &&
                       b < (uint)_RealVertexCount &&
                       c < (uint)_RealVertexCount) ? 1u : 0u;
}

[numthreads(256,1,1)]
void BuildDirectedEdgeHash(uint3 id : SV_DispatchThreadID)
{
    uint he = id.x;
    uint heCount = _RebuildCounters[CTR_HE_USED];
    if (he >= heCount) return;

    int src = _HalfEdges[he].v;
    int dst = Dest((int)he);
    if (src < 0 || dst < 0) return;

    TryInsertEdgeHash((uint)src, (uint)dst, (int)he);
}

[numthreads(256,1,1)]
void ResolveTwinsFromEdgeHash(uint3 id : SV_DispatchThreadID)
{
    uint he = id.x;
    uint heCount = _RebuildCounters[CTR_HE_USED];
    if (he >= heCount) return;

    int src = _HalfEdges[he].v;
    int dst = Dest((int)he);
    if (src < 0 || dst < 0) return;

    int opp = FindEdgeHash((uint)dst, (uint)src);
    _HalfEdges[he].twin = opp;
}

[numthreads(256,1,1)]
void BuildVertexToEdgeAndBoundary(uint3 id : SV_DispatchThreadID)
{
    uint he = id.x;
    uint heCount = _RebuildCounters[CTR_HE_USED];
    if (he >= heCount) return;

    int a = _HalfEdges[he].v;
    int b = Dest((int)he);
    if (a >= 0)
    {
        int prev;
        InterlockedCompareExchange(_VToE[a], -1, (int)he, prev);
    }

    if (_HalfEdges[he].twin < 0)
    {
        _BoundaryEdgeFlags[he] = 1u;
    }
    else
    {
        _BoundaryEdgeFlags[he] = 0u;
    }
}

// -------------------------
// Missing-site recovery
// -------------------------

[numthreads(256,1,1)]
void MarkSeenVerticesFromTriangles(uint3 id : SV_DispatchThreadID)
{
    uint t = id.x;
    uint triCount = _RebuildCounters[CTR_TRI_COUNT];
    if (t >= triCount) return;

    uint3 tri = _TriRaw[t];
    if (tri.x < (uint)_VertexCount) _SiteSeenInTri[tri.x] = 1u;
    if (tri.y < (uint)_VertexCount) _SiteSeenInTri[tri.y] = 1u;
    if (tri.z < (uint)_VertexCount) _SiteSeenInTri[tri.z] = 1u;
}

[numthreads(256,1,1)]
void CollectMissingVertices(uint3 id : SV_DispatchThreadID)
{
    uint v = id.x;
    if (v >= (uint)_RealVertexCount) return;

    if (_SiteSeenInTri[v] != 0u) return;

    uint idx;
    InterlockedAdd(_RebuildCounters[CTR_MISSING_COUNT], 1u, idx);
    if (idx < (uint)_MaxMissingVertices)
        _MissingSites[idx] = (int)v;
}

static int FindContainingTriangleAroundOwner(int owner, float2 p, out int he0)
{
    he0 = -1;
    int start = _VToE[owner];
    if (start < 0) return -1;

    int he = start;
    [loop]
    for (int iter = 0; iter < _InsertionWalkLimit; ++iter)
    {
        int t = _HalfEdges[he].t;
        if (t >= 0)
        {
            int h0 = _TriToHEAll[t];
            if (h0 >= 0)
            {
                int h1 = _HalfEdges[h0].next;
                int h2 = _HalfEdges[h1].next;
                int a = _HalfEdges[h0].v;
                int b = _HalfEdges[h1].v;
                int c = _HalfEdges[h2].v;

                if ((uint)a < (uint)_VertexCount &&
                    (uint)b < (uint)_VertexCount &&
                    (uint)c < (uint)_VertexCount)
                {
                    if (PointInTriStrict(p, _Positions[a], _Positions[b], _Positions[c]))
                    {
                        he0 = h0;
                        return t;
                    }
                }
            }
        }

        int hp = Prev(he);
        int twin = _HalfEdges[hp].twin;
        if (twin < 0) break;
        he = twin;
        if (he == start) break;
    }

    return -1;
}

[numthreads(256,1,1)]
void InsertMissingVerticesSimple(uint3 id : SV_DispatchThreadID)
{
    uint idx = id.x;
    uint missCount = _RebuildCounters[CTR_MISSING_COUNT];
    if (idx >= missCount) return;

    int v = _MissingSites[idx];
    if ((uint)v >= (uint)_RealVertexCount) return;

    int owner = _OwnerByVertex[v];
    if ((uint)owner >= (uint)_VertexCount) return;

    float2 pv = _Positions[v];
    int triBaseHe;
    int t = FindContainingTriangleAroundOwner(owner, pv, triBaseHe);
    if (t < 0 || triBaseHe < 0) return;

    int lockPrev;
    InterlockedCompareExchange(_TriLocks[t], 0, v + 1, lockPrev);
    if (lockPrev != 0) return;

    triBaseHe = _TriToHEAll[t];
    if (triBaseHe < 0)
    {
        UnlockTri(t);
        return;
    }

    int h0 = triBaseHe;
    int h1 = _HalfEdges[h0].next;
    int h2 = _HalfEdges[h1].next;

    int a = _HalfEdges[h0].v;
    int b = _HalfEdges[h1].v;
    int c = _HalfEdges[h2].v;

    if ((uint)a >= (uint)_VertexCount ||
        (uint)b >= (uint)_VertexCount ||
        (uint)c >= (uint)_VertexCount)
    {
        UnlockTri(t);
        return;
    }

    if (!PointInTriStrict(pv, _Positions[a], _Positions[b], _Positions[c]))
    {
        UnlockTri(t);
        return;
    }

    uint oldTriUsed;
    uint oldHeUsed;
    InterlockedAdd(_RebuildCounters[CTR_TRI_USED], 2u, oldTriUsed);
    InterlockedAdd(_RebuildCounters[CTR_HE_USED], 6u, oldHeUsed);

    uint t1 = oldTriUsed;
    uint t2 = oldTriUsed + 1u;
    uint e0 = oldHeUsed + 0u; // b->v
    uint e1 = oldHeUsed + 1u; // v->a
    uint e2 = oldHeUsed + 2u; // c->v
    uint e3 = oldHeUsed + 3u; // v->b
    uint e4 = oldHeUsed + 4u; // a->v
    uint e5 = oldHeUsed + 5u; // v->c

    if (t2 >= (uint)_MaxTriangles || e5 >= (uint)_MaxHalfEdges)
    {
        UnlockTri(t);
        return;
    }

    int twin0 = _HalfEdges[h0].twin;
    int twin1 = _HalfEdges[h1].twin;
    int twin2 = _HalfEdges[h2].twin;

    // Triangle t: (a, b, v)
    _HalfEdges[h0].v = a;
    _HalfEdges[h0].next = (int)e0;
    _HalfEdges[h0].twin = twin0;
    _HalfEdges[h0].t = t;

    _HalfEdges[e0].v = b;
    _HalfEdges[e0].next = (int)e1;
    _HalfEdges[e0].twin = (int)e3;
    _HalfEdges[e0].t = t;

    _HalfEdges[e1].v = v;
    _HalfEdges[e1].next = h0;
    _HalfEdges[e1].twin = (int)e4;
    _HalfEdges[e1].t = t;

    // Triangle t1: (b, c, v)
    _HalfEdges[h1].v = b;
    _HalfEdges[h1].next = (int)e2;
    _HalfEdges[h1].twin = twin1;
    _HalfEdges[h1].t = (int)t1;

    _HalfEdges[e2].v = c;
    _HalfEdges[e2].next = (int)e3;
    _HalfEdges[e2].twin = (int)e5;
    _HalfEdges[e2].t = (int)t1;

    _HalfEdges[e3].v = v;
    _HalfEdges[e3].next = h1;
    _HalfEdges[e3].twin = (int)e0;
    _HalfEdges[e3].t = (int)t1;

    // Triangle t2: (c, a, v)
    _HalfEdges[h2].v = c;
    _HalfEdges[h2].next = (int)e4;
    _HalfEdges[h2].twin = twin2;
    _HalfEdges[h2].t = (int)t2;

    _HalfEdges[e4].v = a;
    _HalfEdges[e4].next = (int)e5;
    _HalfEdges[e4].twin = (int)e1;
    _HalfEdges[e4].t = (int)t2;

    _HalfEdges[e5].v = v;
    _HalfEdges[e5].next = h2;
    _HalfEdges[e5].twin = (int)e2;
    _HalfEdges[e5].t = (int)t2;

    _TriToHE[t] = h0;
    _TriToHEAll[t] = h0;
    _TriToHE[t1] = h1;
    _TriToHEAll[t1] = h1;
    _TriToHE[t2] = h2;
    _TriToHEAll[t2] = h2;

    _TriInternal[t]  = ((uint)a < (uint)_RealVertexCount && (uint)b < (uint)_RealVertexCount && (uint)v < (uint)_RealVertexCount) ? 1u : 0u;
    _TriInternal[t1] = ((uint)b < (uint)_RealVertexCount && (uint)c < (uint)_RealVertexCount && (uint)v < (uint)_RealVertexCount) ? 1u : 0u;
    _TriInternal[t2] = ((uint)c < (uint)_RealVertexCount && (uint)a < (uint)_RealVertexCount && (uint)v < (uint)_RealVertexCount) ? 1u : 0u;

    if (twin0 >= 0) _HalfEdges[twin0].twin = h0;
    if (twin1 >= 0) _HalfEdges[twin1].twin = h1;
    if (twin2 >= 0) _HalfEdges[twin2].twin = h2;

    MarkDirtyVertex(a);
    MarkDirtyVertex(b);
    MarkDirtyVertex(c);
    MarkDirtyVertex(v);

    _SiteSeenInTri[v] = 1u;
    InterlockedAdd(_RebuildCounters[CTR_INSERTED_COUNT], 1u);

    UnlockTri(t);
}

// -------------------------
// Final mesh metadata refresh after insertions
// Re-run ClearEdgeHash + BuildDirectedEdgeHash + ResolveTwinsFromEdgeHash + BuildVertexToEdgeAndBoundary
// -------------------------

[numthreads(256,1,1)]
void RecomputeTriToHECompact(uint3 id : SV_DispatchThreadID)
{
    uint t = id.x;
    uint triUsed = _RebuildCounters[CTR_TRI_USED];
    if (t >= triUsed) return;

    int he = _TriToHEAll[t];
    if (he < 0) return;
    _TriToHE[t] = he;
}

// -------------------------
// Optional cleanup kernels
// -------------------------

[numthreads(256,1,1)]
void ClearDirtyFlags(uint3 id : SV_DispatchThreadID)
{
    uint v = id.x;
    if (v >= (uint)_RealVertexCount) return;
    _DirtyVertexFlags[v] = 0u;
}

[numthreads(256,1,1)]
void ResetFlipCounter(uint3 id : SV_DispatchThreadID)
{
    if (id.x == 0u) _FlipCount[0] = 0u;
}

/*
Dispatch order for full scratch rebuild:

1.  BoundsReducePartials
2.  BoundsFinalize

3.  ClearRebuildGrid
4.  ClearTriangleHash
5.  ClearEdgeHash
6.  ClearMeshState

7.  SeedSitesToGrid
8.  AssignOwnersByCell
9.  InitVoronoiFromSeeds

10. JFA passes:
      set _NeighborCount = highestPow2 >= max(_GridW, _GridH)
      JumpFloodAtoB
      halve _NeighborCount
      JumpFloodBtoA
      ...
      until 1

11. Optional island cleanup, 2-4 alternating passes:
      RemoveIslandsAtoB
      RemoveIslandsBtoA

12. ExtractTrianglesFromVoronoi
13. CompactTrianglesFromHash
14. InitAllocatorsFromTriCount

15. ClearEdgeHash
16. BuildHalfEdgesFromTriangles
17. BuildDirectedEdgeHash
18. ResolveTwinsFromEdgeHash
19. BuildVertexToEdgeAndBoundary

20. MarkSeenVerticesFromTriangles
21. CollectMissingVertices

22. Optional several insertion rounds:
      InsertMissingVerticesSimple
      ClearEdgeHash
      ClearMeshState for VToE / boundary only if you split that out, or just rerun BuildVertexToEdgeAndBoundary after resetting arrays
      BuildDirectedEdgeHash
      ResolveTwinsFromEdgeHash
      BuildVertexToEdgeAndBoundary

23. Set host-side _TriCount      = _RebuildCounters[CTR_TRI_USED]
    Set host-side _HalfEdgeCrs[CTR_TRI_USED]
    Set host-_USED]
*/