// Common types, buffers, constants, and helper functions for DT runtime compute kernels.

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
StructuredBuffer<int> _OwnerByVertex;

int _VertexCount;
int _RealVertexCount;
int _HalfEdgeCount;
int _TriCount;
int _NeighborCount;
int _UseSupportRadiusFilter;
float _SupportRadius2;

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
    a2  * (bp.x * cp.y - bp.y * cp.x);
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
