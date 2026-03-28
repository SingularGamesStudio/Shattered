// Boundary classification and normal generation kernels.

groupshared float2 _BoundsReduceMin[256];
groupshared float2 _BoundsReduceMax[256];

[numthreads(256,1,1)]
void ClearBoundaryData(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i < (uint)_TriCount) {
        _TriToHEAll[i] = -1;
        _TriInternal[i] = 0u;
    }
    if (i < (uint)_HalfEdgeCount)
        _BoundaryEdgeFlags[i] = 0u;
    if (i < (uint)_RealVertexCount)
        _BoundaryNormals[i] = float2(0.0f, 0.0f);
}

[numthreads(256,1,1)]
void BuildTriToHEAll(uint3 id : SV_DispatchThreadID)
{
    uint he = id.x;
    if (he >= (uint)_HalfEdgeCount) return;

    int t = _HalfEdges[he].t;
    if (t < 0 || t >= _TriCount) return;

    int original;
    InterlockedCompareExchange(_TriToHEAll[t], -1, (int)he, original);
}

[numthreads(256,1,1)]
void ClassifyInternalTriangles(uint3 id : SV_DispatchThreadID)
{
    uint tu = id.x;
    if (tu >= (uint)_TriCount) return;

    int he0 = _TriToHEAll[tu];
    if (he0 < 0) {
        _TriInternal[tu] = 0u;
        return;
    }

    int he1 = Next(he0);
    int he2 = Next(he1);

    int a = _HalfEdges[he0].v;
    int b = _HalfEdges[he1].v;
    int c = _HalfEdges[he2].v;

    if ((uint)a >= (uint)_RealVertexCount ||
        (uint)b >= (uint)_RealVertexCount ||
        (uint)c >= (uint)_RealVertexCount) {
        _TriInternal[tu] = 0u;
        return;
    }

    int oa = _OwnerByVertex[a];
    int ob = _OwnerByVertex[b];
    int oc = _OwnerByVertex[c];
    if (!(oa == ob && ob == oc)) {
        _TriInternal[tu] = 0u;
        return;
    }

    bool valid = IsEdgeLengthValid(a, b) && IsEdgeLengthValid(b, c) && IsEdgeLengthValid(c, a);
    _TriInternal[tu] = valid ? 1u : 0u;
}

[numthreads(256,1,1)]
void MarkBoundaryEdges(uint3 id : SV_DispatchThreadID)
{
    uint he = id.x;
    if (he >= (uint)_HalfEdgeCount) return;

    int t = _HalfEdges[he].t;
    if (t < 0 || t >= _TriCount) return;
    if (_TriInternal[t] == 0u) return;

    int tw = _HalfEdges[he].twin;
    bool twinInternal = false;
    if (tw >= 0 && tw < _HalfEdgeCount) {
        int tt = _HalfEdges[tw].t;
        twinInternal = tt >= 0 && tt < _TriCount && _TriInternal[tt] != 0u;
    }

    if (!twinInternal) {
        _BoundaryEdgeFlags[he] = 1u;
        if (tw >= 0 && tw < _HalfEdgeCount)
            _BoundaryEdgeFlags[tw] = 1u;
    }
}

bool IsCanonicalBoundaryHE(int he)
{
    if (he < 0 || he >= _HalfEdgeCount) return false;

    int t = _HalfEdges[he].t;
    if (t < 0 || t >= _TriCount) return false;
    if (_TriInternal[t] == 0u) return false;

    int tw = _HalfEdges[he].twin;
    bool twInternal = false;
    if (tw >= 0 && tw < _HalfEdgeCount)
    {
        int tt = _HalfEdges[tw].t;
        twInternal = (tt >= 0 && tt < _TriCount && _TriInternal[tt] != 0u);
    }

    return !twInternal;
}

float2 BoundaryPos(int v)
{
    return _Positions[v];
    // If your SDF is built in predicted space, replace with predicted positions here.
}

bool CanonicalBoundaryNormal(int heCan, out float2 nOut)
{
    nOut = 0.0;

    if (!IsCanonicalBoundaryHE(heCan))
        return false;

    int he1 = Next(heCan);
    int he2 = Next(he1);

    int v0 = _HalfEdges[heCan].v;
    int v1 = _HalfEdges[he1].v;
    int vOpp = _HalfEdges[he2].v;

    if ((uint)v0 >= (uint)_RealVertexCount ||
        (uint)v1 >= (uint)_RealVertexCount ||
        (uint)vOpp >= (uint)_RealVertexCount)
        return false;

    float2 p0 = BoundaryPos(v0);
    float2 p1 = BoundaryPos(v1);
    float2 pOpp = BoundaryPos(vOpp);

    float2 e = p1 - p0;
    float e2 = dot(e, e);
    if (e2 <= 1e-12f)
        return false;

    float side = (p1.x - p0.x) * (pOpp.y - p0.y) - (p1.y - p0.y) * (pOpp.x - p0.x);

    float invLen = rsqrt(e2);
    float2 nL = float2(-e.y, e.x) * invLen;
    nOut = (side > 0.0f) ? -nL : nL;

    return true;
}

[numthreads(256,1,1)]
void BuildBoundaryNormals(uint3 id : SV_DispatchThreadID)
{
    uint v = id.x;
    if (v >= (uint)_RealVertexCount) return;

    int start = _VToE[v];
    if (start < 0)
    {
        _BoundaryNormals[v] = 0.0;
        return;
    }

    int heOut = start;
    float2 sum = 0.0;
    uint count = 0u;

    [loop]
    for (int iter = 0; iter < 128; ++iter)
    {
        int heCan = -1;

        if (IsCanonicalBoundaryHE(heOut))
        {
            heCan = heOut;
        }
        else
        {
            int tw = _HalfEdges[heOut].twin;
            if (tw >= 0 && tw < _HalfEdgeCount && IsCanonicalBoundaryHE(tw))
                heCan = tw;
        }

        if (heCan >= 0)
        {
            float2 nOut;
            if (CanonicalBoundaryNormal(heCan, nOut))
            {
                sum += nOut;
                count++;
            }
        }

        int twPrev = _HalfEdges[Prev(heOut)].twin;
        if (twPrev < 0) break;

        heOut = twPrev;
        if (heOut == start) break;
    }

    float l2 = dot(sum, sum);
    _BoundaryNormals[v] = (count > 0u && l2 > 1e-12f) ? sum * rsqrt(l2) : float2(0.0, 0.0);
}

[numthreads(256, 1, 1)]
void ReduceBounds(uint3 id : SV_DispatchThreadID, uint3 gid : SV_GroupID, uint3 gtid : SV_GroupThreadID)
{
    uint v = id.x;
    float2 pMin = float2(1e30, 1e30);
    float2 pMax = float2(-1e30, -1e30);

    if (v < (uint)_RealVertexCount)
    {
        float2 p = _Positions[v];
        pMin = p;
        pMax = p;
    }

    _BoundsReduceMin[gtid.x] = pMin;
    _BoundsReduceMax[gtid.x] = pMax;

    GroupMemoryBarrierWithGroupSync();

    [unroll]
    for (uint stride = 128u; stride > 0u; stride >>= 1u)
    {
        if (gtid.x < stride)
        {
            _BoundsReduceMin[gtid.x] = min(_BoundsReduceMin[gtid.x], _BoundsReduceMin[gtid.x + stride]);
            _BoundsReduceMax[gtid.x] = max(_BoundsReduceMax[gtid.x], _BoundsReduceMax[gtid.x + stride]);
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (gtid.x == 0u)
        _BoundsPartials[gid.x] = float4(_BoundsReduceMin[0], _BoundsReduceMax[0]);
}

[numthreads(1, 1, 1)]
void FinalizeBounds(uint3 id : SV_DispatchThreadID)
{
    float2 bMin = float2(1e30, 1e30);
    float2 bMax = float2(-1e30, -1e30);

    [loop]
    for (int i = 0; i < _BoundsPartialCount; i++)
    {
        float4 v = _BoundsPartials[i];
        bMin = min(bMin, v.xy);
        bMax = max(bMax, v.zw);
    }

    if (_RealVertexCount <= 0)
    {
        bMin = 0.0;
        bMax = 0.0;
    }

    _BoundsResult[0] = float4(bMin, bMax);
}
