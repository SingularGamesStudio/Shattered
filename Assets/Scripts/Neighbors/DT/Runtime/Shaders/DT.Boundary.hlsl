// Boundary classification and normal generation kernels.

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

[numthreads(256,1,1)]
void BuildBoundaryNormals(uint3 id : SV_DispatchThreadID)
{
    uint v = id.x;
    if (v >= (uint)_RealVertexCount) return;

    int start = _VToE[v];
    if (start < 0) {
        _BoundaryNormals[v] = float2(0.0f, 0.0f);
        return;
    }

    int he = start;
    float2 sum = float2(0.0f, 0.0f);

    [loop] for (int iter = 0; iter < 128; iter++) {
        if (_BoundaryEdgeFlags[he] != 0u) {
            int dst = Dest(he);
            if ((uint)dst < (uint)_RealVertexCount) {
                float2 pa = _Positions[v];
                float2 pb = _Positions[dst];
                float2 e = pb - pa;
                float len2 = dot(e, e);
                if (len2 > 1e-12f) {
                    float invLen = rsqrt(len2);
                    float2 n = float2(e.y, -e.x) * invLen;
                    sum += n;
                }
            }
        }

        int tw = _HalfEdges[Prev(he)].twin;
        if (tw < 0) break;
        he = tw;
        if (he == start) break;
    }

    float len2Sum = dot(sum, sum);
    _BoundaryNormals[v] = len2Sum > 1e-12f ? sum * rsqrt(len2Sum) : float2(0.0f, 0.0f);
}
