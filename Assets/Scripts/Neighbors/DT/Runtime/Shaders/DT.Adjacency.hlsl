// Adjacency and mapping kernels.

[numthreads(256,1,1)]
void ClearTriLocks(uint3 id : SV_DispatchThreadID)
{
    uint t = id.x;
    if (t >= (uint)_TriCount) return;
    _TriLocks[t] = 0;
}

[numthreads(256,1,1)]
void ClearDirtyVertexFlags(uint3 id : SV_DispatchThreadID)
{
    uint v = id.x;
    if (v >= (uint)_RealVertexCount) return;
    _DirtyVertexFlags[v] = 0u;
}

[numthreads(256,1,1)]
void MarkAllDirty(uint3 id : SV_DispatchThreadID)
{
    uint v = id.x;
    if (v >= (uint)_RealVertexCount) return;
    _DirtyVertexFlags[v] = 1u;
}

[numthreads(256,1,1)]
void CopyHalfEdges(uint3 id : SV_DispatchThreadID)
{
    uint he = id.x;
    if (he >= (uint)_HalfEdgeCount) return;
    _HalfEdgesDst[he] = _HalfEdgesSrc[he];
}

[numthreads(256,1,1)]
void ClearVertexToEdge(uint3 id : SV_DispatchThreadID)
{
    uint v = id.x;
    if (v >= (uint)_VertexCount) return;
    _VToE[v] = -1;
}

[numthreads(256,1,1)]
void BuildVertexToEdge(uint3 id : SV_DispatchThreadID)
{
    uint he = id.x;
    if (he >= (uint)_HalfEdgeCount) return;

    HalfEdge e = _HalfEdges[he];
    if (e.t < 0) return;
    if ((uint)e.v >= (uint)_VertexCount) return;

    int original;
    InterlockedCompareExchange(_VToE[e.v], -1, (int)he, original);
}

[numthreads(256,1,1)]
void BuildNeighbors(uint3 id : SV_DispatchThreadID)
{
    uint v = id.x;
    if (v >= (uint)_RealVertexCount) return;

    int baseIdx = (int)v * _NeighborCount;
    for (int i = 0; i < _NeighborCount; i++)
    _Neighbors[baseIdx + i] = -1;
    _NeighborCounts[v] = 0;

    int start = _VToE[v];
    if (start < 0) return;

    int he = start;
    int count = 0;

    [loop] for (int iter = 0; iter < 128 && count < _NeighborCount; iter++)
    {
        int dst = Dest(he);
        if ((uint)dst >= (uint)_VertexCount) break;

        if (dst < _RealVertexCount)
        {
            if (_UseSupportRadiusFilter != 0) {
                float2 p0 = _Positions[v];
                float2 p1 = _Positions[dst];
                float2 d = p1 - p0;
                if (dot(d, d) > _SupportRadius2) { int twSkip = _HalfEdges[Prev(he)].twin; if (twSkip < 0) break; he = twSkip; if (he == start) break; continue; }
            }

            bool dup = false;
            for (int k = 0; k < count; k++)
            {
                if (_Neighbors[baseIdx + k] == dst)
                {
                    dup = true;
                    break;
                }
            }
            if (!dup)
            _Neighbors[baseIdx + count++] = dst;
        }

        int tw = _HalfEdges[Prev(he)].twin;
        if (tw < 0) break;
        he = tw;
        if (he == start) break;
    }

    _NeighborCounts[v] = count;
}

[numthreads(256,1,1)]
void ClearTriToHE(uint3 id : SV_DispatchThreadID)
{
    uint t = id.x;
    if (t >= (uint)_TriCount) return;
    _TriToHE[t] = -1;
}

[numthreads(256,1,1)]
void BuildRenderableTriToHE(uint3 id : SV_DispatchThreadID)
{
    uint heu = id.x;
    if (heu >= (uint)_HalfEdgeCount) return;

    int he = (int)heu;
    HalfEdge e = _HalfEdges[he];
    int t = e.t;
    if (t < 0 || t >= _TriCount) return;

    int a = e.v;
    int b = Dest(he);
    int c = Dest(Next(he));

    if ((uint)a >= (uint)_RealVertexCount ||
    (uint)b >= (uint)_RealVertexCount ||
    (uint)c >= (uint)_RealVertexCount) return;

    int original;
    InterlockedCompareExchange(_TriToHE[t], -1, he, original);
}
