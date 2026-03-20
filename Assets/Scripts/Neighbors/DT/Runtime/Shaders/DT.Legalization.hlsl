// Topology fix and Delaunay legalization kernels.

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
    if (!TryLockTwo(t0, t1, owner))
    return;

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
    if (!TryLockTwo(t0, t1, owner))
    return;

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
