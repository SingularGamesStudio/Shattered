#ifndef XPBI_SOLVER_CACHE_KERNELS_INCLUDED
#define XPBI_SOLVER_CACHE_KERNELS_INCLUDED

[numthreads(256, 1, 1)] void ApplyGameplayForces(uint3 id : SV_DispatchThreadID)
{
    int ei = (int)id.x;
    if (ei >= _ForceEventCount)
        return;

    XPBI_ForceEvent e = _ForceEvents[ei];
    int gi = e.node;

    if (gi < _Base || gi >= _Base + _TotalCount)
        return;
    if (XPBI_IsFixed(gi))
        return;

    // Assumption: at most one event per node per tick (no contention on _Vel writes).
    float invM = _InvMass[gi];
    if (invM <= 0.0)
        return;

    _Vel[gi] = _Vel[gi] + (e.force * (invM * _Dt));
}

    [numthreads(256, 1, 1)] void ExternalForces(uint3 id : SV_DispatchThreadID)
{
    int li = (int)id.x;
    if (li >= _TotalCount)
        return;

    int gi = _Base + li;
    if (XPBI_IsFixed(gi))
        return;

    float2 v = _Vel[gi];
    v.y += _Gravity * _Dt;
    _Vel[gi] = v;
}

[numthreads(256, 1, 1)] void IntegratePositions(uint3 id : SV_DispatchThreadID)
{
    int li = (int)id.x;
    if (li >= _TotalCount)
        return;

    int gi = _Base + li;
    if (XPBI_IsFixed(gi))
        return;

    _Pos[gi] = _Pos[gi] + _Vel[gi] * _Dt;
}

    [numthreads(256, 1, 1)] void UpdateDtPositions(uint3 id : SV_DispatchThreadID)
{
    int li = (int)id.x;
    if (li >= _ActiveCount)
        return;

    int gi = _Base + li;
    _DtPositions[li] = (_Pos[gi] - _DtNormCenter) * _DtNormInvHalfExtent;
}

[numthreads(256, 1, 1)] void RebuildParentsAtLevel(uint3 id : SV_DispatchThreadID)
{
    int gi = _ParentRangeStart + (int)id.x;
    if (gi >= _ParentRangeEnd)
        return;

    float2 q = _Pos[gi];

    int cur = _ParentIndex[gi];
    if (cur < _Base || cur >= _Base + _ParentCoarseCount)
        cur = _Base;

    float2 d0 = _Pos[cur] - q;
    float best = dot(d0, d0);

    [loop] for (int iter = 0; iter < 32; iter++)
    {
        int li = cur - _Base;
        if (li < 0 || li >= _ParentCoarseCount)
            break;

        int cnt = _DtNeighborCounts[li];
        if (cnt < 0)
            cnt = 0;
        if (cnt > _DtNeighborCount)
            cnt = _DtNeighborCount;

        int bestNext = cur;
        float bestNextDist = best;

        int baseIdx = li * _DtNeighborCount;
        for (int k = 0; k < cnt; k++)
        {
            int njLocal = _DtNeighbors[baseIdx + k];
            int nj = _Base + njLocal;

            if (nj < _Base || nj >= _Base + _ParentCoarseCount)
                continue;

            float2 d = _Pos[nj] - q;
            float dsq = dot(d, d);
            if (dsq < bestNextDist)
            {
                bestNextDist = dsq;
                bestNext = nj;
            }
        }

        if (bestNext == cur)
            break;

        cur = bestNext;
        best = bestNextDist;
    }

    _ParentIndex[gi] = cur;
}

    [numthreads(256, 1, 1)] void ClearCurrentVolume(uint3 id : SV_DispatchThreadID)
{
    int li = (int)id.x;
    if (li >= _ActiveCount)
        return;
    _CurrentVolumeBits[_Base + li] = 0u;
}

[numthreads(256, 1, 1)] void CacheVolumesHierarchical(uint3 id : SV_DispatchThreadID)
{
    int li = (int)id.x;
    if (li >= _TotalCount)
        return;

    int gi = _Base + li;

    float restV = _RestVolume[gi];
    if (restV <= EPS)
        return;

    XPBI_Mat2 F = XPBI_Mat2FromFloat4(_F[gi]);
    float detF = XPBI_DetMat2(F);
    float leafVol = restV * abs(detF);
    if (leafVol <= EPS)
        return;

    int owner = li;
    [loop] for (int it = 0; it < 64; it++)
    {
        if (owner < 0)
            return;
        if (owner < _ActiveCount)
            break;

        int p = _ParentIndex[_Base + owner];
        if (p < 0)
            return;
        owner = p - _Base;
    }

    if (owner < 0 || owner >= _ActiveCount)
        return;

    XPBI_AtomicAddFloatBits(_CurrentVolumeBits, (uint)(_Base + owner), leafVol);
}

    [numthreads(256, 1, 1)] void CacheKernelH(uint3 id : SV_DispatchThreadID)
{
    int li = (int)id.x;
    if (li >= _ActiveCount)
        return;

    int gi = _Base + li;

    int nCount, n0, n1, n2, n3, n4, n5;
    XPBI_GetNeighbors(gi, nCount, n0, n1, n2, n3, n4, n5);
    if (nCount <= 0)
    {
        _KernelH[gi] = 0.0;
        return;
    }

    float2 xi = _Pos[gi];

    float d[targetNeighborCount];
    int n = 0;

    int ns[targetNeighborCount];
    ns[0] = n0;
    ns[1] = n1;
    ns[2] = n2;
    ns[3] = n3;
    ns[4] = n4;
    ns[5] = n5;

    for (int k = 0; k < nCount; k++)
    {
        int gj = ns[k];
        if (gj < _Base || gj >= _Base + _ActiveCount)
            continue;

        float2 xij = _Pos[gj] - xi;
        float r = length(xij);
        if (r <= EPS)
            continue;

        d[n++] = r;
    }

    if (n == 0)
    {
        _KernelH[gi] = 0.0;
        return;
    }

    for (int a = 1; a < n; a++)
    {
        float key = d[a];
        int b = a - 1;
        while (b >= 0 && d[b] > key)
        {
            d[b + 1] = d[b];
            b--;
        }
        d[b + 1] = key;
    }

    float median;
    if ((n & 1) == 1)
    {
        median = d[n >> 1];
    }
    else
    {
        median = 0.5f * (d[(n >> 1) - 1] + d[n >> 1]);
    }
    _KernelH[gi] = median * KERNEL_H_SCALE;
}

[numthreads(256, 1, 1)] void ComputeCorrectionL(uint3 id : SV_DispatchThreadID)
{
    int li = (int)id.x;
    if (li >= _ActiveCount)
        return;

    int gi = _Base + li;

    float h = _KernelH[gi];
    if (h <= EPS)
    {
        _L[gi] = float4(0, 0, 0, 0);
        return;
    }

    int nCount, n0, n1, n2, n3, n4, n5;
    XPBI_GetNeighbors(gi, nCount, n0, n1, n2, n3, n4, n5);
    if (nCount <= 0)
    {
        _L[gi] = float4(0, 0, 0, 0);
        return;
    }

    float2 xi = _Pos[gi];

    XPBI_Mat2 sum = XPBI_Mat2Zero();
    int ns[targetNeighborCount];
    ns[0] = n0;
    ns[1] = n1;
    ns[2] = n2;
    ns[3] = n3;
    ns[4] = n4;
    ns[5] = n5;

    for (int k = 0; k < nCount; k++)
    {
        int gj = ns[k];
        if (gj < _Base || gj >= _Base + _ActiveCount)
            continue;

        float2 xij = _Pos[gj] - xi;

        float2 gradW = XPBI_GradWendlandC2(xij, h, EPS);
        if (dot(gradW, gradW) <= EPS * EPS)
            continue;

        float Vb = XPBI_ReadCurrentVolume(gj);
        if (Vb <= EPS)
            continue;

        sum.c0 += (Vb * xij.x) * gradW;
        sum.c1 += (Vb * xij.y) * gradW;
    }

    XPBI_Mat2 Lm = XPBI_PseudoInverseMat2(sum, STRETCH_EPS, EIGEN_OFFDIAG_EPS);
    _L[gi] = XPBI_Float4FromMat2(Lm);
}

    [numthreads(256, 1, 1)] void CacheF0AndResetLambda(uint3 id : SV_DispatchThreadID)
{
    int li = (int)id.x;
    if (li >= _ActiveCount)
        return;

    int gi = _Base + li;
    _F0[gi] = _F[gi];
    _Lambda[gi] = 0.0;
}

[numthreads(256, 1, 1)] void SaveVelPrefix(uint3 id : SV_DispatchThreadID)
{
    int li = (int)id.x;
    if (li >= _ActiveCount)
        return;

    int gi = _Base + li;
    _SavedVelPrefix[gi] = _Vel[gi];
}

    [numthreads(256, 1, 1)] void ClearVelDelta(uint3 id : SV_DispatchThreadID)
{
    int li = (int)id.x;
    if (li >= _ActiveCount)
        return;

    int gi = _Base + li;
    _VelDeltaBits[XPBI_VelDeltaIndex(gi, 0u)] = 0u;
    _VelDeltaBits[XPBI_VelDeltaIndex(gi, 1u)] = 0u;
}

#endif
