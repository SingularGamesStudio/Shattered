#ifndef XPBI_SOLVER_CACHE_KERNELS_INCLUDED
    #define XPBI_SOLVER_CACHE_KERNELS_INCLUDED

    // ----------------------------------------------------------------------------
    // ApplyGameplayForces: external one‑shot forces (non‑atomic, assumed unique)
    // ----------------------------------------------------------------------------
    [numthreads(256, 1, 1)] void ApplyGameplayForces(uint3 id : SV_DispatchThreadID)
    {
        uint ei = id.x;
        if (ei >= _ForceEventCount)
        return;

        XPBI_ForceEvent e = _ForceEvents[ei];
        uint gi = e.node;

        if (gi < _Base || gi >= _Base + _TotalCount)
        return;
        if (IsFixedVertex(gi))
        return;

        _Vel[gi] += e.force  * (_InvMass[gi] * _Dt);
    }

    // ----------------------------------------------------------------------------
    // ExternalForces: gravity (applied to all particles)
    // ----------------------------------------------------------------------------
    [numthreads(256, 1, 1)] void ExternalForces(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        if (li >= _TotalCount)
        return;

        uint gi = _Base + li;
        if (IsFixedVertex(gi))
        return;

        _Vel[gi].y += _Gravity * _Dt;
    }

    // ----------------------------------------------------------------------------
    // IntegratePositions: forward Euler
    // ----------------------------------------------------------------------------
    [numthreads(256, 1, 1)] void IntegratePositions(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        if (li >= _TotalCount)
        return;

        uint gi = _Base + li;
        if (IsFixedVertex(gi))
        return;

        _Pos[gi] += _Vel[gi] * _Dt;
    }

    // ----------------------------------------------------------------------------
    // UpdateDtPositions: transform to normalised space for downsampling
    // ----------------------------------------------------------------------------
    [numthreads(256, 1, 1)] void UpdateDtPositions(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        if (li >= _ActiveCount)
        return;

        uint gi = _Base + li;
        _DtPositions[li] = (_Pos[gi] - _DtNormCenter) * _DtNormInvHalfExtent;
    }

    // ----------------------------------------------------------------------------
    // RebuildParentsAtLayer: find nearest coarse particle using local search
    // ----------------------------------------------------------------------------
    [numthreads(256, 1, 1)] void RebuildParentsAtLayer(uint3 id : SV_DispatchThreadID)
    {
        uint gi = _ParentRangeStart + id.x;
        if (gi >= _ParentRangeEnd)
        return;

        float2 q = _Pos[gi];

        int cur = _ParentIndex[gi];
        if (cur < int(_Base) || cur >= int(_Base + _ParentCoarseCount))
        cur = int(_Base);

        float2 d0 = _Pos[cur] - q;
        float best = dot(d0, d0);

        [loop] for (uint iter = 0; iter < 32; iter++)
        {
            uint li = uint(cur - int(_Base));
            if (li >= _ParentCoarseCount)
            break;

            uint cnt = _DtNeighborCounts[li];
            cnt = min(cnt, _DtNeighborCount);

            int bestNext = cur;
            float bestNextDist = best;

            uint baseIdx = li * _DtNeighborCount;
            for (uint k = 0; k < cnt; k++)
            {
                uint njLocal = _DtNeighbors[baseIdx + k];
                uint nj = _Base + njLocal;

                if (nj < _Base || nj >= _Base + _ParentCoarseCount)
                continue;

                float2 d = _Pos[nj] - q;
                float dsq = dot(d, d);
                if (dsq < bestNextDist)
                {
                    bestNextDist = dsq;
                    bestNext = int(nj);
                }
            }

            if (bestNext == cur)
            break;

            cur = bestNext;
            best = bestNextDist;
        }

        _ParentIndex[gi] = cur;
    }

    // ----------------------------------------------------------------------------
    // ClearHierarchicalStats: zero out volume accumulation buffer
    // ----------------------------------------------------------------------------
    [numthreads(256, 1, 1)] void ClearHierarchicalStats(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        if (li >= _ActiveCount)
        return;
        _CurrentVolumeBits[_Base + li] = 0u;
        _CoarseFixed[_Base + li] = 0u;
    }

    // ----------------------------------------------------------------------------
    // CacheHierarchicalStats: accumulate leaf volumes, fixedness to their active ancestor
    // ----------------------------------------------------------------------------
    [numthreads(256, 1, 1)] void CacheHierarchicalStats(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        if (li >= _TotalCount)
        return;

        uint gi = _Base + li;

        float restV = _RestVolume[gi];

        Mat2 F = Mat2FromFloat4(_F[gi]);
        float detF = DetMat2(F);
        float leafVol = restV * abs(detF);

        int owner = int(li);
        [loop] for (uint it = 0; it < 64; it++)
        {
            if (owner < 0)
            return;
            if (owner < int(_ActiveCount))
            break;

            int p = _ParentIndex[_Base + uint(owner)];
            if (p < 0)
            return;
            owner = p - int(_Base);
        }

        if (owner < 0 || owner >= int(_ActiveCount))
        return;

        if (IsFixedVertex(gi))
        {
            InterlockedOr(_CoarseFixed[_Base + uint(owner)], 1u);
        }
        if (leafVol <= EPS)
        return;

        AtomicAddFloatBits(_CurrentVolumeBits, _Base + uint(owner), leafVol);
    }

    // ----------------------------------------------------------------------------
    // CacheKernelH: compute median neighbour distance and scale
    // ----------------------------------------------------------------------------
    [numthreads(256, 1, 1)] void CacheKernelH(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        if (li >= _ActiveCount)
        return;

        uint gi = _Base + li;

        uint nCount, n0, n1, n2, n3, n4, n5;
        GetNeighbors(gi, nCount, n0, n1, n2, n3, n4, n5);
        if (nCount == 0)
        {
            _KernelH[gi] = 0.0;
            return;
        }

        float2 xi = _Pos[gi];

        float d[targetNeighborCount];
        uint n = 0;

        uint ns[targetNeighborCount] = {n0, n1, n2, n3, n4, n5};

        [unroll] for (uint k = 0; k < nCount; k++)
        {
            uint gj = ns[k];
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

        [unroll] for (uint a = 1; a < n; a++)
        {
            float key = d[a];
            uint b = a;
            while (b > 0 && d[b - 1] > key)
            {
                d[b] = d[b - 1];
                b--;
            }
            d[b] = key;
        }

        float median;
        if ((n & 1) == 1)
        median = d[n >> 1];
        else
        median = 0.5f * (d[(n >> 1) - 1] + d[n >> 1]);

        _KernelH[gi] = median * KERNEL_H_SCALE;
    }

    // ----------------------------------------------------------------------------
    // ComputeCorrectionL: compute SPH correction matrix
    // ----------------------------------------------------------------------------
    [numthreads(256, 1, 1)] void ComputeCorrectionL(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        if (li >= _ActiveCount)
        return;

        uint gi = _Base + li;

        float h = _KernelH[gi];
        if (h <= EPS)
        {
            _L[gi] = float4(0, 0, 0, 0);
            return;
        }

        uint nCount, n0, n1, n2, n3, n4, n5;
        GetNeighbors(gi, nCount, n0, n1, n2, n3, n4, n5);
        if (nCount == 0)
        {
            _L[gi] = float4(0, 0, 0, 0);
            return;
        }

        float2 xi = _Pos[gi];

        Mat2 sum = Mat2Zero();
        uint ns[targetNeighborCount] = {n0, n1, n2, n3, n4, n5};

        [unroll] for (uint k = 0; k < nCount; k++)
        {
            uint gj = ns[k];
            if (gj < _Base || gj >= _Base + _ActiveCount)
            continue;

            float2 xij = _Pos[gj] - xi;

            float2 gradW = GradWendlandC2(xij, h, EPS);
            if (dot(gradW, gradW) <= EPS * EPS)
            continue;

            float Vb = ReadCurrentVolume(gj);
            if (Vb <= EPS)
            continue;

            sum.c0 += (Vb * xij.x) * gradW;
            sum.c1 += (Vb * xij.y) * gradW;
        }

        Mat2 Lm = PseudoInverseMat2(sum, STRETCH_EPS, EIGEN_OFFDIAG_EPS);
        _L[gi] = Float4FromMat2(Lm);
    }

    // ----------------------------------------------------------------------------
    // CacheF0AndResetLambda: store F as initial for next step, reset λ
    // ----------------------------------------------------------------------------
    [numthreads(256, 1, 1)] void CacheF0AndResetLambda(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        if (li >= _ActiveCount)
        return;

        uint gi = _Base + li;
        _F0[gi] = _F[gi];
        _Lambda[gi] = 0.0;
    }

    // ----------------------------------------------------------------------------
    // SaveVelPrefix: store velocity before multigrid V‑cycle
    // ----------------------------------------------------------------------------
    [numthreads(256, 1, 1)] void SaveVelPrefix(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        if (li >= _ActiveCount)
        return;

        uint gi = _Base + li;
        _SavedVelPrefix[gi] = _Vel[gi];
    }

    // ----------------------------------------------------------------------------
    // ClearVelDelta: zero out velocity delta accumulation buffer
    // ----------------------------------------------------------------------------
    [numthreads(256, 1, 1)] void ClearVelDelta(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        if (li >= _ActiveCount)
        return;

        uint gi = _Base + li;
        _VelDeltaBits[gi * 2u + 0u] = 0u;
        _VelDeltaBits[gi * 2u + 1u] = 0u;
    }

    // ----------------------------------------------------------------------------
    // Clear restricted dV
    // ----------------------------------------------------------------------------
    [numthreads(256, 1, 1)] void ClearRestrictedDeltaV(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        if (li >= _ActiveCount)
        return;

        uint gi = _Base + li;
        _RestrictedDeltaVBits[gi * 2u + 0u] = 0u;
        _RestrictedDeltaVBits[gi * 2u + 1u] = 0u;
        _RestrictedDeltaVCount[gi] = 0u;
        _RestrictedDeltaVAvg[gi] = 0.0f;
    }
    // ----------------------------------------------------------------------------
    // Restrict from force events directly: event -> leaf deltaV -> active owner accumulation.
    // ----------------------------------------------------------------------------
    [numthreads(256, 1, 1)] void RestrictGameplayDeltaVFromEvents(uint3 id : SV_DispatchThreadID)
    {
        uint ei = id.x;
        if (ei >= _ForceEventCount)
        return;

        XPBI_ForceEvent e = _ForceEvents[ei];
        uint gi = e.node;

        if (gi < _Base || gi >= _Base + _FineCount)
        return;
        if (IsFixedVertex(gi))
        return;

        float2 leafDeltaV = e.force * (_InvMass[gi] * _Dt);

        int owner = int(gi - _Base);
        [loop] for (uint it = 0; it < 64; it++)
        {
            if (owner < 0)
            return;
            if (owner < int(_ActiveCount))
            break;

            int p = _ParentIndex[_Base + uint(owner)];
            if (p < 0)
            return;
            owner = p - int(_Base);
        }

        if (owner < 0 || owner >= int(_ActiveCount))
        return;

        uint ownerGi = _Base + uint(owner);

        AtomicAddFloatBits(_RestrictedDeltaVBits, ownerGi * 2u + 0u, leafDeltaV.x);
        AtomicAddFloatBits(_RestrictedDeltaVBits, ownerGi * 2u + 1u, leafDeltaV.y);
        InterlockedAdd(_RestrictedDeltaVCount[ownerGi], 1u);
    }

    // ----------------------------------------------------------------------------
    // Add the normalized hint to both Vel and SavedVelPrefix.
    // ----------------------------------------------------------------------------
    [numthreads(256, 1, 1)] void ApplyRestrictedDeltaVToActiveAndPrefix(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        if (li >= _ActiveCount)
        return;

        uint gi = _Base + li;

        if (_CoarseFixed[gi] != 0u)
        {
            _RestrictedDeltaVAvg[gi] = 0.0f;
            return;
        }

        uint cnt = _RestrictedDeltaVCount[gi];
        if (cnt == 0u)
        {
            _RestrictedDeltaVAvg[gi] = 0.0f;
            return;
        }

        float2 sum;
        sum.x = asfloat(_RestrictedDeltaVBits[gi * 2u + 0u]);
        sum.y = asfloat(_RestrictedDeltaVBits[gi * 2u + 1u]);

        float invCnt = 1.0 / max((float)cnt, 1.0);
        float2 dv = sum * (invCnt * _RestrictedDeltaVScale);

        _RestrictedDeltaVAvg[gi] = dv;
        _Vel[gi] += dv;
        _SavedVelPrefix[gi] += dv;
    }

    // ----------------------------------------------------------------------------
    // remove the hint again after prolongation so it doesn't persist into the physical state.
    // ----------------------------------------------------------------------------
    [numthreads(256, 1, 1)] void RemoveRestrictedDeltaVFromActive(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        if (li >= _ActiveCount)
        return;

        uint gi = _Base + li;
        _Vel[gi] -= _RestrictedDeltaVAvg[gi];
    }

#endif // XPBI_SOLVER_CACHE_KERNELS_INCLUDED
