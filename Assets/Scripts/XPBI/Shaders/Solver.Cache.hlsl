#ifndef XPBI_SOLVER_CACHE_KERNELS_INCLUDED
    #define XPBI_SOLVER_CACHE_KERNELS_INCLUDED

    // ----------------------------------------------------------------------------
    // ApplyGameplayForces: external per-node acceleration/impulse-like input
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

        _Vel[gi] += e.force * _Dt;
    }

    // ----------------------------------------------------------------------------
    // ExternalForces: gravity acceleration
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
    // ClampVelocities: hard safety bound against runaway impulses
    // ----------------------------------------------------------------------------
    [numthreads(256, 1, 1)] void ClampVelocities(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        if (li >= _TotalCount)
        return;

        uint gi = _Base + li;
        if (IsFixedVertex(gi))
        return;

        float2 v = _Vel[gi];
        if (!all(isfinite(v)))
        {
            _Vel[gi] = 0.0;
            return;
        }

        float maxSpeed = max(_MaxSpeed, 0.0);
        if (maxSpeed <= 0.0)
        return;

        float speed = length(v);
        if (speed > maxSpeed)
        _Vel[gi] = v * (maxSpeed / max(speed, EPS));
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

        float2 p = _Pos[gi];
        float2 v = _Vel[gi];

        if (!all(isfinite(p)))
        p = 0.0;
        if (!all(isfinite(v)))
        {
            _Vel[gi] = 0.0;
            _Pos[gi] = p;
            return;
        }

        float2 dx = v * _Dt;
        float maxStep = max(_MaxStep, 0.0);
        if (maxStep > 0.0)
        {
            float stepLen = length(dx);
            if (stepLen > maxStep)
            dx *= maxStep / max(stepLen, EPS);
        }

        _Pos[gi] = p + dx;
    }

    [numthreads(256, 1, 1)] void UpdateDtPositions(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        if (li >= _ActiveCount)
        return;

        uint gi = _Base + li;
        _DtPositions[li] = (_Pos[gi] - _DtNormCenter) * _DtNormInvHalfExtent;
    }

    [numthreads(256, 1, 1)] void UpdateDtPositionsMapped(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        if (li >= _ActiveCount)
        return;

        int giSigned = _DtGlobalNodeMap[li];
        if (giSigned < 0)
        return;

        uint gi = (uint)giSigned;
        _DtPositions[li] = (_Pos[gi] - _DtNormCenter) * _DtNormInvHalfExtent;
    }

    // ----------------------------------------------------------------------------
    // RebuildParentsAtLayer: find nearest coarse particle using local search
    // ----------------------------------------------------------------------------
    [numthreads(256, 1, 1)] void RebuildParentsAtLayer(uint3 id : SV_DispatchThreadID)
    {
        uint liFine = _ParentRangeStart + id.x;
        if (liFine >= _ParentRangeEnd)
        return;

        uint gi = (_UseDtGlobalNodeMap != 0u) ? GlobalIndexFromLocal(liFine) : (_Base + liFine);
        if (gi == ~0u)
        return;

        float2 q = _Pos[gi];

        int cur = _ParentIndex[gi];
        uint curLi = (cur >= 0) ? LocalIndexFromGlobal((uint)cur) : ~0u;
        if (curLi == ~0u || curLi >= _ParentCoarseCount)
        {
            uint firstGi = GlobalIndexFromLocal(0u);
            if (firstGi == ~0u)
            return;
            cur = int(firstGi);
        }

        float2 d0 = _Pos[cur] - q;
        float best = dot(d0, d0);

        [loop] for (uint iter = 0; iter < 32; iter++)
        {
            uint liBase = LocalIndexFromGlobal((uint)cur);
            if (liBase == ~0u)
            break;

            uint li = (_UseDtGlobalNodeMap != 0u) ? liBase : (_DtLocalBase + liBase);
            if (liBase >= _ParentCoarseCount)
            break;

            uint cnt = _DtNeighborCounts[li];
            cnt = min(cnt, _DtNeighborCount);

            int bestNext = cur;
            float bestNextDist = best;

            uint baseIdx = li * _DtNeighborCount;
            for (uint k = 0; k < cnt; k++)
            {
                uint njLocal = _DtNeighbors[baseIdx + k];
                uint nj = GlobalIndexFromLocal(njLocal);
                if (nj == ~0u)
                continue;

                uint njLi = LocalIndexFromGlobal(nj);
                if (njLi == ~0u || njLi >= _ParentCoarseCount)
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
        uint gi = GlobalIndexFromLocal(li);
        if (gi == ~0u)
        return;
        _CurrentVolumeBits[gi] = 0u;
        _CurrentTotalMassBits[gi] = 0u;
        _FixedChildPosBits[gi * 2u + 0u] = 0u;
        _FixedChildPosBits[gi * 2u + 1u] = 0u;
        _FixedChildCount[gi] = 0u;
        _CoarseFixed[gi] = 0u;
    }

    // ----------------------------------------------------------------------------
    // CacheHierarchicalStats: accumulate leaf volumes, fixedness to their active ancestor
    // ----------------------------------------------------------------------------
    [numthreads(256, 1, 1)] void CacheHierarchicalStats(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        if (li >= _FineCount)
        return;

        uint gi = GlobalIndexFromLocal(li);
        if (gi == ~0u)
        return;

        float invM = _InvMass[gi];
        float leafMass = (invM > EPS) ? (1.0 / invM) : 0.0;

        float restV = _RestVolume[gi];

        Mat2 F = Mat2FromFloat4(_F[gi]);
        float detF = DetMat2(F);
        if (!isfinite(detF))
        return;
        float detFAbs = min(abs(detF), 4.0);
        float leafVol = restV * detFAbs;
        if (!isfinite(leafVol))
        return;

        int owner = int(li);
        [loop] for (uint it = 0; it < 64; it++)
        {
            if (owner < 0)
            return;
            if (owner < int(_ActiveCount))
            break;

            uint ownerGi = GlobalIndexFromLocal((uint)owner);
            if (ownerGi == ~0u)
            return;

            int p = _ParentIndex[ownerGi];
            if (p < 0)
            return;

            uint ownerLi = LocalIndexFromGlobal((uint)p);
            if (ownerLi == ~0u)
            return;
            owner = (int)ownerLi;
        }

        if (owner < 0 || owner >= int(_ActiveCount))
        return;

        uint ownerGi = GlobalIndexFromLocal((uint)owner);
        if (ownerGi == ~0u)
        return;

        if (IsFixedVertex(gi))
        {
            AtomicAddFloatBits(_FixedChildPosBits, ownerGi * 2u + 0u, _Pos[gi].x);
            AtomicAddFloatBits(_FixedChildPosBits, ownerGi * 2u + 1u, _Pos[gi].y);
            InterlockedAdd(_FixedChildCount[ownerGi], 1u);
        }
        if (leafMass > EPS)
        AtomicAddFloatBits(_CurrentTotalMassBits, ownerGi, leafMass);

        if (leafVol <= EPS)
        return;

        AtomicAddFloatBits(_CurrentVolumeBits, ownerGi, leafVol);
    }

    // ----------------------------------------------------------------------------
    // FinalizeHierarchicalStats: derive coarse fixed mask from fixed-child count.
    // single fixed child => hinge-like (not fully fixed), multiple => fixed.
    // ----------------------------------------------------------------------------
    [numthreads(256, 1, 1)] void FinalizeHierarchicalStats(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        if (li >= _ActiveCount)
        return;

        uint gi = GlobalIndexFromLocal(li);
        if (gi == ~0u)
        return;
        _CoarseFixed[gi] = (_FixedChildCount[gi] > 1u) ? 1u : 0u;
    }

    // ----------------------------------------------------------------------------
    // ComputeCorrectionL: compute SPH correction matrix
    // ----------------------------------------------------------------------------
    [numthreads(256, 1, 1)] void ComputeCorrectionL(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        if (li >= _ActiveCount)
        return;

        uint gi = GlobalIndexFromLocal(li);
        if (gi == ~0u)
        return;

        float h = max(_LayerKernelH, 1e-4);
        if (h <= EPS)
        {
            _L[gi] = float4(0, 0, 0, 0);
            return;
        }

        float kernelH = WendlandKernelHFromSupport(h);
        if (kernelH <= EPS)
        {
            _L[gi] = float4(0, 0, 0, 0);
            return;
        }

        uint nCount;
        uint ns[targetNeighborCount];
        GetNeighbors(gi, nCount, ns);
        if (nCount == 0)
        {
            _L[gi] = float4(0, 0, 0, 0);
            return;
        }

        float2 xi = _Pos[gi];

        Mat2 sum = Mat2Zero();
        uint validLNeighbors = 0u;

        [unroll] for (uint k = 0; k < nCount; k++)
        {
            uint gj = ns[k];
            uint gjLi = LocalIndexFromGlobal(gj);
            if (gjLi == ~0u || gjLi >= _ActiveCount)
            continue;

            float2 xij = _Pos[gj] - xi;

            float2 gradW = GradWendlandC2(xij, kernelH, EPS);
            if (dot(gradW, gradW) <= EPS * EPS)
            continue;

            float Vb = ReadCurrentVolume(gj);
            if (!(Vb > EPS))
            continue;

            sum.c0 += (Vb * xij.x) * gradW;
            sum.c1 += (Vb * xij.y) * gradW;
            validLNeighbors++;
        }

        if (validLNeighbors < 3u)
        {
            _L[gi] = float4(0, 0, 0, 0);
            return;
        }

        float sumFro2 = dot(sum.c0, sum.c0) + dot(sum.c1, sum.c1);
        if (!(sumFro2 > 1e-12))
        {
            _L[gi] = float4(0, 0, 0, 0);
            return;
        }

        float sumDetAbs = abs(DetMat2(sum));
        if (!(sumDetAbs > 1e-10 * sumFro2))
        {
            _L[gi] = float4(0, 0, 0, 0);
            return;
        }

        Mat2 Lm = PseudoInverseMat2(sum, STRETCH_EPS, EIGEN_OFFDIAG_EPS);

        float4 L4 = Float4FromMat2(Lm);

        // Hard safety: if L is non-finite or absurd, fall back to no correction.
        // This prevents a single ill-conditioned neighborhood (often after DT flips) from blowing up F and velocities.
        if (any(isnan(L4)) || any(abs(L4) > 1e3))
        {
            _L[gi] = float4(0, 0, 0, 0);
            return;
        }

        _L[gi] = L4;
    }


    // ----------------------------------------------------------------------------
    // CacheF0AndResetLambda: store F as initial for next step, reset λ
    // ----------------------------------------------------------------------------
    [numthreads(256, 1, 1)] void CacheF0AndResetLambda(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        if (li >= _ActiveCount)
        return;

        uint gi = GlobalIndexFromLocal(li);
        if (gi == ~0u)
        return;
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

        uint gi = GlobalIndexFromLocal(li);
        if (gi == ~0u)
        return;
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

        uint gi = GlobalIndexFromLocal(li);
        if (gi == ~0u)
        return;
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

        uint gi = GlobalIndexFromLocal(li);
        if (gi == ~0u)
        return;
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

        float2 leafDeltaV = e.force * _Dt;

        int owner = -1;
        uint ownerGi = gi;
        [loop] for (uint it = 0; it < 64; it++)
        {
            uint ownerLi = LocalIndexFromGlobal(ownerGi);
            if (ownerLi != ~0u && ownerLi < _ActiveCount)
            {
                owner = (int)ownerLi;
                break;
            }

            int p = _ParentIndex[ownerGi];
            if (p < 0)
            return;
            ownerGi = (uint)p;
        }

        if (owner < 0 || owner >= int(_ActiveCount))
        return;

        ownerGi = GlobalIndexFromLocal((uint)owner);
        if (ownerGi == ~0u)
        return;

        AtomicAddFloatBits(_RestrictedDeltaVBits, ownerGi * 2u + 0u, leafDeltaV.x);
        AtomicAddFloatBits(_RestrictedDeltaVBits, ownerGi * 2u + 1u, leafDeltaV.y);
        InterlockedAdd(_RestrictedDeltaVCount[ownerGi], 1u);
    }

    // ----------------------------------------------------------------------------
    // Restrict fine residual velocity to active owner: childVel - parentVel
    // ----------------------------------------------------------------------------
    [numthreads(256, 1, 1)] void RestrictFineVelocityResidualToActive(uint3 id : SV_DispatchThreadID)
    {
        uint li = _ActiveCount + id.x;
        if (li >= _FineCount)
        return;

        uint gi = GlobalIndexFromLocal(li);
        if (gi == ~0u)
        return;
        if (IsFixedVertex(gi))
        return;

        int p = _ParentIndex[gi];
        if (p < int(_Base) || p >= int(_Base + _ActiveCount))
        return;

        uint ownerGi = uint(p);

        float2 residual = _Vel[gi] - _Vel[ownerGi];

        AtomicAddFloatBits(_RestrictedDeltaVBits, ownerGi * 2u + 0u, residual.x);
        AtomicAddFloatBits(_RestrictedDeltaVBits, ownerGi * 2u + 1u, residual.y);
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

        uint gi = GlobalIndexFromLocal(li);
        if (gi == ~0u)
        return;

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

        uint gi = GlobalIndexFromLocal(li);
        if (gi == ~0u)
        return;
        _Vel[gi] -= _RestrictedDeltaVAvg[gi];
    }

#endif // XPBI_SOLVER_CACHE_KERNELS_INCLUDED