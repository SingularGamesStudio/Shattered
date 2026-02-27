#ifndef XPBI_SOLVER_CACHE_KERNELS_INCLUDED
    #define XPBI_SOLVER_CACHE_KERNELS_INCLUDED

    // ----------------------------------------------------------------------------
    // RebuildParentsAtLayer: find nearest coarse particle using local search
    // ----------------------------------------------------------------------------
    [numthreads(256, 1, 1)] void RebuildParentsAtLayer(uint3 id : SV_DispatchThreadID)
    {
        uint liFine = _ParentRangeStart + id.x;
        if (liFine >= _ParentRangeEnd)
        return;

        uint gi = ~0u;
        gi = (_UseDtGlobalNodeMap != 0u) ? GlobalIndexFromLocal(liFine) : (_Base + liFine);
        if (gi == ~0u)
        return;

        float2 q = _Pos[gi];

        int cur = _ParentIndex[gi];
        uint curLi = ~0u;
        curLi = (cur >= 0) ? LocalIndexFromGlobal((uint)cur) : ~0u;
        if (curLi == ~0u || curLi >= _ParentCoarseCount)
        {
            uint firstGi = ~0u;
            firstGi = GlobalIndexFromLocal(0u);
            if (firstGi == ~0u)
            return;
            cur = int(firstGi);
        }

        float2 d0 = _Pos[cur] - q;
        float best = dot(d0, d0);

        [loop] for (uint iter = 0; iter < 32; iter++)
        {
            uint liBase = ~0u;
            liBase = LocalIndexFromGlobal((uint)cur);
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
                uint nj = ~0u;
                nj = GlobalIndexFromLocal(njLocal);
                if (nj == ~0u)
                continue;

                uint njLi = ~0u;
                njLi = LocalIndexFromGlobal(nj);
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
        uint gi = ~0u;
        gi = GlobalIndexFromLocal(li);
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

        uint gi = ~0u;
        gi = GlobalIndexFromLocal(li);
        if (gi == ~0u)
        return;

        float invM = _InvMass[gi];
        float leafMass = (invM > EPS) ? (1.0 / invM) : 0.0;

        float restV = _RestVolume[gi];

        Mat2 F = (Mat2)0;
        F = Mat2FromFloat4(_F[gi]);
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

            uint ownerGi = ~0u;
            ownerGi = GlobalIndexFromLocal((uint)owner);
            if (ownerGi == ~0u)
            return;

            int p = _ParentIndex[ownerGi];
            if (p < 0)
            return;

            uint ownerLi = ~0u;
            ownerLi = LocalIndexFromGlobal((uint)p);
            if (ownerLi == ~0u)
            return;
            owner = (int)ownerLi;
        }

        if (owner < 0 || owner >= int(_ActiveCount))
        return;

        uint ownerGi = ~0u;
        ownerGi = GlobalIndexFromLocal((uint)owner);
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

        uint gi = ~0u;
        gi = GlobalIndexFromLocal(li);
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

        uint gi = ~0u;
        gi = GlobalIndexFromLocal(li);
        if (gi == ~0u)
        return;

        float h = max(_LayerKernelH, 1e-4);
        if (h <= EPS)
        {
            _L[gi] = float4(0, 0, 0, 0);
            return;
        }

        float kernelH = 0.0;
        kernelH = WendlandKernelHFromSupport(h);
        if (kernelH <= EPS)
        {
            _L[gi] = float4(0, 0, 0, 0);
            return;
        }

        uint neighborCount = 0u;
        uint ns[targetNeighborCount];
        [unroll] for (uint initIdx0 = 0u; initIdx0 < targetNeighborCount; initIdx0++) ns[initIdx0] = ~0u;
        GetNeighbors(gi, neighborCount, ns);

        float2 xi = _Pos[gi];

        Mat2 sum = (Mat2)0;
        sum = Mat2Zero();
        uint validLNeighbors = 0u;

        [unroll] for (uint k = 0u; k < targetNeighborCount; k++)
        {
            uint gj = ns[k];
            if (gj == ~0u) continue;
            uint gjLi = ~0u;
            gjLi = LocalIndexFromGlobal(gj);
            if (gjLi == ~0u || gjLi >= _ActiveCount)
            continue;

            float2 xij = _Pos[gj] - xi;

            float2 gradW = 0.0;
            gradW = GradWendlandC2(xij, kernelH, EPS);
            if (dot(gradW, gradW) <= EPS * EPS)
            continue;

            float Vb = 0.0;
            Vb = ReadCurrentVolume(gj);
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

        Mat2 Lm = (Mat2)0;
        Lm = PseudoInverseMat2(sum, STRETCH_EPS, EIGEN_OFFDIAG_EPS);

        float4 L4 = 0.0;
        L4 = Float4FromMat2(Lm);

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

        uint gi = ~0u;
        gi = GlobalIndexFromLocal(li);
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

        uint gi = ~0u;
        gi = GlobalIndexFromLocal(li);
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

        uint gi = ~0u;
        gi = GlobalIndexFromLocal(li);
        if (gi == ~0u)
        return;
        _VelDeltaBits[gi * 2u + 0u] = 0u;
        _VelDeltaBits[gi * 2u + 1u] = 0u;
    }

    [numthreads(256, 1, 1)] void ResetCollisionLambda(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        if (li >= _ActiveCount)
        return;

        uint dtLi = (_UseDtGlobalNodeMap != 0u) ? li : (_DtLocalBase + li);
        uint baseIdx = dtLi * _DtNeighborCount;

        [unroll] for (uint k = 0u; k < targetNeighborCount; k++)
        {
            if (k >= _DtNeighborCount) break;
            _DurabilityLambda[baseIdx + k] = 0.0;
            _CollisionLambda[baseIdx + k] = 0.0;
        }
    }

    [numthreads(1, 1, 1)] void ClearCollisionEventCount(uint3 id : SV_DispatchThreadID)
    {
        _CollisionEventCount[0] = 0u;
    }

    [numthreads(256, 1, 1)] void BuildCollisionEventsL0(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        if (li >= _ActiveCount)
        return;

        uint gi = ~0u;
        gi = GlobalIndexFromLocal(li);
        if (gi == ~0u)
        return;

        int ownerI = _DtOwnerByLocal[li];
        if (ownerI < 0)
        return;

        float support = 0.0;
        support = WendlandSupportRadius(max(_LayerKernelH, 1e-4));
        if (support <= EPS)
        return;

        float targetSep = max(EPS, _CollisionSupportScale * support);
        float targetSepSq = targetSep * targetSep;

        uint dtLi = (_UseDtGlobalNodeMap != 0u) ? li : (_DtLocalBase + li);
        uint baseIdx = dtLi * _DtNeighborCount;

        uint rawCount = _DtNeighborCounts[dtLi];
        uint nCount = min(rawCount, _DtNeighborCount);
        nCount = min(nCount, targetNeighborCount);

        float2 xi = _Pos[gi];

        [loop] for (uint k = 0u; k < nCount; k++)
        {
            uint gjLi = _DtNeighbors[baseIdx + k];
            if (gjLi == ~0u || gjLi >= _ActiveCount)
            continue;

            int ownerJ = _DtOwnerByLocal[gjLi];
            if (ownerJ < 0 || ownerJ == ownerI)
            continue;

            uint gj = ~0u;
            gj = GlobalIndexFromLocal(gjLi);
            if (gj == ~0u || gj <= gi)
            continue;

            float2 xj = _Pos[gj];
            float2 dx = xj - xi;
            float distSq = dot(dx, dx);
            if (distSq <= EPS * EPS || distSq > targetSepSq)
            continue;

            float dist = sqrt(distSq);
            float Cn = dist - targetSep;
            if (Cn >= 0.0)
            continue;

            float2 nrm = dx / max(dist, EPS);
            float pen = -Cn;

            uint idx = 0u;
            InterlockedAdd(_CollisionEventCount[0], 1u, idx);
            if (idx >= _CollisionEventCapacity)
            return;

            XPBI_CollisionEvent e;
            e.aGi = gi;
            e.bGi = gj;
            e.nPen = float4(nrm.x, nrm.y, pen, 0.0);
            _CollisionEvents[idx] = e;
        }
    }

    [numthreads(256, 1, 1)] void ClearTransferredCollision(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        if (li >= _ActiveCount)
        return;

        uint dtLi = (_UseDtGlobalNodeMap != 0u) ? li : (_DtLocalBase + li);
        uint baseIdx = dtLi * _DtNeighborCount;

        [unroll] for (uint k = 0u; k < targetNeighborCount; k++)
        {
            if (k >= _DtNeighborCount) break;
            uint slot = baseIdx + k;
            _XferColCount[slot] = 0u;
            _XferColNXBits[slot] = 0u;
            _XferColNYBits[slot] = 0u;
            _XferColPenBits[slot] = 0u;
        }
    }

    static uint FindActiveOwnerLocalFromGlobal(uint leafGi)
    {
        uint curGi = leafGi;
        [loop] for (uint it = 0u; it < 64u; it++)
        {
            uint curLi = ~0u;
            curLi = LocalIndexFromGlobal(curGi);
            if (curLi != ~0u && curLi < _ActiveCount)
            return curLi;

            int p = _ParentIndex[curGi];
            if (p < 0)
            return ~0u;
            curGi = (uint)p;
        }
        return ~0u;
    }

    [numthreads(256, 1, 1)] void RestrictCollisionEventsToActivePairs(uint3 id : SV_DispatchThreadID)
    {
        uint ei = id.x;
        uint count = _CollisionEventCount[0];
        if (ei >= count)
        return;

        XPBI_CollisionEvent e = _CollisionEvents[ei];

        uint ownerLiA = FindActiveOwnerLocalFromGlobal(e.aGi);
        uint ownerLiB = FindActiveOwnerLocalFromGlobal(e.bGi);
        if (ownerLiA == ~0u || ownerLiB == ~0u)
        return;
        if (ownerLiA == ownerLiB)
        return;

        uint ownerGiA = ~0u;
        ownerGiA = GlobalIndexFromLocal(ownerLiA);
        uint ownerGiB = ~0u;
        ownerGiB = GlobalIndexFromLocal(ownerLiB);
        if (ownerGiA == ~0u || ownerGiB == ~0u)
        return;

        float2 nrm = e.nPen.xy;
        float pen = e.nPen.z;
        if (!(pen > EPS))
        return;

        uint ownerLiI = ownerLiA;
        uint ownerLiJ = ownerLiB;
        uint ownerGiI = ownerGiA;
        uint ownerGiJ = ownerGiB;
        if (ownerGiI > ownerGiJ)
        {
            ownerLiI = ownerLiB; ownerLiJ = ownerLiA;
            ownerGiI = ownerGiB; ownerGiJ = ownerGiA;
            nrm = -nrm;
        }

        uint dtLiI = (_UseDtGlobalNodeMap != 0u) ? ownerLiI : (_DtLocalBase + ownerLiI);
        uint baseIdx = dtLiI * _DtNeighborCount;

        uint rawN = _DtNeighborCounts[dtLiI];
        uint nCount = min(rawN, _DtNeighborCount);
        nCount = min(nCount, targetNeighborCount);

        uint kFound = ~0u;
        [loop] for (uint k = 0u; k < nCount; k++)
        {
            if (_DtNeighbors[baseIdx + k] == ownerLiJ)
            {
                kFound = k;
                break;
            }
        }
        if (kFound == ~0u)
        return;

        uint slot = baseIdx + kFound;

        AtomicAddFloatBits(_XferColNXBits, slot, nrm.x * pen);
        AtomicAddFloatBits(_XferColNYBits, slot, nrm.y * pen);
        AtomicAddFloatBits(_XferColPenBits, slot, pen);
        InterlockedAdd(_XferColCount[slot], 1u);
    }

    // ----------------------------------------------------------------------------
    // Clear restricted dV
    // ----------------------------------------------------------------------------
    [numthreads(256, 1, 1)] void ClearRestrictedDeltaV(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        if (li >= _ActiveCount)
        return;

        uint gi = ~0u;
        gi = GlobalIndexFromLocal(li);
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
            uint ownerLi = ~0u;
            ownerLi = LocalIndexFromGlobal(ownerGi);
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

        ownerGi = ~0u;
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

        uint gi = ~0u;
        gi = GlobalIndexFromLocal(li);
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

        uint gi = ~0u;
        gi = GlobalIndexFromLocal(li);
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

        float2 sum = 0.0;
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

        uint gi = ~0u;
        gi = GlobalIndexFromLocal(li);
        if (gi == ~0u)
        return;
        _Vel[gi] -= _RestrictedDeltaVAvg[gi];
    }

#endif // XPBI_SOLVER_CACHE_KERNELS_INCLUDED