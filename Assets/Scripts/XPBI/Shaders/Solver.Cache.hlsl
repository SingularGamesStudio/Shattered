#ifndef XPBI_SOLVER_CACHE_KERNELS_INCLUDED
    #define XPBI_SOLVER_CACHE_KERNELS_INCLUDED

    // ----------------------------------------------------------------------------
    // RebuildParentsAtLayer: find nearest coarse particle, then assign k nearest weighted parents.
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
        uint parentCount = ParentReadCount();
        uint parentBase = ParentSlotBase(gi);

        [unroll] for (uint clearIdx = 0u; clearIdx < targetParentCount; clearIdx++)
        {
            _ParentIndices[parentBase + clearIdx] = -1;
            _ParentWeights[parentBase + clearIdx] = 0.0;
        }

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

        int candidateGi[targetNeighborCount + 1];
        float candidateDistSq[targetNeighborCount + 1];
        uint candidateCount = 0u;

        candidateGi[candidateCount] = cur;
        candidateDistSq[candidateCount] = best;
        candidateCount++;

        uint curLocal = ~0u;
        curLocal = LocalIndexFromGlobal((uint)cur);
        if (curLocal != ~0u && curLocal < _ParentCoarseCount)
        {
            uint dtLi = (_UseDtGlobalNodeMap != 0u) ? curLocal : (_DtLocalBase + curLocal);
            uint rawCount = _DtNeighborCounts[dtLi];
            uint nCount = min(rawCount, _DtNeighborCount);
            nCount = min(nCount, targetNeighborCount);
            uint nBase = dtLi * _DtNeighborCount;

            [loop] for (uint k = 0u; k < nCount; k++)
            {
                uint njLocal = _DtNeighbors[nBase + k];
                uint nj = ~0u;
                nj = GlobalIndexFromLocal(njLocal);
                if (nj == ~0u) continue;

                uint njLi = ~0u;
                njLi = LocalIndexFromGlobal(nj);
                if (njLi == ~0u || njLi >= _ParentCoarseCount) continue;

                bool duplicate = false;
                [unroll] for (uint c = 0u; c < targetNeighborCount + 1; c++)
                {
                    if (c >= candidateCount) break;
                    if (candidateGi[c] == int(nj)) { duplicate = true; break; }
                }
                if (duplicate || candidateCount >= (targetNeighborCount + 1)) continue;

                float2 d = _Pos[nj] - q;
                candidateGi[candidateCount] = int(nj);
                candidateDistSq[candidateCount] = dot(d, d);
                candidateCount++;
            }
        }

        int selectedParent[targetParentCount];
        float selectedDistSq[targetParentCount];
        [unroll] for (uint i = 0u; i < targetParentCount; i++)
        {
            selectedParent[i] = -1;
            selectedDistSq[i] = 1e30;
        }

        [loop] for (uint c = 0u; c < targetNeighborCount + 1; c++)
        {
            if (c >= candidateCount) break;
            int cand = candidateGi[c];
            float candDist = candidateDistSq[c];

            uint insertAt = parentCount;
            [unroll] for (uint s = 0u; s < targetParentCount; s++)
            {
                if (s >= parentCount) break;
                if (candDist < selectedDistSq[s]) { insertAt = s; break; }
            }
            if (insertAt >= parentCount) continue;

            [unroll] for (uint shift = 0u; shift < targetParentCount; shift++)
            {
                uint idx = parentCount - 1u - shift;
                if (idx <= insertAt || idx >= parentCount) continue;
                selectedParent[idx] = selectedParent[idx - 1u];
                selectedDistSq[idx] = selectedDistSq[idx - 1u];
            }

            selectedParent[insertAt] = cand;
            selectedDistSq[insertAt] = candDist;
        }

        uint usedCount = 0u;
        float maxDist = 0.0;
        [unroll] for (uint s = 0u; s < targetParentCount; s++)
        {
            if (s >= parentCount) break;
            int p = selectedParent[s];
            if (p < 0) continue;
            usedCount++;
            maxDist = max(maxDist, sqrt(max(selectedDistSq[s], 0.0)));
        }

        if (usedCount == 0u)
        {
            selectedParent[0] = cur;
            selectedDistSq[0] = best;
            usedCount = 1u;
            maxDist = sqrt(max(best, 0.0));
        }

        float invMaxDist = (maxDist > EPS) ? (1.0 / maxDist) : 0.0;
        float weightSum = 0.0;
        float rawWeight[targetParentCount];
        [unroll] for (uint s = 0u; s < targetParentCount; s++)
        {
            rawWeight[s] = 0.0;
            if (s >= parentCount) continue;
            int p = selectedParent[s];
            if (p < 0) continue;

            float d = sqrt(max(selectedDistSq[s], 0.0));
            float normalizedD = (maxDist > EPS) ? (d * invMaxDist) : 0.0;
            float w = 1.0 / (normalizedD + max(_ParentWeightEpsilon, 1e-6));
            rawWeight[s] = w;
            weightSum += w;
        }

        if (weightSum <= EPS)
        {
            float equalW = 1.0 / max((float)usedCount, 1.0);
            [unroll] for (uint s = 0u; s < targetParentCount; s++)
            {
                if (s >= parentCount) break;
                if (selectedParent[s] < 0) continue;
                rawWeight[s] = equalW;
            }
            weightSum = 1.0;
        }

        [unroll] for (uint s = 0u; s < targetParentCount; s++)
        {
            if (s >= parentCount) break;
            int p = selectedParent[s];
            if (p < 0) continue;
            _ParentIndices[parentBase + s] = p;
            _ParentWeights[parentBase + s] = rawWeight[s] / weightSum;
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

        uint ownerGiList[targetParentCount];
        float ownerWeightList[targetParentCount];
        uint ownerCount = 0u;
        float ownerWeightSum = 0.0;
        [unroll] for (uint initIdx = 0u; initIdx < targetParentCount; initIdx++)
        {
            ownerGiList[initIdx] = ~0u;
            ownerWeightList[initIdx] = 0.0;
        }

        if (li < _ActiveCount)
        {
            ownerGiList[0] = gi;
            ownerWeightList[0] = 1.0;
            ownerCount = 1u;
            ownerWeightSum = 1.0;
        }
        else
        {
            uint parentCount = ParentReadCount();
            [unroll] for (uint slot = 0u; slot < targetParentCount; slot++)
            {
                if (slot >= parentCount) break;
                int p = ReadParentBySlot(gi, slot);
                if (p < 0) continue;

                uint pLi = ~0u;
                pLi = LocalIndexFromGlobal((uint)p);
                if (pLi == ~0u || pLi >= _ActiveCount) continue;

                float w = max(ReadParentWeightBySlot(gi, slot), 0.0);
                if (w <= EPS) continue;

                ownerGiList[ownerCount] = (uint)p;
                ownerWeightList[ownerCount] = w;
                ownerWeightSum += w;
                ownerCount++;
            }

            if (ownerCount == 0u)
            {
                int pDom = ReadDominantParent(gi);
                if (pDom >= 0)
                {
                    uint pLi = ~0u;
                    pLi = LocalIndexFromGlobal((uint)pDom);
                    if (pLi != ~0u && pLi < _ActiveCount)
                    {
                        ownerGiList[0] = (uint)pDom;
                        ownerWeightList[0] = 1.0;
                        ownerCount = 1u;
                        ownerWeightSum = 1.0;
                    }
                }
            }
        }

        if (ownerCount == 0u || ownerWeightSum <= EPS) return;

        [unroll] for (uint oi = 0u; oi < targetParentCount; oi++)
        {
            if (oi >= ownerCount) break;
            uint ownerGi = ownerGiList[oi];
            if (ownerGi == ~0u) continue;

            float wNorm = ownerWeightList[oi] / ownerWeightSum;
            if (wNorm <= EPS) continue;

            if (IsFixedVertex(gi))
            {
                AtomicAddFloatBits(_FixedChildPosBits, ownerGi * 2u + 0u, _Pos[gi].x * wNorm);
                AtomicAddFloatBits(_FixedChildPosBits, ownerGi * 2u + 1u, _Pos[gi].y * wNorm);
                AtomicAddFloatBits(_FixedChildCount, ownerGi, wNorm);
            }
            if (leafMass > EPS) AtomicAddFloatBits(_CurrentTotalMassBits, ownerGi, leafMass * wNorm);
            if (leafVol > EPS) AtomicAddFloatBits(_CurrentVolumeBits, ownerGi, leafVol * wNorm);
        }
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
        _CoarseFixed[gi] = (asfloat(_FixedChildCount[gi]) > 1.0) ? 1u : 0u;
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

            int p = ReadDominantParent(curGi);
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

        uint liOwner = ~0u;
        liOwner = LocalIndexFromGlobal(gi);
        if (liOwner != ~0u && liOwner < _ActiveCount)
        {
            AtomicAddFloatBits(_RestrictedDeltaVBits, gi * 2u + 0u, leafDeltaV.x);
            AtomicAddFloatBits(_RestrictedDeltaVBits, gi * 2u + 1u, leafDeltaV.y);
            InterlockedAdd(_RestrictedDeltaVCount[gi], 1u);
            return;
        }

        uint parentCount = ParentReadCount();
        float weightSum = 0.0;
        [unroll] for (uint slot = 0u; slot < targetParentCount; slot++)
        {
            if (slot >= parentCount) break;
            int p = ReadParentBySlot(gi, slot);
            if (p < 0) continue;

            uint pLi = ~0u;
            pLi = LocalIndexFromGlobal((uint)p);
            if (pLi == ~0u || pLi >= _ActiveCount) continue;

            float w = max(ReadParentWeightBySlot(gi, slot), 0.0);
            weightSum += w;
        }
        if (weightSum <= EPS) return;

        [unroll] for (uint slot = 0u; slot < targetParentCount; slot++)
        {
            if (slot >= parentCount) break;
            int p = ReadParentBySlot(gi, slot);
            if (p < 0) continue;

            uint pGi = (uint)p;
            uint pLi = ~0u;
            pLi = LocalIndexFromGlobal(pGi);
            if (pLi == ~0u || pLi >= _ActiveCount) continue;

            float w = max(ReadParentWeightBySlot(gi, slot), 0.0) / weightSum;
            if (w <= EPS) continue;

            AtomicAddFloatBits(_RestrictedDeltaVBits, pGi * 2u + 0u, leafDeltaV.x * w);
            AtomicAddFloatBits(_RestrictedDeltaVBits, pGi * 2u + 1u, leafDeltaV.y * w);
            InterlockedAdd(_RestrictedDeltaVCount[pGi], 1u);
        }
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

        uint parentCount = ParentReadCount();
        float weightSum = 0.0;
        [unroll] for (uint slot = 0u; slot < targetParentCount; slot++)
        {
            if (slot >= parentCount) break;
            int p = ReadParentBySlot(gi, slot);
            if (p < int(_Base) || p >= int(_Base + _ActiveCount)) continue;
            weightSum += max(ReadParentWeightBySlot(gi, slot), 0.0);
        }
        if (weightSum <= EPS) return;

        [unroll] for (uint slot = 0u; slot < targetParentCount; slot++)
        {
            if (slot >= parentCount) break;
            int p = ReadParentBySlot(gi, slot);
            if (p < int(_Base) || p >= int(_Base + _ActiveCount)) continue;

            uint ownerGi = (uint)p;
            float w = max(ReadParentWeightBySlot(gi, slot), 0.0) / weightSum;
            if (w <= EPS) continue;

            float2 residual = _Vel[gi] - _Vel[ownerGi];
            AtomicAddFloatBits(_RestrictedDeltaVBits, ownerGi * 2u + 0u, residual.x * w);
            AtomicAddFloatBits(_RestrictedDeltaVBits, ownerGi * 2u + 1u, residual.y * w);
            InterlockedAdd(_RestrictedDeltaVCount[ownerGi], 1u);
        }
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