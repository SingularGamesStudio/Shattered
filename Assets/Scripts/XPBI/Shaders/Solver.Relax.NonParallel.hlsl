#ifndef XPBI_SOLVER_RELAX_NON_PARALLEL_INCLUDED
    #define XPBI_SOLVER_RELAX_NON_PARALLEL_INCLUDED

    #define COARSE_MAX_N 256u

    groupshared uint   _SCoarseGi[COARSE_MAX_N];
    groupshared uint   _SCoarseFlags[COARSE_MAX_N];
    groupshared uint   _SCoarseColor[COARSE_MAX_N];
    groupshared float2 _SCoarsePos[COARSE_MAX_N];
    groupshared float2 _SCoarseVel[COARSE_MAX_N];
    groupshared float  _SCoarseLambda[COARSE_MAX_N];
    groupshared float4 _SCoarseL[COARSE_MAX_N];
    groupshared float4 _SCoarseF0[COARSE_MAX_N];

    static void RelaxPersistentCoarseRow(
    uint li,
    uint gi,
    uint active,
    float kernelH,
    float support,
    float supportSq,
    float invDt,
    float invDt2)
    {
        uint dtLi = (_UseDtGlobalNodeMap != 0u) ? li : (_DtLocalBase + li);
        uint rawCount = _DtNeighborCounts[dtLi];
        uint nCount = 0u;
        nCount = min(rawCount, _DtNeighborCount);
        nCount = min(nCount, targetNeighborCount);
        if (nCount == 0u) return;

        uint baseIdx = dtLi * _DtNeighborCount;
        bool useOwnerFilter = (_UseDtOwnerFilter != 0u);
        int ownerI = useOwnerFilter ? _DtOwnerByLocal[li] : -1;

        #define XPBI_GET_GJ(gjLi) _SCoarseGi[gjLi]
        #define XPBI_POS(li_, gi_) _SCoarsePos[li_]
        #define XPBI_VEL(li_, gi_) _SCoarseVel[li_]
        #define XPBI_SET_VEL(li_, gi_, v_) (_SCoarseVel[li_] = (v_))
        #define XPBI_LAMBDA(li_, gi_) _SCoarseLambda[li_]
        #define XPBI_SET_LAMBDA(li_, gi_, l_) (_SCoarseLambda[li_] = (l_))
        #define XPBI_L_FROM_I(li_, gi_) _SCoarseL[li_]
        #define XPBI_F0_FROM_I(li_, gi_) _SCoarseF0[li_]
        #define XPBI_NEIGHBOR_FIXED(gjLi_, gj_) ((_SCoarseFlags[gjLi_] & 1u) != 0u)
        #define XPBI_INV_MASS(gjLi_, gj_) ReadEffectiveInvMass(gj_)
        #define XPBI_ACTIVE_I(li_, gi_) ((_SCoarseFlags[li_] & 2u) != 0u)
        #define XPBI_APPLY_MODE_JR 0
        #define XPBI_SCATTER_DV(gi_, dv_) ((void)0)
        #define XPBI_SCATTER_DL(gi_, dl_) ((void)0)

        #include "Solver.Relax.Core.hlsl"

        #undef XPBI_SCATTER_DL
        #undef XPBI_SCATTER_DV
        #undef XPBI_APPLY_MODE_JR
        #undef XPBI_ACTIVE_I
        #undef XPBI_INV_MASS
        #undef XPBI_NEIGHBOR_FIXED
        #undef XPBI_F0_FROM_I
        #undef XPBI_L_FROM_I
        #undef XPBI_SET_LAMBDA
        #undef XPBI_LAMBDA
        #undef XPBI_SET_VEL
        #undef XPBI_VEL
        #undef XPBI_POS
        #undef XPBI_GET_GJ
    }

    [numthreads(256, 1, 1)]
    void RelaxColoredPersistentCoarse(uint3 gtid : SV_GroupThreadID)
    {
        uint li = gtid.x;
        uint active = _ActiveCount;
        if (active > COARSE_MAX_N)
        {
            return;
        }

        bool inRange = li < active;
        if (li < COARSE_MAX_N)
        {
            _SCoarseGi[li] = ~0u;
            _SCoarseFlags[li] = 0u;
            _SCoarseColor[li] = 0xFFFFFFFFu;
            _SCoarsePos[li] = 0.0;
            _SCoarseVel[li] = 0.0;
            _SCoarseLambda[li] = 0.0;
            _SCoarseL[li] = 0.0;
            _SCoarseF0[li] = 0.0;
        }

        if (inRange)
        {
            uint gi = ~0u;
            gi = GlobalIndexFromLocal(li);
            _SCoarseGi[li] = gi;
            if (gi != ~0u)
            {
                bool fixedI = false;
                fixedI = IsLayerFixed(gi);
                bool activeI = (!fixedI) && (_RestVolume[gi] > EPS);
                _SCoarseFlags[li] = (fixedI ? 1u : 0u) | (activeI ? 2u : 0u);
                _SCoarsePos[li] = _Pos[gi];
                _SCoarseVel[li] = _Vel[gi];
                _SCoarseLambda[li] = _Lambda[gi];
                _SCoarseL[li] = _L[gi];
                _SCoarseF0[li] = _F0[gi];
            }
        }

        GroupMemoryBarrierWithGroupSync();

        if (li == 0u)
        {
            [loop] for (uint c = 0u; c < 16u; c++)
            {
                uint count = _ColorCounts[c];
                uint start = _ColorStarts[c];
                [loop] for (uint idx = 0u; idx < count; idx++)
                {
                    uint l = _ColorOrder[start + idx];
                    if (l < active) _SCoarseColor[l] = c;
                }
            }
        }

        GroupMemoryBarrierWithGroupSync();

        float h = max(_LayerKernelH, 1e-4);
        float kernelH = 0.0;
        kernelH = WendlandKernelHFromSupport(h);
        float support = 0.0;
        support = WendlandSupportRadius(h);
        if (kernelH <= EPS || support <= EPS)
        {
            if (inRange)
            {
                uint gi = _SCoarseGi[li];
                if (gi != ~0u)
                {
                    _Vel[gi] = _SCoarseVel[li];
                    _Lambda[gi] = _SCoarseLambda[li];
                }
            }
            return;
        }

        float supportSq = support * support;
        float invDt = 1.0 / max(_Dt, EPS);
        float invDt2 = invDt * invDt;

        [loop] for (uint iter = 0u; iter < _PersistentIters; iter++)
        {
            if (_ConvergenceDebugEnable != 0 && li == 0u)
            {
                uint baseIter = _ConvergenceDebugOffset + (_PersistentBaseDebugIter + iter);
                uint baseU = baseIter * CONV_DEBUG_UINTS_PER_ITER;
                InterlockedAdd(_ConvergenceDebug[baseU + 7], 1u);
            }

            [loop] for (uint c = 0u; c < 16u; c++)
            {
                if (inRange && _SCoarseColor[li] == c && (_SCoarseFlags[li] & 2u) != 0u)
                {
                    uint gi = _SCoarseGi[li];
                    if (gi != ~0u)
                    {
                        RelaxPersistentCoarseRow(li, gi, active, kernelH, support, supportSq, invDt, invDt2);
                    }
                }

                GroupMemoryBarrierWithGroupSync();
            }
        }

        if (inRange)
        {
            uint gi = _SCoarseGi[li];
            if (gi != ~0u)
            {
                _Vel[gi] = _SCoarseVel[li];
                _Lambda[gi] = _SCoarseLambda[li];
            }
        }
    }

#endif // XPBI_SOLVER_RELAX_NON_PARALLEL_INCLUDED
