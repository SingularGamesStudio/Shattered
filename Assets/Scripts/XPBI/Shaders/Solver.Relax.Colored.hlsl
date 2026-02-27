#ifndef XPBI_SOLVER_RELAX_COLORED_INCLUDED
    #define XPBI_SOLVER_RELAX_COLORED_INCLUDED

    // ----------------------------------------------------------------------------
    // RelaxColored: XPBI iteration for a single colour
    // ----------------------------------------------------------------------------
    [numthreads(256, 1, 1)]
    void RelaxColored(uint3 id : SV_DispatchThreadID)
    {
        uint idx = id.x;
        uint count = _ColorCounts[_ColorIndex];

        if (_ConvergenceDebugEnable != 0 && idx == 0u)
        {
            uint baseIter = _ConvergenceDebugOffset + _ConvergenceDebugIter;
            uint baseU = baseIter * CONV_DEBUG_UINTS_PER_ITER;
            InterlockedAdd(_ConvergenceDebug[baseU + 7], 1u);
        }

        if (idx >= count) return;

        uint start = _ColorStarts[_ColorIndex];
        uint li = _ColorOrder[start + idx];
        uint active = _ActiveCount;
        if (li >= active) return;

        uint gi = ~0u;
        gi = GlobalIndexFromLocal(li);
        if (gi == ~0u) return;

        float h = max(_LayerKernelH, 1e-4);
        float kernelH = 0.0;
        kernelH = WendlandKernelHFromSupport(h);
        if (kernelH <= EPS) return;

        float support = 0.0;
        support = WendlandSupportRadius(h);
        if (support <= EPS) return;
        float supportSq = support * support;

        float invDt = 1.0 / max(_Dt, EPS);
        float invDt2 = invDt * invDt;

        uint dtLi = (_UseDtGlobalNodeMap != 0u) ? li : (_DtLocalBase + li);
        uint rawCount = _DtNeighborCounts[dtLi];
        uint nCount = 0u;
        nCount = min(rawCount, _DtNeighborCount);
        nCount = min(nCount, targetNeighborCount);
        if (nCount == 0u) return;

        uint baseIdx = dtLi * _DtNeighborCount;

        bool useOwnerFilter = (_UseDtOwnerFilter != 0u);
        int ownerI = useOwnerFilter ? _DtOwnerByLocal[li] : -1;

        #define XPBI_GET_GJ(gjLi) GlobalIndexFromLocal(gjLi)
        #define XPBI_POS(li_, gi_) _Pos[gi_]
        #define XPBI_VEL(li_, gi_) _Vel[gi_]
        #define XPBI_SET_VEL(li_, gi_, v_) (_Vel[gi_] = (v_))
        #define XPBI_LAMBDA(li_, gi_) _Lambda[gi_]
        #define XPBI_SET_LAMBDA(li_, gi_, l_) (_Lambda[gi_] = (l_))
        #define XPBI_L_FROM_I(li_, gi_) _L[gi_]
        #define XPBI_F0_FROM_I(li_, gi_) _F0[gi_]
        #define XPBI_NEIGHBOR_FIXED(gjLi_, gj_) IsLayerFixed(gj_)
        #define XPBI_INV_MASS(gjLi_, gj_) ReadEffectiveInvMass(gj_)
        #define XPBI_ACTIVE_I(li_, gi_) (!IsLayerFixed(gi_) && _RestVolume[gi_] > EPS)
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

#endif // XPBI_SOLVER_RELAX_COLORED_INCLUDED