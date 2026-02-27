#ifndef XPBI_SOLVER_RELAX_ATOMIC_INCLUDED
    #define XPBI_SOLVER_RELAX_ATOMIC_INCLUDED

    [numthreads(256,1,1)]
    void JR_ComputeDeltas(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;

        if (_ConvergenceDebugEnable != 0 && li == 0u)
        {
            uint baseIter = _ConvergenceDebugOffset + _ConvergenceDebugIter;
            uint baseU = baseIter * CONV_DEBUG_UINTS_PER_ITER;
            InterlockedAdd(_ConvergenceDebug[baseU + 7], 1u);
        }

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
        #define XPBI_VEL(li_, gi_) _VelPrev[gi_]
        #define XPBI_SET_VEL(li_, gi_, v_) ((void)0)
        #define XPBI_LAMBDA(li_, gi_) _LambdaPrev[gi_]
        #define XPBI_SET_LAMBDA(li_, gi_, l_) ((void)0)
        #define XPBI_L_FROM_I(li_, gi_) _L[gi_]
        #define XPBI_F0_FROM_I(li_, gi_) _F0[gi_]
        #define XPBI_NEIGHBOR_FIXED(gjLi_, gj_) IsLayerFixed(gj_)
        #define XPBI_INV_MASS(gjLi_, gj_) ReadEffectiveInvMass(gj_)
        #define XPBI_ACTIVE_I(li_, gi_) (!IsLayerFixed(gi_) && _RestVolume[gi_] > EPS)
        #define XPBI_APPLY_MODE_JR 1
        #define XPBI_SCATTER_DV(gi_, dv_) AtomicAddFloat2(_JRVelDeltaBits, gi_, (dv_))
        #define XPBI_SCATTER_DL(gi_, dl_) (_JRLambdaDelta[gi_] = (dl_))

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

    [numthreads(256,1,1)]
    void JR_SavePrevAndClear(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        if (li >= _ActiveCount) return;

        uint gi = ~0u;
        gi = GlobalIndexFromLocal(li);
        if (gi == ~0u) return;
        if (IsLayerFixed(gi)) return;

        _VelPrev[gi] = _Vel[gi];
        _LambdaPrev[gi] = _Lambda[gi];

        uint jrBase = gi * 2u;
        _JRVelDeltaBits[jrBase + 0u] = 0u;
        _JRVelDeltaBits[jrBase + 1u] = 0u;
        _JRLambdaDelta[gi] = 0.0;
    }

    [numthreads(256,1,1)]
    void JR_Apply(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        if (li >= _ActiveCount) return;

        uint gi = ~0u;
        gi = GlobalIndexFromLocal(li);
        if (gi == ~0u) return;
        if (IsLayerFixed(gi)) return;

        uint jrBase = gi * 2u;
        float2 dV = float2(asfloat(_JRVelDeltaBits[jrBase + 0u]), asfloat(_JRVelDeltaBits[jrBase + 1u]));
        float  dL = _JRLambdaDelta[gi];

        float omegaV = saturate(_JROmegaV);
        float omegaL = min(saturate(_JROmegaL), omegaV);
        _Vel[gi] += omegaV * dV;
        _Lambda[gi] += omegaL * dL;

        float h = max(_LayerKernelH, 1e-4);
        float support = 0.0;
        support = WendlandSupportRadius(h);
        float maxSpeedLocal = (4.0 * support) / max(_Dt, EPS);
        float maxSpeed2 = maxSpeedLocal * maxSpeedLocal;
        float v2 = dot(_Vel[gi], _Vel[gi]);
        if (v2 > maxSpeed2)
        {
            float invLen = rsqrt(max(v2, EPS * EPS));
            _Vel[gi] *= maxSpeedLocal * invLen;
        }

        float mu = 0.0, lambda = 0.0;
        ComputeMaterialLame(gi, mu, lambda);
        float2 dampedVel = _Vel[gi];
        ApplySingleAnchorRadialDampingOnVel(gi, mu, lambda, _Pos[gi], dampedVel);
        _Vel[gi] = dampedVel;
    }

#endif // XPBI_SOLVER_RELAX_ATOMIC_INCLUDED