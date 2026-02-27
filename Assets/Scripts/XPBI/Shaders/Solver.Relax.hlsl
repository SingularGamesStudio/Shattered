#ifndef XPBI_SOLVER_RELAX_KERNELS_INCLUDED
    #define XPBI_SOLVER_RELAX_KERNELS_INCLUDED

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

    static Mat2 EstimateVelocityGradient(uint gi, float2 xi, float2 vi, Mat2 Lm, float h)
    {
        Mat2 gradV = (Mat2)0;
        gradV = Mat2Zero();
        uint neighborCount = 0u;
        uint ns[targetNeighborCount];
        [unroll] for (uint initIdx1 = 0u; initIdx1 < targetNeighborCount; initIdx1++) ns[initIdx1] = ~0u;

        if (h > EPS)
        {
            float kernelH = 0.0;
            kernelH = WendlandKernelHFromSupport(h);
            if (kernelH > EPS)
            {
                GetNeighbors(gi, neighborCount, ns);

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
                    if (Vb <= EPS)
                    continue;

                    float2 correctedGrad = MulMat2Vec(Lm, gradW);
                    float2 dv = _Vel[gj] - vi;

                    gradV.c0 += dv * (Vb * correctedGrad.x);
                    gradV.c1 += dv * (Vb * correctedGrad.y);
                }
            }
        }

        return gradV;
    }

    [numthreads(256, 1, 1)]
    void Prolongate(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        uint childLi = _ActiveCount + li;
        if (childLi >= _FineCount)
        return;

        uint childGi = ~0u;
        childGi = GlobalIndexFromLocal(childLi);
        if (childGi == ~0u)
        return;

        if (IsFixedVertex(childGi))
        return;

        int p = _ParentIndex[childGi];
        if (p < int(_Base) || p >= int(_Base + _ActiveCount))
        return;

        uint parent = uint(p);
        float2 parentDeltaV = _Vel[parent] - _SavedVelPrefix[parent];

        uint parentFixedCount = 0u;
        parentFixedCount = ReadFixedChildCount(parent);
        if (parentFixedCount == 1u)
        {
            float2 anchor = 0.0;
            anchor = ReadFixedChildAnchor(parent);
            float2 rp = _Pos[parent] - anchor;
            float rpLen = length(rp);
            if (rpLen > EPS)
            {
                float parentMu, parentLambda;
                ComputeMaterialLame(parent, parentMu, parentLambda);

                float2 radial = rp / rpLen;
                float radialDV = dot(parentDeltaV, radial);
                float radialKeep = saturate(_Compliance / (_Compliance + (_Dt * _Dt) * (parentMu + parentLambda) / EffectiveVolumeForCompliance(parent)));
                parentDeltaV -= radial * radialDV * (1.0 - radialKeep);

                if (_UseAffineProlongation != 0u)
                {
                    float2 rc = _Pos[childGi] - anchor;
                    float invRpSq = 1.0 / max(dot(rp, rp), EPS);
                    float omega = (rp.x * parentDeltaV.y - rp.y * parentDeltaV.x) * invRpSq;

                    float2 rotParent = omega * float2(-rp.y, rp.x);
                    float2 transl = parentDeltaV - rotParent;
                    float2 rotChild = omega * float2(-rc.y, rc.x);

                    parentDeltaV = transl + rotChild;
                }
            }
        }

        _Vel[childGi] += parentDeltaV * _ProlongationScale;
    }

    [numthreads(256, 1, 1)]
    void SmoothProlongatedFineVel(uint3 id : SV_DispatchThreadID)
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

        float w = saturate(_PostProlongSmoothing);
        if (w <= 0.0)
        return;

        uint neighborCount = 0u;
        uint ns[targetNeighborCount];
        [unroll] for (uint initIdx2 = 0u; initIdx2 < targetNeighborCount; initIdx2++) ns[initIdx2] = ~0u;
        GetNeighbors(gi, neighborCount, ns);

        float2 sum = 0.0;
        float total = 0.0;

        [unroll] for (uint k = 0u; k < targetNeighborCount; k++)
        {
            uint gj = ns[k];
            if (gj == ~0u)
            continue;

            sum += _Vel[gj];
            total += 1.0;
        }

        if (total <= 0.0)
        return;

        float2 avg = sum / total;
        _Vel[gi] = lerp(_Vel[gi], avg, w);
    }

    [numthreads(256, 1, 1)]
    void CommitDeformation(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        if (li >= _ActiveCount)
        return;

        uint gi = ~0u;
        gi = GlobalIndexFromLocal(li);
        if (gi == ~0u)
        return;

        if (IsLayerFixed(gi))
        return;

        float h = max(_LayerKernelH, 1e-4);
        if (h <= EPS)
        return;

        float2 xi = _Pos[gi];
        float2 vi = _Vel[gi];

        Mat2 Lm = (Mat2)0;
        Lm = Mat2FromFloat4(_L[gi]);
        Mat2 gradV = (Mat2)0;
        gradV = EstimateVelocityGradient(gi, xi, vi, Lm, h);

        Mat2 F0 = (Mat2)0;
        F0 = Mat2FromFloat4(_F0[gi]);
        Mat2 I = (Mat2)0;
        I = Mat2Identity();

        Mat2 dF = Mat2FromCols(I.c0 + gradV.c0 * _Dt, I.c1 + gradV.c1 * _Dt);
        Mat2 Ftrial = MulMat2(dF, F0);
        float yieldHencky = 0.0;
        yieldHencky = ReadMaterialYieldHencky(gi);
        float volHenckyLimit = 0.0;
        volHenckyLimit = ReadMaterialVolHenckyLimit(gi);
        Mat2 Fel = (Mat2)0;
        Fel = ApplyPlasticityReturn(Ftrial, yieldHencky, volHenckyLimit,
        STRETCH_EPS, EIGEN_OFFDIAG_EPS, INV_DET_EPS);

        float4 Fel4 = 0.0;
        Fel4 = Float4FromMat2(Fel);
        if (any(abs(Fel4) > 1e6))
        return;

        Mat2 FpOld = (Mat2)0;
        FpOld = Mat2FromFloat4(_Fp[gi]);

        float detFel = DetMat2(Fel);
        if (abs(detFel) > EPS && abs(detFel) < 1e6)
        {
            float invDet = 1.0 / detFel;

            Mat2 FelInv;
            FelInv.c0 = float2(Fel.c1.y * invDet, -Fel.c0.y * invDet);
            FelInv.c1 = float2(-Fel.c1.x * invDet, Fel.c0.x * invDet);

            Mat2 FpNew = MulMat2(MulMat2(FelInv, Ftrial), FpOld);
            float4 Fp4 = 0.0;
            Fp4 = Float4FromMat2(FpNew);
            if (!any(abs(Fp4) > 1e6))
            _Fp[gi] = Fp4;
        }

        _F[gi] = Fel4;
    }

    StructuredBuffer<uint> _ConvergenceDebugSrc;
    RWStructuredBuffer<uint> _ConvergenceDebugDst;
    uint _ConvergenceDebugCopyCount;

    [numthreads(256, 1, 1)]
    void CopyConvergenceDebug(uint3 id : SV_DispatchThreadID)
    {
        uint copyIdx = id.x;
        if (copyIdx >= _ConvergenceDebugCopyCount)
        return;

        _ConvergenceDebugDst[copyIdx] = _ConvergenceDebugSrc[copyIdx];
    }

    #define CONV_DEBUG_UINTS_PER_ITER 8u

    RWStructuredBuffer<uint> _ConvergenceDebug;
    int   _ConvergenceDebugEnable;
    uint  _ConvergenceDebugOffset;
    uint  _ConvergenceDebugIter;
    uint  _ConvergenceDebugIterCount;
    float _ConvergenceDebugScaleC;
    float _ConvergenceDebugScaleDLambda;

    [numthreads(256, 1, 1)]
    void ClearConvergenceDebugStats(uint3 id : SV_DispatchThreadID)
    {
        if (_ConvergenceDebugEnable == 0)
        return;

        uint iter = id.x;
        if (iter >= _ConvergenceDebugIterCount)
        return;

        uint baseIter = _ConvergenceDebugOffset + iter;
        uint baseU = baseIter * CONV_DEBUG_UINTS_PER_ITER;

        [unroll] for (uint k = 0; k < CONV_DEBUG_UINTS_PER_ITER; k++)
        _ConvergenceDebug[baseU + k] = 0u;
    }

    

    

#endif // XPBI_SOLVER_RELAX_KERNELS_INCLUDED
