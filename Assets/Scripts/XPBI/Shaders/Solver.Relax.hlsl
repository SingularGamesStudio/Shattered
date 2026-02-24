#ifndef XPBI_SOLVER_RELAX_KERNELS_INCLUDED
    #define XPBI_SOLVER_RELAX_KERNELS_INCLUDED

    static bool IsLayerFixed(uint gi)
    {
        if (IsFixedVertex(gi))
        return true;

        // Only meaningful for active vertices in this layer.
        if (gi >= _Base && gi < _Base + _ActiveCount)
        return _CoarseFixed[gi] != 0u;

        return false;
    }

    static float EffectiveVolumeForCompliance(uint gi)
    {
        float currentVol = ReadCurrentVolume(gi);
        if (currentVol > EPS)
        return currentVol;
        return max(_RestVolume[gi], EPS);
    }

    // ----------------------------------------------------------------------------
    // Helper: estimate velocity gradient using SPH
    // ----------------------------------------------------------------------------
    static Mat2 EstimateVelocityGradient(uint gi, float2 xi, float2 vi, Mat2 Lm, float h)
    {
        Mat2 gradV = Mat2Zero();
        if (h <= EPS)
        return gradV;

        float kernelH = WendlandKernelHFromSupport(h);
        if (kernelH <= EPS)
        return gradV;

        uint nCount;
        uint ns[targetNeighborCount];
        GetNeighbors(gi, nCount, ns);
        if (nCount == 0)
        return gradV;

        [unroll] for (uint k = 0; k < nCount; k++)
        {
            uint gj = ns[k];
            if (gj < _Base || gj >= _Base + _ActiveCount)
            continue;

            float2 xij = _Pos[gj] - xi;
            float2 gradW = GradWendlandC2(xij, kernelH, EPS);
            if (dot(gradW, gradW) <= EPS * EPS)
            continue;

            float Vb = ReadCurrentVolume(gj);
            if (Vb <= EPS)
            continue;

            float2 correctedGrad = MulMat2Vec(Lm, gradW);
            float2 dv = _Vel[gj] - vi;

            gradV.c0 += dv * (Vb * correctedGrad.x);
            gradV.c1 += dv * (Vb * correctedGrad.y);
        }

        return gradV;
    }

    // ----------------------------------------------------------------------------
    // Prolongation for multigrid: add parent velocity change to child
    // ----------------------------------------------------------------------------
    [numthreads(256, 1, 1)]
    void Prolongate(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        uint childGi = _Base + (_ActiveCount + li);
        if (childGi >= _Base + _FineCount)
        return;

        if (IsFixedVertex(childGi))
        return;

        int p = _ParentIndex[childGi]; // may be -1
        if (p < int(_Base) || p >= int(_Base + _ActiveCount))
        return;

        uint parent = uint(p);
        float2 parentDeltaV = _Vel[parent] - _SavedVelPrefix[parent];

        uint parentFixedCount = ReadFixedChildCount(parent);
        if (parentFixedCount == 1u)
        {
            float2 anchor = ReadFixedChildAnchor(parent);
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

        uint gi = _Base + li;
        if (IsFixedVertex(gi))
        return;

        float w = saturate(_PostProlongSmoothing);
        if (w <= 0.0)
        return;

        uint nCount;
        uint ns[targetNeighborCount];
        GetNeighbors(gi, nCount, ns);
        if (nCount == 0)
        return;

        float2 sum = 0.0;
        float total = 0.0;

        [unroll] for (uint k = 0; k < nCount; k++)
        {
            uint gj = ns[k];
            if (gj < _Base || gj >= _Base + _FineCount)
            continue;

            sum += _Vel[gj];
            total += 1.0;
        }

        if (total <= 0.0)
        return;

        float2 avg = sum / total;
        _Vel[gi] = lerp(_Vel[gi], avg, w);
    }

    // ----------------------------------------------------------------------------
    // Commit deformation: update F and Fp after velocity integration
    // ----------------------------------------------------------------------------
    [numthreads(256, 1, 1)]
    void CommitDeformation(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        if (li >= _ActiveCount)
        return;

        uint gi = _Base + li;

        uint fixedChildCount = ReadFixedChildCount(gi);
        bool singleFixedAnchor = (fixedChildCount == 1u);
        float2 fixedAnchor = singleFixedAnchor ? ReadFixedChildAnchor(gi) : 0.0;

        if (IsLayerFixed(gi))
        return;

        float h = max(_LayerKernelH, 1e-4);
        if (h <= EPS)
        return;

        float2 xi = _Pos[gi];
        float2 vi = _Vel[gi];

        Mat2 Lm = Mat2FromFloat4(_L[gi]);
        Mat2 gradV = EstimateVelocityGradient(gi, xi, vi, Lm, h);

        Mat2 F0 = Mat2FromFloat4(_F0[gi]);
        Mat2 I = Mat2Identity();

        Mat2 dF = Mat2FromCols(I.c0 + gradV.c0 * _Dt, I.c1 + gradV.c1 * _Dt);
        Mat2 Ftrial = MulMat2(dF, F0);
        float yieldHencky = ReadMaterialYieldHencky(gi);
        float volHenckyLimit = ReadMaterialVolHenckyLimit(gi);
        Mat2 Fel = ApplyPlasticityReturn(Ftrial, yieldHencky, volHenckyLimit,
        STRETCH_EPS, EIGEN_OFFDIAG_EPS, INV_DET_EPS);

        float4 Fel4 = Float4FromMat2(Fel);
        if (any(isnan(Fel4)) || any(abs(Fel4) > 1e6))
        return;

        Mat2 FpOld = Mat2FromFloat4(_Fp[gi]);

        float detFel = DetMat2(Fel);
        if (abs(detFel) > EPS && !isnan(detFel) && abs(detFel) < 1e6)
        {
            float invDet = 1.0 / detFel;

            Mat2 FelInv;
            FelInv.c0 = float2(Fel.c1.y * invDet, -Fel.c0.y * invDet);
            FelInv.c1 = float2(-Fel.c1.x * invDet, Fel.c0.x * invDet);

            Mat2 FpNew = MulMat2(MulMat2(FelInv, Ftrial), FpOld);
            float4 Fp4 = Float4FromMat2(FpNew);
            if (!any(isnan(Fp4)) && !any(abs(Fp4) > 1e6))
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
        uint i = id.x;
        if (i >= _ConvergenceDebugCopyCount)
        return;

        _ConvergenceDebugDst[i] = _ConvergenceDebugSrc[i];
    }

    // ----------------------------------------------------------------------------
    // Convergence debug (optional)
    // Layout per-iteration: [sumAbsC, maxAbsC, sumAbsDLambda, maxAbsDLambda, count, 0, 0, 0]
    // ----------------------------------------------------------------------------
    #define CONV_DEBUG_UINTS_PER_ITER 8u

    RWStructuredBuffer<uint> _ConvergenceDebug;
    int   _ConvergenceDebugEnable;
    uint  _ConvergenceDebugOffset;       // in "iterations" (not uint elements); baseIter = offset + iter
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

    // ----------------------------------------------------------------------------
    // RelaxColored: XPBI iteration for a single colour
    // ----------------------------------------------------------------------------
    [numthreads(256, 1, 1)]
    void RelaxColored(uint3 id : SV_DispatchThreadID)
    {
        uint idx = id.x;
        uint count = _ColorCounts[_ColorIndex];

        if (_ConvergenceDebugEnable != 0 && idx == 0)
        {
            uint baseIter = _ConvergenceDebugOffset + _ConvergenceDebugIter;
            uint baseU = baseIter * CONV_DEBUG_UINTS_PER_ITER;
            InterlockedAdd(_ConvergenceDebug[baseU + 7], 1u);
        }
        if (idx >= count)
        return;

        uint start = _ColorStarts[_ColorIndex];
        uint li = _ColorOrder[start + idx];
        if (li >= _ActiveCount)
        return;

        uint gi = _Base + li;

        if (IsLayerFixed(gi))
        return;
        if (_RestVolume[gi] <= EPS)
        return;

        float h = max(_LayerKernelH, 1e-4);
        if (h <= EPS)
        return;

        uint nCount;
        uint ns[targetNeighborCount];
        GetNeighbors(gi, nCount, ns);
        if (nCount == 0)
        return;

        float2 xi = _Pos[gi];
        float2 vi = _Vel[gi];

        Mat2 Lm = Mat2FromFloat4(_L[gi]);
        Mat2 gradV = EstimateVelocityGradient(gi, xi, vi, Lm, h);

        Mat2 F0 = Mat2FromFloat4(_F0[gi]);
        Mat2 I = Mat2Identity();

        Mat2 dF = Mat2FromCols(I.c0 + gradV.c0 * _Dt, I.c1 + gradV.c1 * _Dt);
        if (DetMat2(dF) <= 0.0)
        return;

        Mat2 Ftrial = MulMat2(dF, F0);
        float yieldHencky = ReadMaterialYieldHencky(gi);
        float volHenckyLimit = ReadMaterialVolHenckyLimit(gi);
        Mat2 Fel = ApplyPlasticityReturn(Ftrial, yieldHencky, volHenckyLimit,
        STRETCH_EPS, EIGEN_OFFDIAG_EPS, INV_DET_EPS);
        if (!all(isfinite(Fel.c0)) || !all(isfinite(Fel.c1)))
        return;

        float mu, lambda;
        ComputeMaterialLame(gi, mu, lambda);

        float C = XPBI_ConstraintC(Fel, mu, lambda, STRETCH_EPS, EIGEN_OFFDIAG_EPS, INV_DET_EPS);
        if (!isfinite(C))
        return;

        if (_ConvergenceDebugEnable != 0)
        {
            uint baseIter = _ConvergenceDebugOffset + _ConvergenceDebugIter;
            uint baseU = baseIter * CONV_DEBUG_UINTS_PER_ITER;

            uint uAbsC = (uint)min(abs(C) * _ConvergenceDebugScaleC, 4294967295.0);

            InterlockedAdd(_ConvergenceDebug[baseU + 0], uAbsC);
            InterlockedMax(_ConvergenceDebug[baseU + 1], uAbsC);
            InterlockedAdd(_ConvergenceDebug[baseU + 4], 1u);
        }

        if (abs(C) < EPS)
        return;
        if (abs(C) > 5.0)
        return;

        Mat2 dCdF = XPBI_ComputeGradient(Fel, mu, lambda, STRETCH_EPS, EIGEN_OFFDIAG_EPS, INV_DET_EPS);
        Mat2 FT = TransposeMat2(Fel);

        float invDt = 1.0 / max(_Dt, EPS);
        float alphaTilde = (_Compliance / EffectiveVolumeForCompliance(gi)) * (invDt * invDt);

        float2 gradC_vi = 0.0f;
        float2 gradC_vj[targetNeighborCount];
        [unroll] for (uint kInit = 0; kInit < targetNeighborCount; kInit++)
        gradC_vj[kInit] = 0.0f;
        uint validConstraintNeighbors = 0u;

        [unroll] for (uint k = 0; k < nCount; k++)
        {
            uint gj = ns[k];
            if (gj < _Base || gj >= _Base + _ActiveCount) continue;

            float2 xij = _Pos[gj] - xi;
            float2 gradW = GradWendlandC2(xij, WendlandKernelHFromSupport(h), EPS);
            if (dot(gradW, gradW) <= EPS * EPS) continue;

            float Vb = ReadCurrentVolume(gj);
            if (Vb <= EPS) continue;

            float2 correctedGrad = MulMat2Vec(Lm, gradW);
            float2 t = MulMat2Vec(FT, correctedGrad);
            float2 q = Vb * MulMat2Vec(dCdF, t);

            gradC_vi -= q;
            gradC_vj[k] = q;
            validConstraintNeighbors++;
        }

        if (validConstraintNeighbors < 3u)
        return;

        float invMassI = ReadEffectiveInvMass(gi);
        float gradNormI2 = dot(gradC_vi, gradC_vi);
        if (!(gradNormI2 > 1e-8))
        return;

        float denom = invMassI * gradNormI2;
        float maxInvMassLocal = invMassI;
        float maxGradNorm2Local = gradNormI2;
        [unroll] for (uint k1 = 0; k1 < nCount; k1++)
        {
            uint gj = ns[k1];
            if (gj < _Base || gj >= _Base + _ActiveCount)
            continue;
            if (IsLayerFixed(gj))
            continue;
            float invMassJ = ReadEffectiveInvMass(gj);
            float gradNormJ2 = dot(gradC_vj[k1], gradC_vj[k1]);
            denom += invMassJ * gradNormJ2;
            maxInvMassLocal = max(maxInvMassLocal, invMassJ);
            maxGradNorm2Local = max(maxGradNorm2Local, gradNormJ2);
        }
        if (denom < 1e-4)
        return;

        float lambdaBefore = _Lambda[gi];
        float dLambda = -(C + alphaTilde * lambdaBefore) / (denom + alphaTilde);
        if (isnan(dLambda) || isinf(dLambda))
        return;
        if (abs(dLambda) > 100.0)
        return;

        if (_ConvergenceDebugEnable != 0)
        {
            uint baseIter = _ConvergenceDebugOffset + _ConvergenceDebugIter;
            uint baseU = baseIter * CONV_DEBUG_UINTS_PER_ITER;

            uint uAbsDL = (uint)min(abs(dLambda) * _ConvergenceDebugScaleDLambda, 4294967295.0);

            InterlockedAdd(_ConvergenceDebug[baseU + 2], uAbsDL);
            InterlockedMax(_ConvergenceDebug[baseU + 3], uAbsDL);
        }

        float velScale = dLambda * invDt;
        float support = WendlandSupportRadius(h);
        float maxDeltaVPerIter = support * invDt;
        float maxSpeedLocal = (4.0 * support) * invDt;

        float predictedMaxDv = abs(velScale) * maxInvMassLocal * sqrt(max(maxGradNorm2Local, 1e-12));
        if (predictedMaxDv > maxDeltaVPerIter)
        return;
        if (predictedMaxDv > 0.5 * maxSpeedLocal)
        return;

        float2 dVi = invMassI * velScale * gradC_vi;
        float dViLen = length(dVi);
        if (dViLen > maxDeltaVPerIter)
        dVi *= maxDeltaVPerIter / max(dViLen, EPS);
        _Vel[gi] += dVi;
        float vILen = length(_Vel[gi]);
        if (vILen > maxSpeedLocal)
        _Vel[gi] *= maxSpeedLocal / max(vILen, EPS);

        [unroll] for (uint k2 = 0; k2 < nCount; k2++)
        {
            uint gj = ns[k2];
            if (gj < _Base || gj >= _Base + _ActiveCount)
            continue;
            if (IsLayerFixed(gj))
            continue;
            float invMassJ = ReadEffectiveInvMass(gj);
            float2 dVj = invMassJ * velScale * gradC_vj[k2];
            float dVjLen = length(dVj);
            if (dVjLen > maxDeltaVPerIter)
            dVj *= maxDeltaVPerIter / max(dVjLen, EPS);
            _Vel[gj] += dVj;
            float vJLen = length(_Vel[gj]);
            if (vJLen > maxSpeedLocal)
            _Vel[gj] *= maxSpeedLocal / max(vJLen, EPS);
        }

        _Lambda[gi] = lambdaBefore + dLambda;

        uint fixedChildCount = ReadFixedChildCount(gi);
        bool singleFixedAnchor = (fixedChildCount == 1u);

        if (singleFixedAnchor)
        {
            float2 fixedAnchor = ReadFixedChildAnchor(gi);
            float2 r = _Pos[gi] - fixedAnchor;
            float rLen = length(r);
            if (rLen > EPS)
            {
                float2 radial = r / rLen;
                float vr = dot(_Vel[gi], radial);
                float radialKeep = saturate(_Compliance / (_Compliance + (_Dt * _Dt) * (mu + lambda) / EffectiveVolumeForCompliance(gi)));
                _Vel[gi] -= radial * vr * (1.0 - radialKeep);
            }
        }
    }

#endif // XPBI_SOLVER_RELAX_KERNELS_INCLUDED