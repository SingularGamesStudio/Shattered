#ifndef XPBI_SOLVER_RELAX_KERNELS_INCLUDED
    #define XPBI_SOLVER_RELAX_KERNELS_INCLUDED

    static bool IsLayerFixed(uint gi)
    {
        if (IsFixedVertex(gi))
        return true;

        // Only meaningful for active vertices in this layer.
        uint li = LocalIndexFromGlobal(gi);
        if (li != ~0u && li < _ActiveCount)
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
            uint gjLi = LocalIndexFromGlobal(gj);
            if (gjLi == ~0u || gjLi >= _ActiveCount)
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
        uint childLi = _ActiveCount + li;
        if (childLi >= _FineCount)
        return;

        uint childGi = GlobalIndexFromLocal(childLi);
        if (childGi == ~0u)
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

        uint gi = GlobalIndexFromLocal(li);
        if (gi == ~0u)
        return;
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

    // ----------------------------------------------------------------------------
    // Commit deformation: update F and Fp after velocity integration
    // ----------------------------------------------------------------------------
    [numthreads(256, 1, 1)]
    void CommitDeformation(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        if (li >= _ActiveCount)
        return;

        uint gi = GlobalIndexFromLocal(li);
        if (gi == ~0u)
        return;

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

        if (idx >= count) return;

        uint start = _ColorStarts[_ColorIndex];
        uint li = _ColorOrder[start + idx];
        if (li >= _ActiveCount) return;

        uint gi = GlobalIndexFromLocal(li);
        if (gi == ~0u) return;

        if (IsLayerFixed(gi)) return;
        if (_RestVolume[gi] <= EPS) return;

        float h = max(_LayerKernelH, 1e-4);
        float kernelH = WendlandKernelHFromSupport(h);
        if (kernelH <= EPS) return;

        float support = WendlandSupportRadius(h);
        if (support <= EPS) return;

        uint dtLi = (_UseDtGlobalNodeMap != 0u) ? li : (_DtLocalBase + li);
        uint rawCount = _DtNeighborCounts[dtLi];
        uint nCount = min(rawCount, _DtNeighborCount);
        nCount = min(nCount, targetNeighborCount);
        if (nCount == 0u) return;

        uint baseIdx = dtLi * _DtNeighborCount;

        bool useOwnerFilter = (_UseDtOwnerFilter != 0u);
        int ownerI = useOwnerFilter ? _DtOwnerByLocal[li] : -1;

        float supportSq = support * support;

        float2 xi = _Pos[gi];
        float2 vi = _Vel[gi];

        Mat2 Lm = Mat2FromFloat4(_L[gi]);

        Mat2 gradV = Mat2Zero();
        {
            [loop] for (uint k = 0u; k < nCount; k++)
            {
                uint gjLi = _DtNeighbors[baseIdx + k];
                if (gjLi == ~0u || gjLi >= _ActiveCount) continue;

                if (useOwnerFilter && _DtOwnerByLocal[gjLi] != ownerI) continue;

                uint gj = GlobalIndexFromLocal(gjLi);
                if (gj == ~0u) continue;

                float2 xij = _Pos[gj] - xi;
                if (dot(xij, xij) > supportSq) continue;

                float2 gradW = GradWendlandC2(xij, kernelH, EPS);
                float gradW2 = dot(gradW, gradW);
                if (gradW2 <= EPS * EPS) continue;

                float Vb = ReadCurrentVolume(gj);
                if (Vb <= EPS) continue;

                float2 correctedGrad = MulMat2Vec(Lm, gradW);
                float2 dv = _Vel[gj] - vi;

                gradV.c0 += dv * (Vb * correctedGrad.x);
                gradV.c1 += dv * (Vb * correctedGrad.y);
            }
        }

        Mat2 F0 = Mat2FromFloat4(_F0[gi]);
        Mat2 I = Mat2Identity();

        Mat2 dF = Mat2FromCols(I.c0 + gradV.c0 * _Dt, I.c1 + gradV.c1 * _Dt);
        if (DetMat2(dF) <= 0.0) return;

        Mat2 Ftrial = MulMat2(dF, F0);

        float yieldHencky = ReadMaterialYieldHencky(gi);
        float volHenckyLimit = ReadMaterialVolHenckyLimit(gi);
        Mat2 Fel = ApplyPlasticityReturn(Ftrial, yieldHencky, volHenckyLimit,
        STRETCH_EPS, EIGEN_OFFDIAG_EPS, INV_DET_EPS);
        if (!all(isfinite(Fel.c0)) || !all(isfinite(Fel.c1))) return;

        float mu, lambda;
        ComputeMaterialLame(gi, mu, lambda);

        float C = XPBI_ConstraintC(Fel, mu, lambda, STRETCH_EPS, EIGEN_OFFDIAG_EPS, INV_DET_EPS);
        if (!isfinite(C)) return;

        if (_ConvergenceDebugEnable != 0)
        {
            uint baseIter = _ConvergenceDebugOffset + _ConvergenceDebugIter;
            uint baseU = baseIter * CONV_DEBUG_UINTS_PER_ITER;

            uint uAbsC = (uint)min(abs(C) * _ConvergenceDebugScaleC, 4294967295.0);
            InterlockedAdd(_ConvergenceDebug[baseU + 0], uAbsC);
            InterlockedMax(_ConvergenceDebug[baseU + 1], uAbsC);
            InterlockedAdd(_ConvergenceDebug[baseU + 4], 1u);
        }

        if (abs(C) < EPS) return;
        if (abs(C) > 5.0) return;

        Mat2 dCdF = XPBI_ComputeGradient(Fel, mu, lambda, STRETCH_EPS, EIGEN_OFFDIAG_EPS, INV_DET_EPS);
        Mat2 FT = TransposeMat2(Fel);

        float invDt = 1.0 / max(_Dt, EPS);
        float invDt2 = invDt * invDt;
        float alphaTilde = (_Compliance / EffectiveVolumeForCompliance(gi)) * invDt2;

        float2 gradC_vi = 0.0;
        float denomNeighbors = 0.0;
        float maxInvMassLocal = ReadEffectiveInvMass(gi);
        float maxGradNorm2Local = 0.0;
        uint valid = 0u;

        [loop] for (uint k = 0u; k < nCount; k++)
        {
            uint gjLi = _DtNeighbors[baseIdx + k];
            if (gjLi == ~0u || gjLi >= _ActiveCount) continue;

            if (useOwnerFilter && _DtOwnerByLocal[gjLi] != ownerI) continue;

            uint gj = GlobalIndexFromLocal(gjLi);
            if (gj == ~0u) continue;

            float2 xij = _Pos[gj] - xi;
            if (dot(xij, xij) > supportSq) continue;

            float2 gradW = GradWendlandC2(xij, kernelH, EPS);
            if (dot(gradW, gradW) <= EPS * EPS) continue;

            float Vb = ReadCurrentVolume(gj);
            if (Vb <= EPS) continue;

            float2 correctedGrad = MulMat2Vec(Lm, gradW);
            float2 t = MulMat2Vec(FT, correctedGrad);
            float2 q = Vb * MulMat2Vec(dCdF, t);

            gradC_vi -= q;
            valid++;

            float invMassJ = IsLayerFixed(gj) ? 0.0 : ReadEffectiveInvMass(gj);
            float q2 = dot(q, q);

            denomNeighbors += invMassJ * q2;
            maxInvMassLocal = max(maxInvMassLocal, invMassJ);
            maxGradNorm2Local = max(maxGradNorm2Local, q2);
        }

        if (valid < 3u) return;

        float invMassI = ReadEffectiveInvMass(gi);
        float gradNormI2 = dot(gradC_vi, gradC_vi);
        if (!(gradNormI2 > 1e-8)) return;

        float denom = invMassI * gradNormI2 + denomNeighbors;
        if (denom < 1e-4) return;

        float lambdaBefore = _Lambda[gi];
        float dLambda = -(C + alphaTilde * lambdaBefore) / (denom + alphaTilde);
        if (isnan(dLambda) || isinf(dLambda)) return;
        if (abs(dLambda) > 100.0) return;

        if (_ConvergenceDebugEnable != 0)
        {
            uint baseIter = _ConvergenceDebugOffset + _ConvergenceDebugIter;
            uint baseU = baseIter * CONV_DEBUG_UINTS_PER_ITER;

            uint uAbsDL = (uint)min(abs(dLambda) * _ConvergenceDebugScaleDLambda, 4294967295.0);
            InterlockedAdd(_ConvergenceDebug[baseU + 2], uAbsDL);
            InterlockedMax(_ConvergenceDebug[baseU + 3], uAbsDL);
        }

        float velScale = dLambda * invDt;

        float maxDeltaVPerIter = support * invDt;
        float maxSpeedLocal = (4.0 * support) * invDt;

        float pred2 = (velScale * velScale) * (maxInvMassLocal * maxInvMassLocal) * max(maxGradNorm2Local, 1e-12);
        float maxDv2 = maxDeltaVPerIter * maxDeltaVPerIter;
        float maxSpeedHalf2 = (0.5 * maxSpeedLocal) * (0.5 * maxSpeedLocal);
        if (pred2 > maxDv2) return;
        if (pred2 > maxSpeedHalf2) return;

        float2 dVi = invMassI * velScale * gradC_vi;
        float dVi2 = dot(dVi, dVi);
        if (dVi2 > maxDv2)
        {
            float invLen = rsqrt(max(dVi2, EPS * EPS));
            dVi *= maxDeltaVPerIter * invLen;
        }

        float2 vI = _Vel[gi] + dVi;
        float vI2 = dot(vI, vI);
        float maxSpeed2 = maxSpeedLocal * maxSpeedLocal;
        if (vI2 > maxSpeed2)
        {
            float invLen = rsqrt(max(vI2, EPS * EPS));
            vI *= maxSpeedLocal * invLen;
        }
        _Vel[gi] = vI;

        [loop] for (uint k = 0u; k < nCount; k++)
        {
            uint gjLi = _DtNeighbors[baseIdx + k];
            if (gjLi == ~0u || gjLi >= _ActiveCount) continue;

            if (useOwnerFilter && _DtOwnerByLocal[gjLi] != ownerI) continue;

            uint gj = GlobalIndexFromLocal(gjLi);
            if (gj == ~0u) continue;
            if (IsLayerFixed(gj)) continue;

            float2 xij = _Pos[gj] - xi;
            if (dot(xij, xij) > supportSq) continue;

            float2 gradW = GradWendlandC2(xij, kernelH, EPS);
            if (dot(gradW, gradW) <= EPS * EPS) continue;

            float Vb = ReadCurrentVolume(gj);
            if (Vb <= EPS) continue;

            float2 correctedGrad = MulMat2Vec(Lm, gradW);
            float2 t = MulMat2Vec(FT, correctedGrad);
            float2 q = Vb * MulMat2Vec(dCdF, t);

            float invMassJ = ReadEffectiveInvMass(gj);
            float2 dVj = invMassJ * velScale * q;

            float dVj2 = dot(dVj, dVj);
            if (dVj2 > maxDv2)
            {
                float invLen = rsqrt(max(dVj2, EPS * EPS));
                dVj *= maxDeltaVPerIter * invLen;
            }

            float2 vJ = _Vel[gj] + dVj;
            float vJ2 = dot(vJ, vJ);
            if (vJ2 > maxSpeed2)
            {
                float invLen = rsqrt(max(vJ2, EPS * EPS));
                vJ *= maxSpeedLocal * invLen;
            }
            _Vel[gj] = vJ;
        }

        if (_CollisionEnable != 0u)
        {
            int ownerI = (li < _ActiveCount) ? _DtOwnerByLocal[li] : -1;
            if (ownerI >= 0)
            {
                float hCol = max(_LayerKernelH, 1e-4);
                float supportCol = WendlandSupportRadius(hCol);

                float targetSeparation = max(EPS, _CollisionSupportScale * supportCol);
                float targetSeparationSq = targetSeparation * targetSeparation;

                float alphaCollision = _CollisionCompliance / max(_Dt * _Dt, EPS);

                uint dtLiCol = (_UseDtGlobalNodeMap != 0u) ? li : (_DtLocalBase + li);
                uint baseIdxCol = dtLiCol * _DtNeighborCount;

                uint rawCountCol = _DtNeighborCounts[dtLiCol];
                uint nCountCol = min(rawCountCol, _DtNeighborCount);
                nCountCol = min(nCountCol, targetNeighborCount);

                uint contactBase = dtLiCol * _DtNeighborCount;

                bool fixedI = IsLayerFixed(gi);
                float invMassICol = fixedI ? 0.0 : ReadEffectiveInvMass(gi);

                [loop] for (uint k = 0u; k < nCountCol; k++)
                {
                    uint gjLi = _DtNeighbors[baseIdxCol + k];
                    if (gjLi == ~0u || gjLi >= _ActiveCount) continue;

                    int ownerJ = _DtOwnerByLocal[gjLi];
                    if (ownerJ < 0 || ownerJ == ownerI) continue;

                    uint gj = GlobalIndexFromLocal(gjLi);
                    if (gj == ~0u || gj <= gi) continue;

                    bool fixedJ = IsLayerFixed(gj);
                    if (fixedI && fixedJ) continue;

                    float2 dx = _Pos[gj] - _Pos[gi];
                    float distSq = dot(dx, dx);
                    if (distSq <= EPS * EPS || distSq > targetSeparationSq) continue;

                    float dist = sqrt(distSq);
                    float Cn = dist - targetSeparation;
                    if (Cn >= 0.0) continue;

                    float2 nrm = dx / max(dist, EPS);

                    float invMassJCol = fixedJ ? 0.0 : ReadEffectiveInvMass(gj);
                    float denomCol = invMassICol + invMassJCol;
                    if (denomCol <= EPS) continue;

                    uint lambdaIdx = contactBase + k;
                    float lambdaPrev = _CollisionLambda[lambdaIdx];

                    float dLambdaCol = -(Cn + alphaCollision * lambdaPrev) / (denomCol + alphaCollision);
                    float lambdaNew = max(0.0, lambdaPrev + dLambdaCol);
                    dLambdaCol = lambdaNew - lambdaPrev;
                    if (dLambdaCol <= 0.0) continue;

                    float2 relVelBefore = _Vel[gj] - _Vel[gi];
                    float relNormalBefore = dot(relVelBefore, nrm);

                    float impulseVel = dLambdaCol / max(_Dt, EPS);

                    if (!fixedI) _Vel[gi] -= invMassICol * impulseVel * nrm;
                    if (!fixedJ) _Vel[gj] += invMassJCol * impulseVel * nrm;

                    if (_CollisionFriction > 0.0)
                    {
                        float2 relVel = _Vel[gj] - _Vel[gi];
                        float relNormal = dot(relVel, nrm);
                        float2 tangentVel = relVel - relNormal * nrm;
                        float tangentLen = length(tangentVel);
                        if (tangentLen > EPS)
                        {
                            float2 tangentDir = tangentVel / tangentLen;
                            float frictionImpulseVel = -tangentLen / max(denomCol, EPS);
                            float frictionLimit = _CollisionFriction * impulseVel;
                            frictionImpulseVel = clamp(frictionImpulseVel, -frictionLimit, frictionLimit);

                            if (!fixedI) _Vel[gi] -= invMassICol * frictionImpulseVel * tangentDir;
                            if (!fixedJ) _Vel[gj] += invMassJCol * frictionImpulseVel * tangentDir;
                        }
                    }

                    if (_CollisionRestitution > 0.0 && relNormalBefore < -_CollisionRestitutionThreshold)
                    {
                        float restitutionImpulseVel =
                        (-(1.0 + _CollisionRestitution) * relNormalBefore) / max(denomCol, EPS);

                        if (!fixedI) _Vel[gi] -= invMassICol * restitutionImpulseVel * nrm;
                        if (!fixedJ) _Vel[gj] += invMassJCol * restitutionImpulseVel * nrm;
                    }

                    _CollisionLambda[lambdaIdx] = lambdaNew;
                }
            }
        }


        _Lambda[gi] = lambdaBefore + dLambda;

        uint fixedChildCount = ReadFixedChildCount(gi);
        if (fixedChildCount == 1u)
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

    [numthreads(256,1,1)]
    void JR_SavePrevAndClear(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        if (li >= _ActiveCount) return;

        uint gi = GlobalIndexFromLocal(li);
        if (gi == ~0u) return;

        _VelPrev[gi] = _Vel[gi];
        _LambdaPrev[gi] = _Lambda[gi];

        uint jrBase = gi * 2u;
        _JRVelDeltaBits[jrBase + 0u] = 0u;
        _JRVelDeltaBits[jrBase + 1u] = 0u;
        _JRLambdaDelta[gi] = 0.0;
    }


    [numthreads(256,1,1)]
    void JR_ComputeDeltas(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;

        if (_ConvergenceDebugEnable != 0 && li == 0)
        {
            uint baseIter = _ConvergenceDebugOffset + _ConvergenceDebugIter;
            uint baseU = baseIter * CONV_DEBUG_UINTS_PER_ITER;
            InterlockedAdd(_ConvergenceDebug[baseU + 7], 1u);
        }

        if (li >= _ActiveCount) return;

        uint gi = GlobalIndexFromLocal(li);
        if (gi == ~0u) return;

        if (IsLayerFixed(gi)) return;
        if (_RestVolume[gi] <= EPS) return;

        float h = max(_LayerKernelH, 1e-4);
        float kernelH = WendlandKernelHFromSupport(h);
        if (kernelH <= EPS) return;

        float support = WendlandSupportRadius(h);
        if (support <= EPS) return;

        uint dtLi = (_UseDtGlobalNodeMap != 0u) ? li : (_DtLocalBase + li);
        uint rawCount = _DtNeighborCounts[dtLi];
        uint nCount = min(rawCount, _DtNeighborCount);
        nCount = min(nCount, targetNeighborCount);
        if (nCount == 0u) return;

        uint baseIdx = dtLi * _DtNeighborCount;

        bool useOwnerFilter = (_UseDtOwnerFilter != 0u);
        int ownerI = useOwnerFilter ? _DtOwnerByLocal[li] : -1;

        float supportSq = support * support;

        float2 xi = _Pos[gi];
        float2 vi = _VelPrev[gi];

        Mat2 Lm = Mat2FromFloat4(_L[gi]);

        Mat2 gradV = Mat2Zero();
        [loop] for (uint k = 0u; k < nCount; k++)
        {
            uint gjLi = _DtNeighbors[baseIdx + k];
            if (gjLi == ~0u || gjLi >= _ActiveCount) continue;
            if (useOwnerFilter && _DtOwnerByLocal[gjLi] != ownerI) continue;

            uint gj = GlobalIndexFromLocal(gjLi);
            if (gj == ~0u) continue;

            float2 xij = _Pos[gj] - xi;
            if (dot(xij, xij) > supportSq) continue;

            float2 gradW = GradWendlandC2(xij, kernelH, EPS);
            if (dot(gradW, gradW) <= EPS * EPS) continue;

            float Vb = ReadCurrentVolume(gj);
            if (Vb <= EPS) continue;

            float2 correctedGrad = MulMat2Vec(Lm, gradW);
            float2 dv = _VelPrev[gj] - vi;

            gradV.c0 += dv * (Vb * correctedGrad.x);
            gradV.c1 += dv * (Vb * correctedGrad.y);
        }

        Mat2 F0 = Mat2FromFloat4(_F0[gi]);
        Mat2 I = Mat2Identity();
        Mat2 dF = Mat2FromCols(I.c0 + gradV.c0 * _Dt, I.c1 + gradV.c1 * _Dt);
        if (DetMat2(dF) <= 0.0) return;

        Mat2 Ftrial = MulMat2(dF, F0);

        float yieldHencky = ReadMaterialYieldHencky(gi);
        float volHenckyLimit = ReadMaterialVolHenckyLimit(gi);
        Mat2 Fel = ApplyPlasticityReturn(Ftrial, yieldHencky, volHenckyLimit,
        STRETCH_EPS, EIGEN_OFFDIAG_EPS, INV_DET_EPS);
        if (!all(isfinite(Fel.c0)) || !all(isfinite(Fel.c1))) return;

        float mu, lambda;
        ComputeMaterialLame(gi, mu, lambda);

        float C = XPBI_ConstraintC(Fel, mu, lambda, STRETCH_EPS, EIGEN_OFFDIAG_EPS, INV_DET_EPS);
        if (!isfinite(C)) return;

        if (_ConvergenceDebugEnable != 0)
        {
            uint baseIter = _ConvergenceDebugOffset + _ConvergenceDebugIter;
            uint baseU = baseIter * CONV_DEBUG_UINTS_PER_ITER;

            uint uAbsC = (uint)min(abs(C) * _ConvergenceDebugScaleC, 4294967295.0);
            InterlockedAdd(_ConvergenceDebug[baseU + 0], uAbsC);
            InterlockedMax(_ConvergenceDebug[baseU + 1], uAbsC);
            InterlockedAdd(_ConvergenceDebug[baseU + 4], 1u);
        }

        if (abs(C) < EPS) return;
        if (abs(C) > 5.0) return;

        Mat2 dCdF = XPBI_ComputeGradient(Fel, mu, lambda, STRETCH_EPS, EIGEN_OFFDIAG_EPS, INV_DET_EPS);
        Mat2 FT = TransposeMat2(Fel);

        float invDt = 1.0 / max(_Dt, EPS);
        float invDt2 = invDt * invDt;
        float alphaTilde = (_Compliance / EffectiveVolumeForCompliance(gi)) * invDt2;

        float2 gradC_vi = 0.0;
        float denomNeighbors = 0.0;
        float maxInvMassLocal = ReadEffectiveInvMass(gi);
        float maxGradNorm2Local = 0.0;
        uint valid = 0u;

        [loop] for (uint k = 0u; k < nCount; k++)
        {
            uint gjLi = _DtNeighbors[baseIdx + k];
            if (gjLi == ~0u || gjLi >= _ActiveCount) continue;
            if (useOwnerFilter && _DtOwnerByLocal[gjLi] != ownerI) continue;

            uint gj = GlobalIndexFromLocal(gjLi);
            if (gj == ~0u) continue;

            float2 xij = _Pos[gj] - xi;
            if (dot(xij, xij) > supportSq) continue;

            float2 gradW = GradWendlandC2(xij, kernelH, EPS);
            if (dot(gradW, gradW) <= EPS * EPS) continue;

            float Vb = ReadCurrentVolume(gj);
            if (Vb <= EPS) continue;

            float2 correctedGrad = MulMat2Vec(Lm, gradW);
            float2 t = MulMat2Vec(FT, correctedGrad);
            float2 q = Vb * MulMat2Vec(dCdF, t);

            gradC_vi -= q;
            valid++;

            float invMassJ = IsLayerFixed(gj) ? 0.0 : ReadEffectiveInvMass(gj);
            float q2 = dot(q, q);
            denomNeighbors += invMassJ * q2;
            maxInvMassLocal = max(maxInvMassLocal, invMassJ);
            maxGradNorm2Local = max(maxGradNorm2Local, q2);
        }

        if (valid < 3u) return;

        float invMassI = ReadEffectiveInvMass(gi);
        float gradNormI2 = dot(gradC_vi, gradC_vi);
        if (!(gradNormI2 > 1e-8)) return;

        float denom = invMassI * gradNormI2 + denomNeighbors;
        if (denom < 1e-4) return;

        float lambdaBefore = _LambdaPrev[gi];
        float dLambda = -(C + alphaTilde * lambdaBefore) / (denom + alphaTilde);
        if (!isfinite(dLambda)) return;
        if (abs(dLambda) > 100.0) return;

        if (_ConvergenceDebugEnable != 0)
        {
            uint baseIter = _ConvergenceDebugOffset + _ConvergenceDebugIter;
            uint baseU = baseIter * CONV_DEBUG_UINTS_PER_ITER;

            uint uAbsDL = (uint)min(abs(dLambda) * _ConvergenceDebugScaleDLambda, 4294967295.0);
            InterlockedAdd(_ConvergenceDebug[baseU + 2], uAbsDL);
            InterlockedMax(_ConvergenceDebug[baseU + 3], uAbsDL);
        }

        float velScale = dLambda * invDt;

        float maxDeltaVPerIter = support * invDt;
        float maxSpeedLocal = (4.0 * support) * invDt;

        float pred2 = (velScale * velScale) * (maxInvMassLocal * maxInvMassLocal) * max(maxGradNorm2Local, 1e-12);
        float maxDv2 = maxDeltaVPerIter * maxDeltaVPerIter;
        float maxSpeedHalf2 = (0.5 * maxSpeedLocal) * (0.5 * maxSpeedLocal);
        if (pred2 > maxDv2) return;
        if (pred2 > maxSpeedHalf2) return;

        // Scatter deltas into scratch (atomics)
        float2 dVi = invMassI * velScale * gradC_vi;
        float dVi2 = dot(dVi, dVi);
        if (dVi2 > maxDv2) dVi *= maxDeltaVPerIter * rsqrt(max(dVi2, EPS*EPS));
        AtomicAddFloat2(_JRVelDeltaBits, gi, dVi);
        _JRLambdaDelta[gi] = dLambda;

        [loop] for (uint k = 0u; k < nCount; k++)
        {
            uint gjLi = _DtNeighbors[baseIdx + k];
            if (gjLi == ~0u || gjLi >= _ActiveCount) continue;
            if (useOwnerFilter && _DtOwnerByLocal[gjLi] != ownerI) continue;

            uint gj = GlobalIndexFromLocal(gjLi);
            if (gj == ~0u) continue;
            if (IsLayerFixed(gj)) continue;

            float2 xij = _Pos[gj] - xi;
            if (dot(xij, xij) > supportSq) continue;

            float2 gradW = GradWendlandC2(xij, kernelH, EPS);
            if (dot(gradW, gradW) <= EPS * EPS) continue;

            float Vb = ReadCurrentVolume(gj);
            if (Vb <= EPS) continue;

            float2 correctedGrad = MulMat2Vec(Lm, gradW);
            float2 t = MulMat2Vec(FT, correctedGrad);
            float2 q = Vb * MulMat2Vec(dCdF, t);

            float invMassJ = ReadEffectiveInvMass(gj);
            float2 dVj = invMassJ * velScale * q;

            float dVj2 = dot(dVj, dVj);
            if (dVj2 > maxDv2) dVj *= maxDeltaVPerIter * rsqrt(max(dVj2, EPS*EPS));

            AtomicAddFloat2(_JRVelDeltaBits, gj, dVj);
        }
    }

    [numthreads(256,1,1)]
    void JR_Apply(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        if (li >= _ActiveCount) return;

        uint gi = GlobalIndexFromLocal(li);
        if (gi == ~0u) return;

        uint jrBase = gi * 2u;
        float2 dV = float2(asfloat(_JRVelDeltaBits[jrBase + 0u]), asfloat(_JRVelDeltaBits[jrBase + 1u]));
        float  dL = _JRLambdaDelta[gi];

        float omegaV = saturate(_JROmegaV);
        float omegaL = min(saturate(_JROmegaL), omegaV);
        _Vel[gi] += omegaV * dV;
        _Lambda[gi] += omegaL * dL;

        float h = max(_LayerKernelH, 1e-4);
        float support = WendlandSupportRadius(h);
        float maxSpeedLocal = (4.0 * support) / max(_Dt, EPS);
        float maxSpeed2 = maxSpeedLocal * maxSpeedLocal;
        float v2 = dot(_Vel[gi], _Vel[gi]);
        if (v2 > maxSpeed2)
        {
            float invLen = rsqrt(max(v2, EPS * EPS));
            _Vel[gi] *= maxSpeedLocal * invLen;
        }

        uint fixedChildCount = ReadFixedChildCount(gi);
        if (fixedChildCount == 1u)
        {
            float mu, lambda;
            ComputeMaterialLame(gi, mu, lambda);

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