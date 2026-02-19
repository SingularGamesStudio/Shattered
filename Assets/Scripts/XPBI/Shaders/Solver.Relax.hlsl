#ifndef XPBI_SOLVER_RELAX_KERNELS_INCLUDED
    #define XPBI_SOLVER_RELAX_KERNELS_INCLUDED

    // ----------------------------------------------------------------------------
    // Helper: estimate velocity gradient using SPH
    // ----------------------------------------------------------------------------
    static Mat2 EstimateVelocityGradient(uint gi, float2 xi, float2 vi, Mat2 Lm, float h)
    {
        Mat2 gradV = Mat2Zero();
        if (h <= EPS)
        return gradV;

        uint nCount, n0, n1, n2, n3, n4, n5;
        GetNeighbors(gi, nCount, n0, n1, n2, n3, n4, n5);
        if (nCount == 0)
        return gradV;
        
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
    [numthreads(256, 1, 1)] void Prolongate(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        uint childGi = _Base + (_ActiveCount + li);
        if (childGi >= _Base + _FineCount)
        return;

        int p = _ParentIndex[childGi]; // may be -1
        if (p < int(_Base) || p >= int(_Base + _ActiveCount))
        return;

        uint parent = uint(p); // safe now
        float2 parentDeltaV = _Vel[parent] - _SavedVelPrefix[parent];
        _Vel[childGi] += parentDeltaV;
    }

    // ----------------------------------------------------------------------------
    // Commit deformation: update F and Fp after velocity integration
    // ----------------------------------------------------------------------------
    [numthreads(256, 1, 1)] void CommitDeformation(uint3 id : SV_DispatchThreadID)
    {
        uint li = id.x;
        if (li >= _ActiveCount)
        return;

        uint gi = _Base + li;

        if (IsFixedVertex(gi))
        return;

        float h = _KernelH[gi];
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
        Mat2 Fel = ApplyPlasticityReturn(Ftrial, YIELD_HENCKY, VOL_HENCKY_LIMIT,
        STRETCH_EPS, EIGEN_OFFDIAG_EPS, INV_DET_EPS);

        Mat2 FpOld = Mat2FromFloat4(_Fp[gi]);

        float detFel = DetMat2(Fel);
        if (abs(detFel) > EPS)
        {
            float invDet = 1.0 / detFel;

            Mat2 FelInv;
            FelInv.c0 = float2(Fel.c1.y * invDet, -Fel.c0.y * invDet);
            FelInv.c1 = float2(-Fel.c1.x * invDet, Fel.c0.x * invDet);

            Mat2 FpNew = MulMat2(MulMat2(FelInv, Ftrial), FpOld);
            _Fp[gi] = Float4FromMat2(FpNew);
        }

        _F[gi] = Float4FromMat2(Fel);
    }

    // ----------------------------------------------------------------------------
    // RelaxColored: XPBI iteration for a single colour
    // ----------------------------------------------------------------------------
    [numthreads(256, 1, 1)] void RelaxColored(uint3 id : SV_DispatchThreadID)
    {
        uint idx = id.x;
        uint count = _ColorCounts[_ColorIndex];
        if (idx >= count)
        return;

        uint start = _ColorStarts[_ColorIndex];
        uint li = _ColorOrder[start + idx];
        if (li >= _ActiveCount)
        return;

        uint gi = _Base + li;

        if (IsFixedVertex(gi))
        return;
        if (_RestVolume[gi] <= EPS)
        return;

        float h = _KernelH[gi];
        if (h <= EPS)
        return;

        uint nCount, n0, n1, n2, n3, n4, n5;
        GetNeighbors(gi, nCount, n0, n1, n2, n3, n4, n5);
        if (nCount == 0)
        return;

        float2 xi = _Pos[gi];
        float2 vi = _Vel[gi];

        Mat2 Lm = Mat2FromFloat4(_L[gi]);
        Mat2 gradV = EstimateVelocityGradient(gi, xi, vi, Lm, h);

        Mat2 F0 = Mat2FromFloat4(_F0[gi]);
        Mat2 I = Mat2Identity();

        Mat2 dF = Mat2FromCols(I.c0 + gradV.c0 * _Dt, I.c1 + gradV.c1 * _Dt);
        Mat2 Ftrial = MulMat2(dF, F0);
        Mat2 Fel = ApplyPlasticityReturn(Ftrial, YIELD_HENCKY, VOL_HENCKY_LIMIT,
        STRETCH_EPS, EIGEN_OFFDIAG_EPS, INV_DET_EPS);

        float C = XPBI_ConstraintC(Fel, MU, LAMBDA, STRETCH_EPS, EIGEN_OFFDIAG_EPS, INV_DET_EPS);
        if (abs(C) < EPS)
        return;

        Mat2 dCdF = XPBI_ComputeGradient(Fel, MU, LAMBDA, STRETCH_EPS, EIGEN_OFFDIAG_EPS, INV_DET_EPS);
        Mat2 FT = TransposeMat2(Fel);

        float invDt = 1.0 / max(_Dt, EPS);
        float alphaTilde = (_Compliance / max(_RestVolume[gi], EPS)) * (invDt * invDt);

        float2 gradC_vi = 0.0f;
        float2 gradC_vj[targetNeighborCount] = {(float2)0, (float2)0, (float2)0, (float2)0, (float2)0, (float2)0};

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

            float2 correctedGrad = MulMat2Vec(Lm, gradW);
            float2 t = MulMat2Vec(FT, correctedGrad);
            float2 q = Vb * MulMat2Vec(dCdF, t);

            gradC_vi -= q;
            gradC_vj[k] = q;
        }

        float denom = _InvMass[gi] * dot(gradC_vi, gradC_vi);
        [unroll] for (uint k1 = 0; k1 < nCount; k1++)
        {
            uint gj = ns[k1];
            if (gj < _Base || gj >= _Base + _ActiveCount)
            continue;
            if (IsFixedVertex(gj))
            continue;
            denom += _InvMass[gj] * dot(gradC_vj[k1], gradC_vj[k1]);
        }
        if (denom < EPS)
        return;

        float lambdaBefore = _Lambda[gi];
        float dLambda = -(C + alphaTilde * lambdaBefore) / (denom + alphaTilde);
        if (isnan(dLambda) || isinf(dLambda))
        return;

        float velScale = dLambda * invDt;

        _Vel[gi] += _InvMass[gi] * velScale * gradC_vi;

        [unroll] for (uint k2 = 0; k2 < nCount; k2++)
        {
            uint gj = ns[k2];
            if (gj < _Base || gj >= _Base + _ActiveCount)
            continue;
            if (IsFixedVertex(gj))
            continue;
            _Vel[gj] += _InvMass[gj] * velScale * gradC_vj[k2];
        }

        _Lambda[gi] = lambdaBefore + dLambda;
    }

#endif // XPBI_SOLVER_RELAX_KERNELS_INCLUDED