#ifndef XPBI_SOLVER_RELAX_KERNELS_INCLUDED
#define XPBI_SOLVER_RELAX_KERNELS_INCLUDED

static XPBI_Mat2 EstimateVelocityGradient(int gi, float2 xi, float2 vi, XPBI_Mat2 Lm, float h)
{
    XPBI_Mat2 gradV = XPBI_Mat2Zero();
    if (h <= EPS)
        return gradV;

    int nCount, n0, n1, n2, n3, n4, n5;
    XPBI_GetNeighbors(gi, nCount, n0, n1, n2, n3, n4, n5);
    if (nCount <= 0)
        return gradV;

    int ns[targetNeighborCount];
    ns[0] = n0;
    ns[1] = n1;
    ns[2] = n2;
    ns[3] = n3;
    ns[4] = n4;
    ns[5] = n5;

    for (int k = 0; k < nCount; k++)
    {
        int gj = ns[k];
        if (gj < _Base || gj >= _Base + _ActiveCount)
            continue;

        float2 xij = _Pos[gj] - xi;

        float2 gradW = XPBI_GradWendlandC2(xij, h, EPS);
        if (dot(gradW, gradW) <= EPS * EPS)
            continue;

        float Vb = XPBI_ReadCurrentVolume(gj);
        if (Vb <= EPS)
            continue;

        float2 correctedGrad = XPBI_MulMat2Vec(Lm, gradW);
        float2 dv = _Vel[gj] - vi;

        gradV.c0 += dv * (Vb * correctedGrad.x);
        gradV.c1 += dv * (Vb * correctedGrad.y);
    }

    return gradV;
}

[numthreads(256, 1, 1)] void RelaxScatter(uint3 id : SV_DispatchThreadID)
{
    int li = (int)id.x;
    if (li >= _ActiveCount)
        return;

    int gi = _Base + li;

    if (XPBI_IsFixed(gi))
        return;
    if (_RestVolume[gi] <= EPS)
        return;

    float h = _KernelH[gi];
    if (h <= EPS)
        return;

    int nCount, n0, n1, n2, n3, n4, n5;
    XPBI_GetNeighbors(gi, nCount, n0, n1, n2, n3, n4, n5);
    if (nCount <= 0)
        return;

    float2 xi = _Pos[gi];
    float2 vi = _Vel[gi];

    XPBI_Mat2 Lm = XPBI_Mat2FromFloat4(_L[gi]);
    XPBI_Mat2 gradV = EstimateVelocityGradient(gi, xi, vi, Lm, h);

    XPBI_Mat2 F0 = XPBI_Mat2FromFloat4(_F0[gi]);
    XPBI_Mat2 I = XPBI_Mat2Identity();

    XPBI_Mat2 dF = XPBI_Mat2FromCols(I.c0 + gradV.c0 * _Dt, I.c1 + gradV.c1 * _Dt);
    XPBI_Mat2 Ftrial = XPBI_MulMat2(dF, F0);
    XPBI_Mat2 Fel = XPBI_ApplyPlasticityReturn(Ftrial, YIELD_HENCKY, VOL_HENCKY_LIMIT, STRETCH_EPS, EIGEN_OFFDIAG_EPS, INV_DET_EPS);

    float C = XPBI_ConstraintC(Fel, MU, LAMBDA, STRETCH_EPS, EIGEN_OFFDIAG_EPS, INV_DET_EPS);
    if (abs(C) < EPS)
        return;

    XPBI_Mat2 dCdF = XPBI_ComputeGradient(Fel, MU, LAMBDA, STRETCH_EPS, EIGEN_OFFDIAG_EPS, INV_DET_EPS);
    XPBI_Mat2 FT = XPBI_TransposeMat2(Fel);

    float invDt = 1.0 / max(_Dt, EPS);
    float alphaTilde = (_Compliance / max(_RestVolume[gi], EPS)) * (invDt * invDt);

    float2 gradC_vi = float2(0, 0);
    float2 gradC_vj[targetNeighborCount];
    gradC_vj[0] = float2(0, 0);
    gradC_vj[1] = float2(0, 0);
    gradC_vj[2] = float2(0, 0);
    gradC_vj[3] = float2(0, 0);
    gradC_vj[4] = float2(0, 0);
    gradC_vj[5] = float2(0, 0);

    int ns[targetNeighborCount];
    ns[0] = n0;
    ns[1] = n1;
    ns[2] = n2;
    ns[3] = n3;
    ns[4] = n4;
    ns[5] = n5;

    for (int k = 0; k < nCount; k++)
    {
        int gj = ns[k];
        if (gj < _Base || gj >= _Base + _ActiveCount)
            continue;

        float2 xij = _Pos[gj] - xi;

        float2 gradW = XPBI_GradWendlandC2(xij, h, EPS);
        if (dot(gradW, gradW) <= EPS * EPS)
            continue;

        float Vb = XPBI_ReadCurrentVolume(gj);
        if (Vb <= EPS)
            continue;

        float2 correctedGrad = XPBI_MulMat2Vec(Lm, gradW);
        float2 t = XPBI_MulMat2Vec(FT, correctedGrad);
        float2 q = Vb * XPBI_MulMat2Vec(dCdF, t);

        gradC_vi -= q;
        gradC_vj[k] = q;
    }

    float denom = _InvMass[gi] * dot(gradC_vi, gradC_vi);
    for (int k1 = 0; k1 < nCount; k1++)
    {
        int gj = ns[k1];
        if (gj < _Base || gj >= _Base + _ActiveCount)
            continue;
        if (XPBI_IsFixed(gj))
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

    XPBI_AddVelDelta(gi, _InvMass[gi] * velScale * gradC_vi);

    for (int k2 = 0; k2 < nCount; k2++)
    {
        int gj = ns[k2];
        if (gj < _Base || gj >= _Base + _ActiveCount)
            continue;
        if (XPBI_IsFixed(gj))
            continue;
        XPBI_AddVelDelta(gj, _InvMass[gj] * velScale * gradC_vj[k2]);
    }

    _Lambda[gi] = lambdaBefore + dLambda;
}

    [numthreads(256, 1, 1)] void ApplyVelDelta(uint3 id : SV_DispatchThreadID)
{
    int li = (int)id.x;
    if (li >= _ActiveCount)
        return;

    int gi = _Base + li;

    float2 dv = XPBI_ReadVelDelta(gi);
    _Vel[gi] = _Vel[gi] + dv;

    _VelDeltaBits[XPBI_VelDeltaIndex(gi, 0u)] = 0u;
    _VelDeltaBits[XPBI_VelDeltaIndex(gi, 1u)] = 0u;
}

[numthreads(256, 1, 1)] void Prolongate(uint3 id : SV_DispatchThreadID)
{
    int li = (int)id.x;
    int childGi = _Base + (_ActiveCount + li);
    if (childGi >= _Base + _FineCount)
        return;

    int p = _ParentIndex[childGi];
    if (p < _Base || p >= _Base + _ActiveCount)
        return;

    float2 parentDeltaV = _Vel[p] - _SavedVelPrefix[p];
    _Vel[childGi] = _Vel[childGi] + parentDeltaV;
}

    [numthreads(256, 1, 1)] void CommitDeformation(uint3 id : SV_DispatchThreadID)
{
    int li = (int)id.x;
    if (li >= _ActiveCount)
        return;

    int gi = _Base + li;

    if (XPBI_IsFixed(gi))
        return;

    float h = _KernelH[gi];
    if (h <= EPS)
        return;

    float2 xi = _Pos[gi];
    float2 vi = _Vel[gi];

    XPBI_Mat2 Lm = XPBI_Mat2FromFloat4(_L[gi]);
    XPBI_Mat2 gradV = EstimateVelocityGradient(gi, xi, vi, Lm, h);

    XPBI_Mat2 F0 = XPBI_Mat2FromFloat4(_F0[gi]);
    XPBI_Mat2 I = XPBI_Mat2Identity();

    XPBI_Mat2 dF = XPBI_Mat2FromCols(I.c0 + gradV.c0 * _Dt, I.c1 + gradV.c1 * _Dt);
    XPBI_Mat2 Ftrial = XPBI_MulMat2(dF, F0);
    XPBI_Mat2 Fel = XPBI_ApplyPlasticityReturn(Ftrial, YIELD_HENCKY, VOL_HENCKY_LIMIT, STRETCH_EPS, EIGEN_OFFDIAG_EPS, INV_DET_EPS);

    XPBI_Mat2 FpOld = XPBI_Mat2FromFloat4(_Fp[gi]);

    float detFel = XPBI_DetMat2(Fel);
    if (abs(detFel) > EPS)
    {
        float invDet = 1.0 / detFel;

        XPBI_Mat2 FelInv;
        FelInv.c0 = float2(Fel.c1.y * invDet, -Fel.c0.y * invDet);
        FelInv.c1 = float2(-Fel.c1.x * invDet, Fel.c0.x * invDet);

        XPBI_Mat2 FpNew = XPBI_MulMat2(XPBI_MulMat2(FelInv, Ftrial), FpOld);
        _Fp[gi] = XPBI_Float4FromMat2(FpNew);
    }

    _F[gi] = XPBI_Float4FromMat2(Fel);
}

[numthreads(256, 1, 1)] void RelaxColored(uint3 id : SV_DispatchThreadID)
{
    int idx = (int)id.x;
    if (idx >= _ColorCount)
        return;

    int li = _ColorOrder[_ColorStart + idx];
    if (li < 0 || li >= _ActiveCount)
        return;

    int gi = _Base + li;

    if (XPBI_IsFixed(gi))
        return;
    if (_RestVolume[gi] <= EPS)
        return;

    float h = _KernelH[gi];
    if (h <= EPS)
        return;

    int nCount, n0, n1, n2, n3, n4, n5;
    XPBI_GetNeighbors(gi, nCount, n0, n1, n2, n3, n4, n5);
    if (nCount <= 0)
        return;

    float2 xi = _Pos[gi];
    float2 vi = _Vel[gi];

    XPBI_Mat2 Lm = XPBI_Mat2FromFloat4(_L[gi]);
    XPBI_Mat2 gradV = EstimateVelocityGradient(gi, xi, vi, Lm, h);

    XPBI_Mat2 F0m = XPBI_Mat2FromFloat4(_F0[gi]);
    XPBI_Mat2 I = XPBI_Mat2Identity();

    XPBI_Mat2 dF = XPBI_Mat2FromCols(I.c0 + gradV.c0 * _Dt, I.c1 + gradV.c1 * _Dt);
    XPBI_Mat2 Ftrial = XPBI_MulMat2(dF, F0m);
    XPBI_Mat2 Fel = XPBI_ApplyPlasticityReturn(Ftrial, YIELD_HENCKY, VOL_HENCKY_LIMIT, STRETCH_EPS, EIGEN_OFFDIAG_EPS, INV_DET_EPS);

    float C = XPBI_ConstraintC(Fel, MU, LAMBDA, STRETCH_EPS, EIGEN_OFFDIAG_EPS, INV_DET_EPS);
    if (abs(C) < EPS)
        return;

    XPBI_Mat2 dCdF = XPBI_ComputeGradient(Fel, MU, LAMBDA, STRETCH_EPS, EIGEN_OFFDIAG_EPS, INV_DET_EPS);
    XPBI_Mat2 FT = XPBI_TransposeMat2(Fel);

    float invDt = 1.0 / max(_Dt, EPS);
    float alphaTilde = (_Compliance / max(_RestVolume[gi], EPS)) * (invDt * invDt);

    float2 gradC_vi = float2(0, 0);
    float2 gradC_vj[targetNeighborCount];
    gradC_vj[0] = float2(0, 0);
    gradC_vj[1] = float2(0, 0);
    gradC_vj[2] = float2(0, 0);
    gradC_vj[3] = float2(0, 0);
    gradC_vj[4] = float2(0, 0);
    gradC_vj[5] = float2(0, 0);

    int ns[targetNeighborCount];
    ns[0] = n0;
    ns[1] = n1;
    ns[2] = n2;
    ns[3] = n3;
    ns[4] = n4;
    ns[5] = n5;

    for (int k = 0; k < nCount; k++)
    {
        int gj = ns[k];
        if (gj < _Base || gj >= _Base + _ActiveCount)
            continue;

        float2 xij = _Pos[gj] - xi;

        float2 gradW = XPBI_GradWendlandC2(xij, h, EPS);
        if (dot(gradW, gradW) <= EPS * EPS)
            continue;

        float Vb = XPBI_ReadCurrentVolume(gj);
        if (Vb <= EPS)
            continue;

        float2 correctedGrad = XPBI_MulMat2Vec(Lm, gradW);
        float2 t = XPBI_MulMat2Vec(FT, correctedGrad);
        float2 q = Vb * XPBI_MulMat2Vec(dCdF, t);

        gradC_vi -= q;
        gradC_vj[k] = q;
    }

    float denom = _InvMass[gi] * dot(gradC_vi, gradC_vi);
    for (int k1 = 0; k1 < nCount; k1++)
    {
        int gj = ns[k1];
        if (gj < _Base || gj >= _Base + _ActiveCount)
            continue;
        if (XPBI_IsFixed(gj))
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

    _Vel[gi] = _Vel[gi] + (_InvMass[gi] * velScale * gradC_vi);

    for (int k2 = 0; k2 < nCount; k2++)
    {
        int gj = ns[k2];
        if (gj < _Base || gj >= _Base + _ActiveCount)
            continue;
        if (XPBI_IsFixed(gj))
            continue;
        _Vel[gj] = _Vel[gj] + (_InvMass[gj] * velScale * gradC_vj[k2]);
    }

    _Lambda[gi] = lambdaBefore + dLambda;
}

#endif
