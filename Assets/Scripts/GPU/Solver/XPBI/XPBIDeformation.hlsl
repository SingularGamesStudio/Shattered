#ifndef XPBI_DEFORMATION_INCLUDED
#define XPBI_DEFORMATION_INCLUDED

#include "XPBICommon.hlsl"

static XPBI_Mat2 XPBI_EigenBasisSymmetric2x2(XPBI_Mat2 M, float e1, float e2, float offDiagEps) {
    float b = M.c0.y;
    if (abs(b) <= offDiagEps) return XPBI_Mat2Identity();

    float2 v1 = normalize(float2(b, e1 - M.c0.x));
    float2 v2 = normalize(float2(b, e2 - M.c0.x));

    XPBI_Mat2 V;
    V.c0 = v1;
    V.c1 = v2;
    return V;
}

static void XPBI_PolarDecompose2D(
    XPBI_Mat2 F,
    out XPBI_Mat2 R,
    out XPBI_Mat2 U,
    out float s1,
    out float s2,
    float stretchEps,
    float offDiagEps,
    float invDetEps
) {
    XPBI_Mat2 FT = XPBI_TransposeMat2(F);
    XPBI_Mat2 C = XPBI_MulMat2(FT, F);

    float tr = C.c0.x + C.c1.y;
    float det = XPBI_DetMat2(C);
    float disc = sqrt(max(tr * tr - 4.0 * det, 0.0));

    float l1 = 0.5 * (tr + disc);
    float l2 = 0.5 * (tr - disc);

    s1 = sqrt(max(l1, 0.0));
    s2 = sqrt(max(l2, 0.0));
    if (s1 < stretchEps) s1 = 1.0;
    if (s2 < stretchEps) s2 = 1.0;

    XPBI_Mat2 V = XPBI_EigenBasisSymmetric2x2(C, l1, l2, offDiagEps);

    XPBI_Mat2 Sdiag;
    Sdiag.c0 = float2(s1, 0);
    Sdiag.c1 = float2(0, s2);

    U = XPBI_MulMat2(XPBI_MulMat2(V, Sdiag), XPBI_TransposeMat2(V));

    float detU = XPBI_DetMat2(U);
    if (abs(detU) < invDetEps) {
        R = XPBI_Mat2Identity();
        U = XPBI_Mat2Identity();
        s1 = 1.0;
        s2 = 1.0;
        return;
    }
    float invS1 = 1.0 / max(s1, stretchEps);
    float invS2 = 1.0 / max(s2, stretchEps);

    XPBI_Mat2 Sinv;
    Sinv.c0 = float2(invS1, 0);
    Sinv.c1 = float2(0, invS2);

    XPBI_Mat2 Uinv = XPBI_MulMat2(XPBI_MulMat2(V, Sinv), XPBI_TransposeMat2(V));

    R = XPBI_MulMat2(F, Uinv);
}


static float XPBI_ComputePsiHencky(
    XPBI_Mat2 F,
    float mu,
    float lambda,
    float stretchEps,
    float offDiagEps,
    float invDetEps
) {
    XPBI_Mat2 R, U;
    float s1, s2;
    XPBI_PolarDecompose2D(F, R, U, s1, s2, stretchEps, offDiagEps, invDetEps);

    float h1 = log(max(s1, stretchEps));
    float h2 = log(max(s2, stretchEps));

    float k = h1 + h2;
    float mean = 0.5 * k;
    float h1Dev = h1 - mean;
    float h2Dev = h2 - mean;

    return mu * (h1Dev * h1Dev + h2Dev * h2Dev) + 0.5 * lambda * (k * k);
}

static float XPBI_ConstraintC(
    XPBI_Mat2 F,
    float mu,
    float lambda,
    float stretchEps,
    float offDiagEps,
    float invDetEps
) {
    return sqrt(2.0 * XPBI_ComputePsiHencky(F, mu, lambda, stretchEps, offDiagEps, invDetEps));
}

static XPBI_Mat2 XPBI_ApplyPlasticityReturn(
    XPBI_Mat2 F_elastic,
    float yieldHencky,
    float volHenckyLimit,
    float stretchEps,
    float offDiagEps,
    float invDetEps
) {
    XPBI_Mat2 R, S;
    float s1, s2;
    XPBI_PolarDecompose2D(F_elastic, R, S, s1, s2, stretchEps, offDiagEps, invDetEps);

    float h1 = log(max(s1, stretchEps));
    float h2 = log(max(s2, stretchEps));

    float k = h1 + h2;
    float mean = 0.5 * k;
    float h1Dev = h1 - mean;
    float h2Dev = h2 - mean;

    float gammaEq = sqrt(2.0 * (h1Dev * h1Dev + h2Dev * h2Dev));
    bool devYield = gammaEq > yieldHencky;
    bool volYield = abs(k) > volHenckyLimit;

    if (!devYield && !volYield) return F_elastic;

    float devScale = 1.0;
    if (devYield) devScale =  min(yieldHencky / max(gammaEq, stretchEps), 1.0);
    float kProj = k;
    if(volYield) kProj = clamp(k, -volHenckyLimit, volHenckyLimit);

    float s1Proj = exp((h1Dev * devScale) + 0.5 * kProj);
    float s2Proj = exp((h2Dev * devScale) + 0.5 * kProj);

    XPBI_Mat2 V = XPBI_EigenBasisSymmetric2x2(S, s1, s2, offDiagEps);

    XPBI_Mat2 SprojDiag;
    SprojDiag.c0 = float2(s1Proj, 0);
    SprojDiag.c1 = float2(0, s2Proj);

    XPBI_Mat2 Sproj = XPBI_MulMat2(XPBI_MulMat2(V, SprojDiag), XPBI_TransposeMat2(V));
    return XPBI_MulMat2(R, Sproj);
}

static XPBI_Mat2 XPBI_ComputeGradient(
    XPBI_Mat2 F,
    float mu,
    float lambda,
    float stretchEps,
    float offDiagEps,
    float invDetEps
) {
    XPBI_Mat2 R, S;
    float s1, s2;
    XPBI_PolarDecompose2D(F, R, S, s1, s2, stretchEps, offDiagEps, invDetEps);

    float h1 = log(max(s1, stretchEps));
    float h2 = log(max(s2, stretchEps));

    float k = h1 + h2;
    float mean = 0.5 * k;
    float h1Dev = h1 - mean;
    float h2Dev = h2 - mean;

    float psi = mu * (h1Dev * h1Dev + h2Dev * h2Dev) + 0.5 * lambda * (k * k);
    float C = sqrt(2.0 * psi);
    if (C < stretchEps) return XPBI_Mat2Zero();

    float dPsi_dh1 = mu * (h1Dev - h2Dev) + lambda * k;
    float dPsi_dh2 = mu * (h2Dev - h1Dev) + lambda * k;

    float dPsi_ds1 = dPsi_dh1 / max(s1, stretchEps);
    float dPsi_ds2 = dPsi_dh2 / max(s2, stretchEps);

    XPBI_Mat2 V = XPBI_EigenBasisSymmetric2x2(S, s1, s2, offDiagEps);

    XPBI_Mat2 D;
    D.c0 = float2(dPsi_ds1, 0);
    D.c1 = float2(0, dPsi_ds2);

    XPBI_Mat2 dPsi_dS = XPBI_MulMat2(XPBI_MulMat2(V, D), XPBI_TransposeMat2(V));
    XPBI_Mat2 dPsi_dF = XPBI_MulMat2(R, dPsi_dS);

    float invC = 1.0 / C;

    XPBI_Mat2 dC_dF;
    dC_dF.c0 = dPsi_dF.c0 * invC;
    dC_dF.c1 = dPsi_dF.c1 * invC;
    return dC_dF;
}


#endif
