#ifndef XPBI_DEFORMATION_INCLUDED
    #define XPBI_DEFORMATION_INCLUDED

    #include "Utils.hlsl"

    static void PolarDecompose2D(
    Mat2 F,
    out Mat2 R,
    out Mat2 U,
    out float s1,
    out float s2,
    float stretchEps,
    float offDiagEps,
    float invDetEps)
    {
        Mat2 FT = TransposeMat2(F);
        Mat2 C = MulMat2(FT, F);

        float tr = C.c0.x + C.c1.y;
        float det = DetMat2(C);
        float disc = sqrt(max(tr * tr - 4.0 * det, 0.0));

        float l1 = 0.5 * (tr + disc);
        float l2 = 0.5 * (tr - disc);

        s1 = sqrt(max(l1, 0.0));
        s2 = sqrt(max(l2, 0.0));
        if (s1 < stretchEps)
        s1 = 1.0;
        if (s2 < stretchEps)
        s2 = 1.0;

        Mat2 V = EigenBasisSymmetric2x2(C, l1, l2, offDiagEps);

        Mat2 Sdiag;
        Sdiag.c0 = float2(s1, 0);
        Sdiag.c1 = float2(0, s2);

        U = MulMat2(MulMat2(V, Sdiag), TransposeMat2(V));

        float detU = DetMat2(U);
        if (abs(detU) < invDetEps)
        {
            R = Mat2Identity();
            U = Mat2Identity();
            s1 = 1.0;
            s2 = 1.0;
            return;
        }
        float invS1 = 1.0 / max(s1, stretchEps);
        float invS2 = 1.0 / max(s2, stretchEps);

        Mat2 Sinv;
        Sinv.c0 = float2(invS1, 0);
        Sinv.c1 = float2(0, invS2);

        Mat2 Uinv = MulMat2(MulMat2(V, Sinv), TransposeMat2(V));

        R = MulMat2(F, Uinv);
    }

    static float ComputePsiHencky(
    Mat2 F,
    float mu,
    float lambda,
    float stretchEps,
    float offDiagEps,
    float invDetEps)
    {
        Mat2 R, U;
        float s1, s2;
        PolarDecompose2D(F, R, U, s1, s2, stretchEps, offDiagEps, invDetEps);

        float h1 = log(max(s1, stretchEps));
        float h2 = log(max(s2, stretchEps));

        float k = h1 + h2;
        float mean = 0.5 * k;
        float h1Dev = h1 - mean;
        float h2Dev = h2 - mean;

        return mu * (h1Dev * h1Dev + h2Dev * h2Dev) + 0.5 * lambda * (k * k);
    }

    static float XPBI_ConstraintC(
    Mat2 F,
    float mu,
    float lambda,
    float stretchEps,
    float offDiagEps,
    float invDetEps)
    {
        return sqrt(2.0 * ComputePsiHencky(F, mu, lambda, stretchEps, offDiagEps, invDetEps));
    }

    static Mat2 ApplyPlasticityReturn(
    Mat2 F_elastic,
    float yieldHencky,
    float volHenckyLimit,
    float stretchEps,
    float offDiagEps,
    float invDetEps)
    {
        if (DetMat2(F_elastic) <= 0.0)
        return Mat2Identity();

        Mat2 R, S;
        float s1, s2;
        PolarDecompose2D(F_elastic, R, S, s1, s2, stretchEps, offDiagEps, invDetEps);

        if (!all(isfinite(R.c0)) || !all(isfinite(R.c1)) ||
        !all(isfinite(S.c0)) || !all(isfinite(S.c1)) ||
        !isfinite(s1) || !isfinite(s2))
        return Mat2Identity();

        float h1 = log(max(s1, stretchEps));
        float h2 = log(max(s2, stretchEps));

        float k = h1 + h2;
        float mean = 0.5 * k;
        float h1Dev = h1 - mean;
        float h2Dev = h2 - mean;

        float gammaEq = sqrt(2.0 * (h1Dev * h1Dev + h2Dev * h2Dev));
        bool devYield = gammaEq > yieldHencky;
        bool volYield = abs(k) > volHenckyLimit;

        if (!devYield && !volYield)
        return F_elastic;

        float devScale = 1.0;
        if (devYield)
        devScale = min(yieldHencky / max(gammaEq, stretchEps), 1.0);
        float kProj = k;
        if (volYield)
        kProj = clamp(k, -volHenckyLimit, volHenckyLimit);

        float e1 = clamp((h1Dev * devScale) + 0.5 * kProj, -3.0, 3.0);
        float e2 = clamp((h2Dev * devScale) + 0.5 * kProj, -3.0, 3.0);
        float s1Proj = exp(e1);
        float s2Proj = exp(e2);

        Mat2 V = EigenBasisSymmetric2x2(S, s1, s2, offDiagEps);

        Mat2 SprojDiag;
        SprojDiag.c0 = float2(s1Proj, 0);
        SprojDiag.c1 = float2(0, s2Proj);

        Mat2 Sproj = MulMat2(MulMat2(V, SprojDiag), TransposeMat2(V));
        Mat2 Fel = MulMat2(R, Sproj);
        if (!all(isfinite(Fel.c0)) || !all(isfinite(Fel.c1)))
        return Mat2Identity();
        return Fel;
    }

    static Mat2 XPBI_ComputeGradient(
    Mat2 F,
    float mu,
    float lambda,
    float stretchEps,
    float offDiagEps,
    float invDetEps)
    {
        Mat2 R, S;
        float s1, s2;
        PolarDecompose2D(F, R, S, s1, s2, stretchEps, offDiagEps, invDetEps);

        float h1 = log(max(s1, stretchEps));
        float h2 = log(max(s2, stretchEps));

        float k = h1 + h2;
        float mean = 0.5 * k;
        float h1Dev = h1 - mean;
        float h2Dev = h2 - mean;

        float psi = mu * (h1Dev * h1Dev + h2Dev * h2Dev) + 0.5 * lambda * (k * k);
        float C = sqrt(2.0 * psi);
        if (C < stretchEps)
        return Mat2Zero();

        float dPsi_dh1 = mu * (h1Dev - h2Dev) + lambda * k;
        float dPsi_dh2 = mu * (h2Dev - h1Dev) + lambda * k;

        float dPsi_ds1 = dPsi_dh1 / max(s1, stretchEps);
        float dPsi_ds2 = dPsi_dh2 / max(s2, stretchEps);

        Mat2 V = EigenBasisSymmetric2x2(S, s1, s2, offDiagEps);

        Mat2 D;
        D.c0 = float2(dPsi_ds1, 0);
        D.c1 = float2(0, dPsi_ds2);

        Mat2 dPsi_dS = MulMat2(MulMat2(V, D), TransposeMat2(V));
        Mat2 dPsi_dF = MulMat2(R, dPsi_dS);

        float invC = 1.0 / C;

        Mat2 dC_dF;
        dC_dF.c0 = dPsi_dF.c0 * invC;
        dC_dF.c1 = dPsi_dF.c1 * invC;
        return dC_dF;
    }

    [forceinline]
    static float Alpha2D(float h)
    {
        return 7.0 / (4.0 * 3.14159265358979323846 * h * h);
    }

    [forceinline]
    static float2 GradWendlandC2(float2 xij, float h, float eps)
    {
        float hSafe = max(h, 1e-4);

        float r2 = dot(xij, xij);
        if (hSafe <= eps || r2 <= eps * eps)
        return float2(0, 0);

        float r = sqrt(r2);

        // Avoid the 1/r singularity injecting huge finite values when two vertices get extremely close.
        // Close pairs are effectively treated as coincident for gradient purposes.
        if (r < 0.05 * hSafe)
        return float2(0, 0);

        float q = r / hSafe;
        if (q >= 2.0)
        return float2(0, 0);

        float alpha = Alpha2D(hSafe);

        float s = 1.0 - 0.5 * q;
        float s2 = s * s;
        float s3 = s2 * s;
        float s4 = s2 * s2;

        float dFdq = 4.0 * s3 * (-0.5) * (2.0 * q + 1.0) + s4 * 2.0;
        float dWdr = alpha * dFdq / hSafe;

        return -(dWdr / r) * xij;
    }


#endif // XPBI_DEFORMATION_INCLUDED