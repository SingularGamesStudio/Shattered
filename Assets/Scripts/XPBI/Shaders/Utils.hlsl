#ifndef XPBI_UTILS_INCLUDED
    #define XPBI_UTILS_INCLUDED

    struct Mat2
    {
        float2 c0;
        float2 c1;
    };

    static Mat2 Mat2Identity()
    {
        Mat2 m = (Mat2)0;
        m.c0 = float2(1, 0);
        m.c1 = float2(0, 1);
        return m;
    }

    static Mat2 Mat2Zero()
    {
        Mat2 m = (Mat2)0;
        m.c0 = float2(0, 0);
        m.c1 = float2(0, 0);
        return m;
    }

    static Mat2 Mat2FromCols(float2 c0, float2 c1)
    {
        Mat2 m = (Mat2)0;
        m.c0 = c0;
        m.c1 = c1;
        return m;
    }

    static Mat2 Mat2FromFloat4(float4 v)
    {
        Mat2 m = (Mat2)0;
        m.c0 = v.xy;
        m.c1 = v.zw;
        return m;
    }

    static float4 Float4FromMat2(Mat2 m)
    {
        return float4(m.c0, m.c1);
    }

    static float2 MulMat2Vec(Mat2 A, float2 v)
    {
        return A.c0 * v.x + A.c1 * v.y;
    }

    static Mat2 MulMat2(Mat2 A, Mat2 B)
    {
        Mat2 r = (Mat2)0;
        r.c0 = MulMat2Vec(A, B.c0);
        r.c1 = MulMat2Vec(A, B.c1);
        return r;
    }

    static Mat2 TransposeMat2(Mat2 A)
    {
        Mat2 r = (Mat2)0;
        r.c0 = float2(A.c0.x, A.c1.x);
        r.c1 = float2(A.c0.y, A.c1.y);
        return r;
    }

    static float Dot2(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }

    static float DetMat2(Mat2 A)
    {
        return A.c0.x * A.c1.y - A.c1.x * A.c0.y;
    }

    static Mat2 EigenBasisSymmetric2x2(Mat2 M, float e1, float e2, float offDiagEps)
    {
        Mat2 result = (Mat2)0;
        result = Mat2Identity();
        float b = M.c0.y;
        if (abs(b) > offDiagEps)
        {
            float2 r1 = float2(b, e1 - M.c0.x);
            float2 r2 = float2(b, e2 - M.c0.x);

            float n1 = dot(r1, r1);
            float n2 = dot(r2, r2);

            if (isfinite(n1) && isfinite(n2) && n1 > offDiagEps * offDiagEps && n2 > offDiagEps * offDiagEps)
            {
                float2 v1 = normalize(r1);
                float2 v2 = normalize(r2);

                if (all(isfinite(v1)) && all(isfinite(v2)))
                {
                    if (abs(dot(v1, v2)) > 0.999f)
                    v2 = float2(-v1.y, v1.x);

                    result = Mat2FromCols(v1, v2);
                }
            }
        }

        return result;
    }

    static Mat2 PseudoInverseMat2(Mat2 A, float stretchEps, float eigenOffDiagEps)
    {
        float a00 = Dot2(A.c0, A.c0);
        float a01 = Dot2(A.c0, A.c1);
        float a11 = Dot2(A.c1, A.c1);

        Mat2 ATA = Mat2FromCols(float2(a00, a01), float2(a01, a11));

        float tr = a00 + a11;
        float det = a00 * a11 - a01 * a01;
        float disc = sqrt(max(tr * tr - 4.0 * det, 0.0));

        float l1 = 0.5 * (tr + disc);
        float l2 = 0.5 * (tr - disc);

        float s1 = sqrt(max(l1, 0.0));
        float s2 = sqrt(max(l2, 0.0));

        Mat2 V = EigenBasisSymmetric2x2(ATA, l1, l2, eigenOffDiagEps);

        Mat2 U = Mat2Identity();
        if (s1 > stretchEps && s2 > stretchEps)
        {
            U.c0 = MulMat2Vec(A, V.c0) / s1;
            U.c1 = MulMat2Vec(A, V.c1) / s2;
        }

        float invS1 = (s1 > stretchEps) ? (1.0 / s1) : 0.0;
        float invS2 = (s2 > stretchEps) ? (1.0 / s2) : 0.0;

        Mat2 VSinv = (Mat2)0;
        VSinv.c0 = V.c0 * invS1;
        VSinv.c1 = V.c1 * invS2;

        return MulMat2(VSinv, TransposeMat2(U));
    }

    static void AtomicAddFloatBits(RWStructuredBuffer<uint> buf, uint idx, float add)
    {
        uint expected = 0u;
        uint original = 0u;

        [loop] for (uint it = 0; it < 64; it++)
        {
            expected = buf[idx];
            float cur = asfloat(expected);
            uint desired = asuint(cur + add);

            InterlockedCompareExchange(buf[idx], expected, desired, original);
            if (original == expected)
            return;
        }
    }

    static void AtomicAddFloat2(RWStructuredBuffer<uint> bits, uint gi, float2 dv)
    {
        uint baseIdx = gi * 2u;
        AtomicAddFloatBits(bits, baseIdx + 0u, dv.x);
        AtomicAddFloatBits(bits, baseIdx + 1u, dv.y);
    }

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
        R = Mat2Identity();
        U = Mat2Identity();
        s1 = 1.0;
        s2 = 1.0;

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

        Mat2 Sdiag = (Mat2)0;
        Sdiag.c0 = float2(s1, 0);
        Sdiag.c1 = float2(0, s2);

        U = MulMat2(MulMat2(V, Sdiag), TransposeMat2(V));

        float detU = DetMat2(U);
        if (abs(detU) < invDetEps)
        {
            return;
        }
        float invS1 = 1.0 / max(s1, stretchEps);
        float invS2 = 1.0 / max(s2, stretchEps);

        Mat2 Sinv = (Mat2)0;
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
        Mat2 R = (Mat2)0;
        Mat2 U = (Mat2)0;
        float s1 = 1.0;
        float s2 = 1.0;
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
        Mat2 result = (Mat2)0;
        result = Mat2Identity();

        if (DetMat2(F_elastic) > 0.0)
        {
            Mat2 R = (Mat2)0;
            Mat2 S = (Mat2)0;
            float s1 = 1.0;
            float s2 = 1.0;
            PolarDecompose2D(F_elastic, R, S, s1, s2, stretchEps, offDiagEps, invDetEps);

            bool decompValid = all(isfinite(R.c0)) && all(isfinite(R.c1)) &&
            all(isfinite(S.c0)) && all(isfinite(S.c1)) &&
            isfinite(s1) && isfinite(s2);

            if (decompValid)
            {
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
                {
                    result = F_elastic;
                }
                else
                {
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

                    Mat2 V = (Mat2)0;
                    V = EigenBasisSymmetric2x2(S, s1, s2, offDiagEps);

                    Mat2 SprojDiag = (Mat2)0;
                    SprojDiag.c0 = float2(s1Proj, 0);
                    SprojDiag.c1 = float2(0, s2Proj);

                    Mat2 Sproj = MulMat2(MulMat2(V, SprojDiag), TransposeMat2(V));
                    Mat2 Fel = MulMat2(R, Sproj);
                    if (all(isfinite(Fel.c0)) && all(isfinite(Fel.c1)))
                    result = Fel;
                }
            }
        }

        return result;
    }

    static Mat2 XPBI_ComputeGradient(
    Mat2 F,
    float mu,
    float lambda,
    float stretchEps,
    float offDiagEps,
    float invDetEps)
    {
        Mat2 result = (Mat2)0;
        result = Mat2Zero();

        Mat2 R = (Mat2)0;
        Mat2 S = (Mat2)0;
        float s1 = 1.0;
        float s2 = 1.0;
        PolarDecompose2D(F, R, S, s1, s2, stretchEps, offDiagEps, invDetEps);

        float h1 = log(max(s1, stretchEps));
        float h2 = log(max(s2, stretchEps));

        float k = h1 + h2;
        float mean = 0.5 * k;
        float h1Dev = h1 - mean;
        float h2Dev = h2 - mean;

        float psi = mu * (h1Dev * h1Dev + h2Dev * h2Dev) + 0.5 * lambda * (k * k);
        float C = sqrt(2.0 * psi);
        if (C >= stretchEps)
        {
            float dPsi_dh1 = mu * (h1Dev - h2Dev) + lambda * k;
            float dPsi_dh2 = mu * (h2Dev - h1Dev) + lambda * k;

            float dPsi_ds1 = dPsi_dh1 / max(s1, stretchEps);
            float dPsi_ds2 = dPsi_dh2 / max(s2, stretchEps);

            Mat2 V = (Mat2)0;
            V = EigenBasisSymmetric2x2(S, s1, s2, offDiagEps);

            Mat2 D = (Mat2)0;
            D.c0 = float2(dPsi_ds1, 0);
            D.c1 = float2(0, dPsi_ds2);

            Mat2 dPsi_dS = MulMat2(MulMat2(V, D), TransposeMat2(V));
            Mat2 dPsi_dF = MulMat2(R, dPsi_dS);

            float invC = 1.0 / C;

            Mat2 dC_dF = (Mat2)0;
            dC_dF.c0 = dPsi_dF.c0 * invC;
            dC_dF.c1 = dPsi_dF.c1 * invC;
            result = dC_dF;
        }

        return result;
    }

    static float Alpha2D(float h)
    {
        return 7.0 / (4.0 * 3.14159265358979323846 * h * h);
    }

    static float2 GradWendlandC2(float2 xij, float h, float eps)
    {
        float2 result = 0.0;
        float hSafe = max(h, 1e-4);

        float r2 = dot(xij, xij);
        if (hSafe > eps && r2 > eps * eps)
        {
            float r = sqrt(r2);

            if (r >= 0.05 * hSafe)
            {
                float q = r / hSafe;
                if (q < 2.0)
                {
                    float alpha = Alpha2D(hSafe);

                    float s = 1.0 - 0.5 * q;
                    float s2 = s * s;
                    float s3 = s2 * s;
                    float s4 = s2 * s2;

                    float dFdq = 4.0 * s3 * (-0.5) * (2.0 * q + 1.0) + s4 * 2.0;
                    float dWdr = alpha * dFdq / hSafe;

                    result = -(dWdr / r) * xij;
                }
            }
        }

        return result;
    }

#endif // XPBI_UTILS_INCLUDED