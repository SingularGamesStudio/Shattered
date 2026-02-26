#ifndef XPBI_UTILS_INCLUDED
    #define XPBI_UTILS_INCLUDED

    struct Mat2
    {
        float2 c0;
        float2 c1;
    };

    [forceinline]
    static Mat2 Mat2Identity()
    {
        Mat2 m;
        m.c0 = float2(1, 0);
        m.c1 = float2(0, 1);
        return m;
    }

    [forceinline]
    static Mat2 Mat2Zero()
    {
        Mat2 m;
        m.c0 = float2(0, 0);
        m.c1 = float2(0, 0);
        return m;
    }

    [forceinline]
    static Mat2 Mat2FromCols(float2 c0, float2 c1)
    {
        Mat2 m;
        m.c0 = c0;
        m.c1 = c1;
        return m;
    }

    [forceinline]
    static Mat2 Mat2FromFloat4(float4 v)
    {
        Mat2 m;
        m.c0 = v.xy;
        m.c1 = v.zw;
        return m;
    }

    [forceinline]
    static float4 Float4FromMat2(Mat2 m)
    {
        return float4(m.c0, m.c1);
    }

    [forceinline]
    static float2 MulMat2Vec(Mat2 A, float2 v)
    {
        return A.c0 * v.x + A.c1 * v.y;
    }

    [forceinline]
    static Mat2 MulMat2(Mat2 A, Mat2 B)
    {
        Mat2 r;
        r.c0 = MulMat2Vec(A, B.c0);
        r.c1 = MulMat2Vec(A, B.c1);
        return r;
    }

    [forceinline]
    static Mat2 TransposeMat2(Mat2 A)
    {
        Mat2 r;
        r.c0 = float2(A.c0.x, A.c1.x);
        r.c1 = float2(A.c0.y, A.c1.y);
        return r;
    }

    [forceinline]
    static float Dot2(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }

    [forceinline]
    static float DetMat2(Mat2 A)
    {
        return A.c0.x * A.c1.y - A.c1.x * A.c0.y;
    }

    static Mat2 EigenBasisSymmetric2x2(Mat2 M, float e1, float e2, float offDiagEps)
    {
        float b = M.c0.y;
        if (abs(b) <= offDiagEps)
        return Mat2Identity();

        float2 r1 = float2(b, e1 - M.c0.x);
        float2 r2 = float2(b, e2 - M.c0.x);

        float n1 = dot(r1, r1);
        float n2 = dot(r2, r2);

        if (!isfinite(n1) || !isfinite(n2) || n1 <= offDiagEps * offDiagEps || n2 <= offDiagEps * offDiagEps)
        return Mat2Identity();

        float2 v1 = normalize(r1);
        float2 v2 = normalize(r2);

        if (!all(isfinite(v1)) || !all(isfinite(v2)))
        return Mat2Identity();

        if (abs(dot(v1, v2)) > 0.999f)
        v2 = float2(-v1.y, v1.x);

        return Mat2FromCols(v1, v2);
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

        Mat2 VSinv;
        VSinv.c0 = V.c0 * invS1;
        VSinv.c1 = V.c1 * invS2;

        return MulMat2(VSinv, TransposeMat2(U));
    }

    static void AtomicAddFloatBits(RWStructuredBuffer<uint> buf, uint idx, float add)
    {
        uint expected;
        uint original;

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


#endif // XPBI_UTILS_INCLUDED