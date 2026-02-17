#ifndef XPBI_UTILS_INCLUDED
#define XPBI_UTILS_INCLUDED

struct XPBI_Mat2
{
    float2 c0;
    float2 c1;
};

static XPBI_Mat2 XPBI_Mat2Identity()
{
    XPBI_Mat2 m;
    m.c0 = float2(1, 0);
    m.c1 = float2(0, 1);
    return m;
}

static XPBI_Mat2 XPBI_Mat2Zero()
{
    XPBI_Mat2 m;
    m.c0 = float2(0, 0);
    m.c1 = float2(0, 0);
    return m;
}

static XPBI_Mat2 XPBI_Mat2FromCols(float2 c0, float2 c1)
{
    XPBI_Mat2 m;
    m.c0 = c0;
    m.c1 = c1;
    return m;
}

static XPBI_Mat2 XPBI_Mat2FromFloat4(float4 v)
{
    XPBI_Mat2 m;
    m.c0 = v.xy;
    m.c1 = v.zw;
    return m;
}

static float4 XPBI_Float4FromMat2(XPBI_Mat2 m)
{
    return float4(m.c0, m.c1);
}

static float2 XPBI_MulMat2Vec(XPBI_Mat2 A, float2 v)
{
    return A.c0 * v.x + A.c1 * v.y;
}

static XPBI_Mat2 XPBI_MulMat2(XPBI_Mat2 A, XPBI_Mat2 B)
{
    XPBI_Mat2 r;
    r.c0 = XPBI_MulMat2Vec(A, B.c0);
    r.c1 = XPBI_MulMat2Vec(A, B.c1);
    return r;
}

static XPBI_Mat2 XPBI_TransposeMat2(XPBI_Mat2 A)
{
    XPBI_Mat2 r;
    r.c0 = float2(A.c0.x, A.c1.x);
    r.c1 = float2(A.c0.y, A.c1.y);
    return r;
}

static float XPBI_Dot2(float2 a, float2 b) { return a.x * b.x + a.y * b.y; }

static float XPBI_DetMat2(XPBI_Mat2 A)
{
    return A.c0.x * A.c1.y - A.c1.x * A.c0.y;
}

static XPBI_Mat2 XPBI_EigenBasisSymmetric2x2_Common(XPBI_Mat2 M, float e1, float e2, float offDiagEps)
{
    float b = M.c0.y;
    if (abs(b) <= offDiagEps)
        return XPBI_Mat2Identity();

    float2 v1 = normalize(float2(b, e1 - M.c0.x));
    float2 v2 = normalize(float2(b, e2 - M.c0.x));
    return XPBI_Mat2FromCols(v1, v2);
}

static XPBI_Mat2 XPBI_PseudoInverseMat2(XPBI_Mat2 A, float stretchEps, float eigenOffDiagEps)
{
    float a00 = XPBI_Dot2(A.c0, A.c0);
    float a01 = XPBI_Dot2(A.c0, A.c1);
    float a11 = XPBI_Dot2(A.c1, A.c1);

    XPBI_Mat2 ATA = XPBI_Mat2FromCols(float2(a00, a01), float2(a01, a11));

    float tr = a00 + a11;
    float det = a00 * a11 - a01 * a01;
    float disc = sqrt(max(tr * tr - 4.0 * det, 0.0));

    float l1 = 0.5 * (tr + disc);
    float l2 = 0.5 * (tr - disc);

    float s1 = sqrt(max(l1, 0.0));
    float s2 = sqrt(max(l2, 0.0));

    XPBI_Mat2 V = XPBI_EigenBasisSymmetric2x2_Common(ATA, l1, l2, eigenOffDiagEps);

    XPBI_Mat2 U = XPBI_Mat2Identity();
    if (s1 > stretchEps && s2 > stretchEps)
    {
        U.c0 = XPBI_MulMat2Vec(A, V.c0) / s1;
        U.c1 = XPBI_MulMat2Vec(A, V.c1) / s2;
    }

    float invS1 = (s1 > stretchEps) ? (1.0 / s1) : 0.0;
    float invS2 = (s2 > stretchEps) ? (1.0 / s2) : 0.0;

    XPBI_Mat2 VSinv;
    VSinv.c0 = V.c0 * invS1;
    VSinv.c1 = V.c1 * invS2;

    return XPBI_MulMat2(VSinv, XPBI_TransposeMat2(U));
}

#endif
