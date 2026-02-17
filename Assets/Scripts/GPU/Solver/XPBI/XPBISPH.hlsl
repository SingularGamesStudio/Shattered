#ifndef XPBI_SPH_INCLUDED
#define XPBI_SPH_INCLUDED

static float XPBI_Alpha2D(float h)
{
    return 7.0 / (4.0 * 3.14159265358979323846 * h * h);
}

static float2 XPBI_GradWendlandC2(float2 xij, float h, float eps)
{
    float r2 = dot(xij, xij);
    if (h <= eps || r2 <= eps * eps)
        return float2(0, 0);

    float r = sqrt(r2);
    float q = r / h;
    if (q >= 2.0)
        return float2(0, 0);

    float alpha = XPBI_Alpha2D(h);

    float s = 1.0 - 0.5 * q;
    float s2 = s * s;
    float s3 = s2 * s;
    float s4 = s2 * s2;

    float dFdq = 4.0 * s3 * (-0.5) * (2.0 * q + 1.0) + s4 * 2.0;
    float dWdr = alpha * dFdq / h;

    return -(dWdr / r) * xij;
}

#endif
