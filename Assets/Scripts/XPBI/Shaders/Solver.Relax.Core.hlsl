#ifndef XPBI_DEBUG_ITER
    #define XPBI_DEBUG_ITER _ConvergenceDebugIter
#endif

// Expected locals in scope:
//   uint li, gi, active
//   float kernelH, support, supportSq, invDt, invDt2
//   uint dtLi, baseIdx, nCount
//   bool useOwnerFilter
//   int ownerI
//
// Expected backend macros:
//   XPBI_GET_GJ(gjLi)
//   XPBI_POS(li, gi)
//   XPBI_VEL(li, gi)
//   XPBI_SET_VEL(li, gi, v)
//   XPBI_LAMBDA(li, gi)
//   XPBI_SET_LAMBDA(li, gi, l)
//   XPBI_L_FROM_I(li, gi)
//   XPBI_F0_FROM_I(li, gi)
//   XPBI_NEIGHBOR_FIXED(gjLi, gj)
//   XPBI_INV_MASS(gjLi, gj)
//   XPBI_ACTIVE_I(li, gi)
//   XPBI_APPLY_MODE_JR (0/1)
//   XPBI_SCATTER_DV(gi, dv)      (JR only)
//   XPBI_SCATTER_DL(gi, dl)      (JR only)

if (!XPBI_ACTIVE_I(li, gi)) return;

float2 xi = 0.0;
xi = XPBI_POS(li, gi);
float2 vi = 0.0;
vi = XPBI_VEL(li, gi);

Mat2 Lm = (Mat2)0;
Lm = Mat2FromFloat4(XPBI_L_FROM_I(li, gi));

Mat2 gradV = (Mat2)0;
gradV = Mat2Zero();
static const uint MAX_N = 16u;
uint nb_gjLi[MAX_N];
float2 nb_corrGrad[MAX_N];
float nb_Vb[MAX_N];
float nb_invMassJ[MAX_N];
float2 nb_q[MAX_N];
uint k = 0u;

[loop] for (uint nIdx0 = 0u; nIdx0 < MAX_N; nIdx0++)
{
    if (nIdx0 >= nCount) break;

    uint gjLi = _DtNeighbors[baseIdx + nIdx0];
    if (gjLi == ~0u || gjLi >= active) continue;
    if (useOwnerFilter && _DtOwnerByLocal[gjLi] != ownerI) continue;

    uint gj = ~0u;
    gj = XPBI_GET_GJ(gjLi);
    if (gj == ~0u) continue;

    float2 xij = XPBI_POS(gjLi, gj) - xi;

    float2 gradW = 0.0;
    gradW = GradWendlandC2(xij, kernelH, EPS);
    if (dot(gradW, gradW) <= EPS * EPS) continue;

    float Vb = 0.0;
    Vb = ReadCurrentVolume(gj);
    if (Vb <= EPS) continue;

    float2 correctedGrad = MulMat2Vec(Lm, gradW);
    float2 dv = XPBI_VEL(gjLi, gj) - vi;

    gradV.c0 += dv * (Vb * correctedGrad.x);
    gradV.c1 += dv * (Vb * correctedGrad.y);

    nb_gjLi[k] = gjLi;
    nb_corrGrad[k] = correctedGrad;
    nb_Vb[k] = Vb;
    nb_invMassJ[k] = XPBI_NEIGHBOR_FIXED(gjLi, gj) ? 0.0 : XPBI_INV_MASS(gjLi, gj);
    k++;
}

if (k < 3u) return;

Mat2 F0 = (Mat2)0;
F0 = Mat2FromFloat4(XPBI_F0_FROM_I(li, gi));
Mat2 I = (Mat2)0;
I = Mat2Identity();
Mat2 dF = Mat2FromCols(I.c0 + gradV.c0 * _Dt, I.c1 + gradV.c1 * _Dt);
if (DetMat2(dF) <= 0.0) return;

Mat2 Ftrial = MulMat2(dF, F0);

float yieldHencky = 0.0;
yieldHencky = ReadMaterialYieldHencky(gi);
float volHenckyLimit = 0.0;
volHenckyLimit = ReadMaterialVolHenckyLimit(gi);
Mat2 Fel = (Mat2)0;
Fel = ApplyPlasticityReturn(Ftrial, yieldHencky, volHenckyLimit,
STRETCH_EPS, EIGEN_OFFDIAG_EPS, INV_DET_EPS);
if (!all(isfinite(Fel.c0)) || !all(isfinite(Fel.c1))) return;

float mu = 0.0, lambda = 0.0;
ComputeMaterialLame(gi, mu, lambda);

float C = 0.0;
C = XPBI_ConstraintC(Fel, mu, lambda, STRETCH_EPS, EIGEN_OFFDIAG_EPS, INV_DET_EPS);
if (!isfinite(C)) return;

if (_ConvergenceDebugEnable != 0)
{
    uint baseIter = _ConvergenceDebugOffset + XPBI_DEBUG_ITER;
    uint baseU = baseIter * CONV_DEBUG_UINTS_PER_ITER;

    uint uAbsC = (uint)min(abs(C) * _ConvergenceDebugScaleC, 4294967295.0);
    InterlockedAdd(_ConvergenceDebug[baseU + 0], uAbsC);
    InterlockedMax(_ConvergenceDebug[baseU + 1], uAbsC);
    InterlockedAdd(_ConvergenceDebug[baseU + 4], 1u);
}

if (abs(C) < EPS) return;
if (abs(C) > 5.0) return;

Mat2 dCdF = (Mat2)0;
dCdF = XPBI_ComputeGradient(Fel, mu, lambda, STRETCH_EPS, EIGEN_OFFDIAG_EPS, INV_DET_EPS);
Mat2 FT = (Mat2)0;
FT = TransposeMat2(Fel);
Mat2 Md = (Mat2)0;
Md = MulMat2(dCdF, FT);

float alphaTilde = 0.0;
alphaTilde = (_Compliance / EffectiveVolumeForCompliance(gi)) * invDt2;

float2 gradC_vi = 0.0;
float denomNeighbors = 0.0;
float maxInvMassLocal = 0.0;
maxInvMassLocal = XPBI_INV_MASS(li, gi);
float maxGradNorm2Local = 0.0;
[loop] for (uint nIdx1 = 0u; nIdx1 < MAX_N; nIdx1++)
{
    if (nIdx1 >= k) break;

    float2 q = nb_Vb[nIdx1] * MulMat2Vec(Md, nb_corrGrad[nIdx1]);
    nb_q[nIdx1] = q;

    gradC_vi -= q;

    float invMassJ = nb_invMassJ[nIdx1];
    float q2 = dot(q, q);
    denomNeighbors += invMassJ * q2;
    maxInvMassLocal = max(maxInvMassLocal, invMassJ);
    maxGradNorm2Local = max(maxGradNorm2Local, q2);
}

float invMassI = 0.0;
invMassI = XPBI_INV_MASS(li, gi);
float gradNormI2 = dot(gradC_vi, gradC_vi);
if (!(gradNormI2 > 1e-8)) return;

float denom = invMassI * gradNormI2 + denomNeighbors;
if (denom < 1e-4) return;

float lambdaBefore = 0.0;
lambdaBefore = XPBI_LAMBDA(li, gi);
float dLambda = -(C + alphaTilde * lambdaBefore) / (denom + alphaTilde);
if (!isfinite(dLambda)) return;
if (abs(dLambda) > 100.0) return;

if (_ConvergenceDebugEnable != 0)
{
    uint baseIter = _ConvergenceDebugOffset + XPBI_DEBUG_ITER;
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
float maxSpeed2 = maxSpeedLocal * maxSpeedLocal;
float maxSpeedHalf2 = (0.5 * maxSpeedLocal) * (0.5 * maxSpeedLocal);
if (pred2 > maxDv2) return;
if (pred2 > maxSpeedHalf2) return;

#if XPBI_APPLY_MODE_JR
    float2 dVi = invMassI * velScale * gradC_vi;
    float dVi2 = dot(dVi, dVi);
    if (dVi2 > maxDv2) dVi *= maxDeltaVPerIter * rsqrt(max(dVi2, EPS * EPS));
    XPBI_SCATTER_DV(gi, dVi);
    XPBI_SCATTER_DL(gi, dLambda);

    [loop] for (uint nIdx2 = 0u; nIdx2 < MAX_N; nIdx2++)
    {
        if (nIdx2 >= k) break;

        float invMassJ = nb_invMassJ[nIdx2];
        if (invMassJ <= 0.0) continue;

        uint gjLi = nb_gjLi[nIdx2];
        uint gj = XPBI_GET_GJ(gjLi);
        if (gj == ~0u) continue;

        float2 q = nb_q[nIdx2];
        float2 dVj = invMassJ * velScale * q;

        float dVj2 = dot(dVj, dVj);
        if (dVj2 > maxDv2) dVj *= maxDeltaVPerIter * rsqrt(max(dVj2, EPS * EPS));
        XPBI_SCATTER_DV(gj, dVj);
    }
#else
    float2 dVi = invMassI * velScale * gradC_vi;
    float dVi2 = dot(dVi, dVi);
    if (dVi2 > maxDv2)
    {
        float invLen = rsqrt(max(dVi2, EPS * EPS));
        dVi *= maxDeltaVPerIter * invLen;
    }

    float2 vI = XPBI_VEL(li, gi) + dVi;
    float vI2 = dot(vI, vI);
    if (vI2 > maxSpeed2)
    {
        float invLen = rsqrt(max(vI2, EPS * EPS));
        vI *= maxSpeedLocal * invLen;
    }
    XPBI_SET_VEL(li, gi, vI);

    [loop] for (uint nIdx3 = 0u; nIdx3 < MAX_N; nIdx3++)
    {
        if (nIdx3 >= k) break;

        float invMassJ = nb_invMassJ[nIdx3];
        if (invMassJ <= 0.0) continue;

        uint gjLi = nb_gjLi[nIdx3];
        uint gj = XPBI_GET_GJ(gjLi);
        if (gj == ~0u) continue;

        float2 q = nb_q[nIdx3];
        float2 dVj = invMassJ * velScale * q;

        float dVj2 = dot(dVj, dVj);
        if (dVj2 > maxDv2)
        {
            float invLen = rsqrt(max(dVj2, EPS * EPS));
            dVj *= maxDeltaVPerIter * invLen;
        }

        float2 vJ = XPBI_VEL(gjLi, gj) + dVj;
        float vJ2 = dot(vJ, vJ);
        if (vJ2 > maxSpeed2)
        {
            float invLen = rsqrt(max(vJ2, EPS * EPS));
            vJ *= maxSpeedLocal * invLen;
        }
        XPBI_SET_VEL(gjLi, gj, vJ);
    }

    if (_CollisionEnable != 0u)
    {
        #include "Solver.Relax.CollisionCore.hlsl"
    }

    XPBI_SET_LAMBDA(li, gi, lambdaBefore + dLambda);
    float2 dampedVel = XPBI_VEL(li, gi);
    ApplySingleAnchorRadialDampingOnVel(gi, mu, lambda, XPBI_POS(li, gi), dampedVel);
    XPBI_SET_VEL(li, gi, dampedVel);
#endif
