using Unity.Mathematics;
using UnityEngine;

namespace Physics {

    public static class XPBIConfig {
        public const float YoungsModulus = 5e4f;
        public const float PoissonsRatio = 0.3f;

        // Lamé parameters from (E, ν).
        public static float Mu => YoungsModulus / (2f * (1f + PoissonsRatio));
        public static float Lambda => (YoungsModulus * PoissonsRatio) / ((1f + PoissonsRatio) * (1f - 2f * PoissonsRatio));

        // Hencky-space yield parameters (tunable).
        public const float YieldHencky = 0.05f;
        public const float VolumetricHenckyLimit = 0.3f;
    }

    public static class DeformationUtils {

        const float StretchEps = 1e-6f;
        const float EigenOffDiagEps = 1e-5f;
        const float InvDetEps = 1e-8f;

        public static float2x2 ComputeGradient(float2x2 F, int nodeIdx) {
            PolarDecompose2D(F, out float2x2 R, out float2x2 S, out float s1, out float s2, nodeIdx);

            // Spectral Hencky strain: h_i = log(s_i); dev via subtracting mean. (Quadratic Hencky energy model.) [web:63]
            float h1 = math.log(math.max(s1, StretchEps));
            float h2 = math.log(math.max(s2, StretchEps));
            float k = h1 + h2;
            float mean = 0.5f * k;
            float h1Dev = h1 - mean;
            float h2Dev = h2 - mean;

            // dΨ/dh_i for Ψ = μ||dev(log U)||^2 + (λ/2) tr(log U)^2 (2D specialization). [web:63]
            float dPsi_dh1 = XPBIConfig.Mu * (h1Dev - h2Dev) + XPBIConfig.Lambda * k;
            float dPsi_dh2 = XPBIConfig.Mu * (h2Dev - h1Dev) + XPBIConfig.Lambda * k;

            float dPsi_ds1 = dPsi_dh1 / math.max(s1, StretchEps);
            float dPsi_ds2 = dPsi_dh2 / math.max(s2, StretchEps);

            float2x2 V = EigenBasisSymmetric2x2(S, s1, s2);
            float2x2 dPsi_dS = math.mul(math.mul(V, new float2x2(new float2(dPsi_ds1, 0f), new float2(0f, dPsi_ds2))), math.transpose(V));
            float2x2 dPsi_dF = math.mul(R, dPsi_dS);

            float C = XPBIConstraint(F, nodeIdx);
            if (C < StretchEps) return float2x2.zero;

            float invC = 1f / C;
            return new float2x2(dPsi_dF.c0 * invC, dPsi_dF.c1 * invC);
        }

        public static float2x2 ApplyPlasticityReturn(float2x2 F_elastic, int nodeIdx, ref DebugData debug) {
            PolarDecompose2D(F_elastic, out float2x2 R, out float2x2 S, out float s1, out float s2, nodeIdx);

            float h1 = math.log(math.max(s1, StretchEps));
            float h2 = math.log(math.max(s2, StretchEps));

            float k = h1 + h2;
            float mean = 0.5f * k;
            float h1Dev = h1 - mean;
            float h2Dev = h2 - mean;

            float gammaEq = math.sqrt(2f * (h1Dev * h1Dev + h2Dev * h2Dev));
            bool devYield = gammaEq > XPBIConfig.YieldHencky;
            bool volYield = math.abs(k) > XPBIConfig.VolumetricHenckyLimit;

            if (!devYield && !volYield) return F_elastic;

            debug.plasticFlowCount++;

            // Return mapping in Hencky space:
            // - deviatoric: radial projection to yield surface
            // - volumetric: clamp (a cap, not true volumetric plastic flow)
            float devScale = devYield ? math.min(XPBIConfig.YieldHencky / math.max(gammaEq, StretchEps), 1f) : 1f;
            float kProj = volYield ? math.clamp(k, -XPBIConfig.VolumetricHenckyLimit, XPBIConfig.VolumetricHenckyLimit) : k;

            float s1Proj = math.exp((h1Dev * devScale) + 0.5f * kProj);
            float s2Proj = math.exp((h2Dev * devScale) + 0.5f * kProj);

            float2x2 V = EigenBasisSymmetric2x2(S, s1, s2);
            float2x2 Sproj = math.mul(math.mul(V, new float2x2(new float2(s1Proj, 0f), new float2(0f, s2Proj))), math.transpose(V));
            return math.mul(R, Sproj);
        }

        public static float ComputePsiHencky(float2x2 F, int nodeIdx) {
            PolarDecompose2D(F, out _, out _, out float s1, out float s2, nodeIdx);

            float h1 = math.log(math.max(s1, StretchEps));
            float h2 = math.log(math.max(s2, StretchEps));

            float k = h1 + h2;
            float mean = 0.5f * k;
            float h1Dev = h1 - mean;
            float h2Dev = h2 - mean;

            return XPBIConfig.Mu * (h1Dev * h1Dev + h2Dev * h2Dev) + 0.5f * XPBIConfig.Lambda * (k * k);
        }

        // XPBI constitutive constraint: C(F) = sqrt(2 Ψ(F)).
        public static float XPBIConstraint(float2x2 F, int nodeIdx) =>
            math.sqrt(2f * ComputePsiHencky(F, nodeIdx));

        // Polar decomposition in 2D using eigendecomposition of C = F^T F: U = sqrt(C), R = F U^{-1}. [web:64]
        public static void PolarDecompose2D(float2x2 F, out float2x2 R, out float2x2 U, out float s1, out float s2, int nodeIdx = -1) {
            float2x2 C = math.mul(math.transpose(F), F);

            float tr = C.c0.x + C.c1.y;
            float det = C.c0.x * C.c1.y - C.c0.y * C.c1.x;
            float disc = math.sqrt(math.max(tr * tr - 4f * det, 0f));

            float l1 = 0.5f * (tr + disc);
            float l2 = 0.5f * (tr - disc);

            s1 = math.sqrt(math.max(l1, 0f));
            s2 = math.sqrt(math.max(l2, 0f));
            if (s1 < StretchEps) s1 = 1f;
            if (s2 < StretchEps) s2 = 1f;

            float2x2 V = EigenBasisSymmetric2x2(C, l1, l2);
            U = math.mul(math.mul(V, new float2x2(new float2(s1, 0f), new float2(0f, s2))), math.transpose(V));

            float detU = U.c0.x * U.c1.y - U.c0.y * U.c1.x;
            if (math.abs(detU) < InvDetEps || AnyNaNOrInf(U)) {
                Debug.LogWarning($"PolarDecompose2D: Singular/NaN U for node {nodeIdx}; returning identities.");
                R = float2x2.identity;
                U = float2x2.identity;
                s1 = 1f;
                s2 = 1f;
                return;
            }

            R = math.mul(F, math.inverse(U));

            if (AnyNaNOrInf(R) || AnyNaNOrInf(U)) {
                Debug.LogWarning($"PolarDecompose2D: NaN/Inf for node {nodeIdx}; returning identities.");
                R = float2x2.identity;
                U = float2x2.identity;
                s1 = 1f;
                s2 = 1f;
            }
        }

        public static void SVD2x2(float2x2 A, out float2x2 U, out float2 sigma, out float2x2 V) {
            float2x2 ATA = math.mul(math.transpose(A), A);

            float tr = ATA.c0.x + ATA.c1.y;
            float det = ATA.c0.x * ATA.c1.y - ATA.c0.y * ATA.c1.x;
            float disc = math.sqrt(math.max(tr * tr - 4f * det, 0f));

            float l1 = 0.5f * (tr + disc);
            float l2 = 0.5f * (tr - disc);

            sigma = new float2(math.sqrt(math.max(l1, 0f)), math.sqrt(math.max(l2, 0f)));

            V = EigenBasisSymmetric2x2(ATA, l1, l2);

            U = float2x2.identity;
            if (sigma.x > StretchEps && sigma.y > StretchEps) {
                U = new float2x2(
                    math.mul(A, V.c0) / sigma.x,
                    math.mul(A, V.c1) / sigma.y
                );
            }
        }

        public static float2x2 PseudoInverse(float2x2 A) {
            SVD2x2(A, out float2x2 U, out float2 s, out float2x2 V);
            return math.mul(
                math.mul(V, new float2x2(new float2(s.x > StretchEps ? 1f / s.x : 0f, 0f),
                                        new float2(0f, s.y > StretchEps ? 1f / s.y : 0f))),
                math.transpose(U)
            );
        }

        static float2x2 EigenBasisSymmetric2x2(float2x2 M, float e1, float e2) {
            // Minimal branching eigenbasis for symmetric 2x2; reuse across C, S, ATA.
            float b = M.c0.y;
            if (math.abs(b) <= EigenOffDiagEps) return float2x2.identity;

            // For symmetric [[a, b],[b, c]], an eigenvector for eigenvalue e is (b, e - a).
            float2 v1 = math.normalize(new float2(b, e1 - M.c0.x));
            float2 v2 = math.normalize(new float2(b, e2 - M.c0.x));
            return new float2x2(v1, v2);
        }

        static bool AnyNaNOrInf(float2x2 m) =>
            float.IsNaN(m.c0.x) || float.IsNaN(m.c0.y) || float.IsNaN(m.c1.x) || float.IsNaN(m.c1.y) ||
            float.IsInfinity(m.c0.x) || float.IsInfinity(m.c0.y) || float.IsInfinity(m.c1.x) || float.IsInfinity(m.c1.y);
    }
}
