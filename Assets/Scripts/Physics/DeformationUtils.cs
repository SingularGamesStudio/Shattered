using Unity.Mathematics;
using UnityEngine;

namespace Physics {
    public static class DeformationUtils {
        // Polar decomposition: F = R * S where R is rotation, S is symmetric stretch
        public static void PolarDecompose2D(float2x2 F, out float2x2 R, out float2x2 S, out float s1, out float s2, int nodeIdx = -1) {
            // C = F^T * F (right Cauchy-Green tensor)
            float2x2 C = math.mul(math.transpose(F), F);

            // Eigenvalues of C
            float tr = C.c0.x + C.c1.y;
            float det = C.c0.x * C.c1.y - C.c0.y * C.c1.x;
            float disc = math.sqrt(math.max(tr * tr - 4f * det, 0f));
            float l1 = 0.5f * (tr + disc);
            float l2 = 0.5f * (tr - disc);

            // Principal stretches
            s1 = math.sqrt(math.max(l1, 0f));
            s2 = math.sqrt(math.max(l2, 0f));
            if (s1 < 1e-6f) s1 = 1f;
            if (s2 < 1e-6f) s2 = 1f;

            // Eigenvectors of C
            float2x2 V;
            if (math.abs(C.c0.y) > 1e-5f) {
                V = new float2x2(
                    math.normalize(new float2(l1 - C.c1.y, C.c0.y)),
                    math.normalize(new float2(l2 - C.c1.y, C.c0.y))
                );
            } else {
                V = float2x2.identity;
            }

            // Reconstruct S = V * D * V^T
            S = math.mul(
                math.mul(V, new float2x2(new float2(s1, 0f), new float2(0f, s2))),
                math.transpose(V)
            );

            // R = F * S^-1
            float detS = S.c0.x * S.c1.y - S.c0.y * S.c1.x;
            if (math.abs(detS) < 1e-8f || AnyNaNOrInf(S)) {
                Debug.LogWarning($"PolarDecompose2D: Singular/NaN S for node {nodeIdx}; setting R=I.");
                R = float2x2.identity;
            } else {
                R = math.mul(F, math.inverse(S));
            }

            // Final NaN/Inf check
            if (AnyNaNOrInf(R) || AnyNaNOrInf(S)) {
                Debug.LogWarning($"PolarDecompose2D: R or S contain NaN/Inf for node {nodeIdx}. Returning identities.");
                R = float2x2.identity;
                S = float2x2.identity;
                s1 = 1f;
                s2 = 1f;
            }
        }

        public static void SVD2x2(float2x2 A, out float2x2 U, out float2 sigma, out float2x2 V) {
            float2x2 ATA = math.mul(math.transpose(A), A);

            float trace = ATA.c0.x + ATA.c1.y;
            float det = ATA.c0.x * ATA.c1.y - ATA.c0.y * ATA.c1.x;
            float discriminant = math.max(trace * trace - 4f * det, 0f);

            // Eigenvalues of A^T*A
            float lambda1 = 0.5f * (trace + math.sqrt(discriminant));
            float lambda2 = 0.5f * (trace - math.sqrt(discriminant));

            sigma = new float2(math.sqrt(math.max(lambda1, 0f)), math.sqrt(math.max(lambda2, 0f)));

            // Right singular vectors (eigenvectors of A^T*A)
            V = float2x2.identity;
            if (math.abs(ATA.c0.y) > 1e-6f) {
                V = new float2x2(
                    math.normalize(new float2(lambda1 - ATA.c1.y, ATA.c0.y)),
                    math.normalize(new float2(lambda2 - ATA.c1.y, ATA.c0.y))
                );
            }

            // Left singular vectors: U = A*V / sigma
            U = float2x2.identity;
            if (sigma.x > 1e-6f && sigma.y > 1e-6f) {
                U = new float2x2(
                    math.mul(A, V.c0) / sigma.x,
                    math.mul(A, V.c1) / sigma.y
                );
            }
        }

        public static float2x2 PseudoInverse(float2x2 A) {
            SVD2x2(A, out float2x2 U, out float2 sigma, out float2x2 V);

            // Invert non-zero singular values
            float2 sigmaInv = new float2(
                sigma.x > 1e-6f ? 1f / sigma.x : 0f,
                sigma.y > 1e-6f ? 1f / sigma.y : 0f
            );

            // A^+ = V * Σ^+ * U^T
            return math.mul(
                math.mul(V, new float2x2(new float2(sigmaInv.x, 0f), new float2(0f, sigmaInv.y))),
                math.transpose(U)
            );
        }

        static bool AnyNaNOrInf(float2x2 m) =>
            float.IsNaN(m.c0.x) || float.IsNaN(m.c0.y) || float.IsNaN(m.c1.x) || float.IsNaN(m.c1.y) ||
            float.IsInfinity(m.c0.x) || float.IsInfinity(m.c0.y) || float.IsInfinity(m.c1.x) || float.IsInfinity(m.c1.y);
    }
}
