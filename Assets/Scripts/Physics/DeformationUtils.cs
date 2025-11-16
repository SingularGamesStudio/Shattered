using Unity.Mathematics;
using UnityEngine;

namespace Physics {
    public static class DeformationUtils {
        // Returns deformation gradient F given restEdge and currentEdge lists
        public static float2x2 FitDeformationGradient(float2[] restEdges, float2[] currentEdges, int nodeIdx = -1) {
            float2x2 A = float2x2.zero, B = float2x2.zero;
            int N = restEdges.Length;
            for (int i = 0; i < N; i++) {
                var r = restEdges[i];
                var c = currentEdges[i];
                A.c0 += r.x * r;
                A.c1 += r.y * r;
                B.c0 += r.x * c;
                B.c1 += r.y * c;
            }
            float detA = A.c0.x * A.c1.y - A.c0.y * A.c1.x;
            if (math.abs(detA) < 1e-8f) {
                Debug.LogWarning($"FitDeformationGradient: Singular A matrix for node {nodeIdx}; returning identity.");
                return float2x2.identity;
            }
            var invA = math.inverse(A);
            var F = math.mul(B, invA);

            if (AnyNaNOrInf(F)) {
                Debug.LogWarning($"FitDeformationGradient: NaN/Inf detected in F for node {nodeIdx}; returning identity.");
                return float2x2.identity;
            }
            return F;
        }

        // 2D polar decomposition: returns R (rotation), S (stretch), s1/s2 (principal stretches)
        public static void PolarDecompose2D(float2x2 F, out float2x2 R, out float2x2 S, out float s1, out float s2, int nodeIdx = -1) {
            float2x2 C = math.mul(math.transpose(F), F);
            float a = C.c0.x, b = C.c0.y, c = C.c1.x, d = C.c1.y;
            float tr = a + d;
            float det = a * d - b * c;
            float disc = math.sqrt(math.max(tr * tr - 4f * det, 0f));
            float l1 = 0.5f * (tr + disc);
            float l2 = 0.5f * (tr - disc);

            s1 = math.sqrt(math.max(l1, 0f));
            s2 = math.sqrt(math.max(l2, 0f));
            if (s1 < 1e-6f) s1 = 1f;
            if (s2 < 1e-6f) s2 = 1f;

            // Compute eigenvectors V of symmetric C
            float2x2 V;
            if (math.abs(b) > 1e-5f) {
                float2 v1 = math.normalize(new float2(l1 - d, b));
                float2 v2 = math.normalize(new float2(l2 - d, b));
                V = new float2x2(v1, v2);
            } else {
                V = float2x2.identity;
            }

            float2x2 D = new float2x2(new float2(s1, 0f), new float2(0f, s2));
            S = math.mul(math.mul(V, D), math.transpose(V));
            float2x2 Sinv;
            float detS = S.c0.x * S.c1.y - S.c0.y * S.c1.x;
            if (math.abs(detS) < 1e-8f || AnyNaNOrInf(S)) {
                Debug.LogWarning($"PolarDecompose2D: Singular/NaN S for node {nodeIdx}; setting R=I.");
                Sinv = float2x2.identity;
            } else {
                Sinv = math.inverse(S);
            }
            R = math.mul(F, Sinv);

            if (AnyNaNOrInf(R) || AnyNaNOrInf(S)) {
                Debug.LogWarning($"PolarDecompose2D: R or S contain NaN/Inf for node {nodeIdx}. Returning identities.");
                R = float2x2.identity;
                S = float2x2.identity;
                s1 = 1f; s2 = 1f;
            }
        }

        static bool AnyNaNOrInf(float2x2 m) =>
            float.IsNaN(m.c0.x) || float.IsNaN(m.c0.y) || float.IsNaN(m.c1.x) || float.IsNaN(m.c1.y) ||
            float.IsInfinity(m.c0.x) || float.IsInfinity(m.c0.y) || float.IsInfinity(m.c1.x) || float.IsInfinity(m.c1.y);
    }
}
