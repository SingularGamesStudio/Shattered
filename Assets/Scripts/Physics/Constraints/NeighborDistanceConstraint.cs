using Unity.Mathematics;
using UnityEngine;

namespace Physics {
    public class NeighborDistanceConstraint : Constraint {
        public float Compliance = 0f;
        public float Damping = 0f;
        public string GetConstraintType() => "NeighborDistanceConstraint";

        public void Initialise(NodeBatch nodes) {
            nodes.CacheNeighbors();
            nodes.ResetDebugData(GetConstraintType());
        }

        public void Relax(NodeBatch data, float stiffness, float timeStep) {
            float alphaTilde = stiffness / math.max(1e-6f, timeStep * timeStep);
            float gammaDt = Damping * timeStep;
            string constraintType = GetConstraintType();

            for (int i = 0; i < data.Count; i++) {
                var node = data.nodes[i];
                var cache = data.caches[i];
                var debug = data.GetOrCreateDebugData(i, constraintType);
                if (cache?.neighbors == null) continue;

                float wi = node.invMass;

                for (int k = 0; k < cache.neighbors.Count; k++) {
                    int j = cache.neighbors[k];
                    if (j < 0 || j >= data.Count || j == i) continue;

                    float wj = data.nodes[j].invMass;
                    if (wi + wj <= 0f) continue;

                    float2 oldPosI = node.predPos;
                    float2 r = node.predPos - data.nodes[j].predPos;
                    float len = math.length(r);

                    if (len < Const.Eps) {
                        debug.degenerateCount++;
                        continue;
                    }

                    // On-the-fly rest length
                    float2 xi0 = node.originalPos;
                    float2 xj0 = data.nodes[j].originalPos;
                    float2 restEdge = math.mul(node.Fp, xj0 - xi0);
                    float restLen = math.length(restEdge);

                    float dLambda = -(len - restLen + alphaTilde * cache.lambdas.neighborDistance[k]) /
                                    math.max(wi + wj + alphaTilde + gammaDt, 1e-8f);

                    if (float.IsNaN(dLambda) || float.IsInfinity(dLambda)) {
                        debug.nanInfCount++;
                        continue;
                    }

                    float2 correction = (-dLambda / len) * r;
                    node.predPos += wi * correction;
                    data.nodes[j].predPos -= wj * correction;

                    cache.lambdas.neighborDistance[k] += dLambda;
                    debug.RecordPositionUpdate(node.predPos - oldPosI);
                }
            }

            data.FinalizeDebugData(constraintType);
        }

        public void PlasticFlow(NodeBatch data, float dt) {
            const float yieldStretch = 1.05f; // threshold, adjust for your material

            for (int i = 0; i < data.Count; ++i) {
                var node = data.nodes[i];
                var cache = data.caches[i];
                if (cache?.neighbors == null) continue;

                int n = cache.neighbors.Count;
                if (n == 0) continue;
                float2[] restEdges = new float2[n];
                float2[] curEdges = new float2[n];
                var xi0 = node.originalPos;
                var xi = node.predPos;
                for (int k = 0; k < n; ++k) {
                    int j = cache.neighbors[k];
                    var xj0 = data.nodes[j].originalPos;
                    var xj = data.nodes[j].predPos;
                    restEdges[k] = math.mul(node.Fp, xj0 - xi0);
                    curEdges[k] = xj - xi;
                }

                // Fit F, polar decompose
                float2x2 F = DeformationUtils.FitDeformationGradient(restEdges, curEdges);
                float2x2 R, S;
                float s1, s2;
                DeformationUtils.PolarDecompose2D(F, out R, out S, out s1, out s2);

                bool yielded = (math.max(s1, s2) > yieldStretch) || (math.min(s1, s2) < 1f / yieldStretch);
                if (yielded) {
                    // Return-mapping: clamp principal stretches
                    float clamped_s1 = math.clamp(s1, 1f / yieldStretch, yieldStretch);
                    float clamped_s2 = math.clamp(s2, 1f / yieldStretch, yieldStretch);

                    // Rebuild projected S ("Sp") with same eigenvectors as S
                    // V: eigenvector matrix from last polar, Dcl: new diag (clamped principal stretches)
                    float2x2 V;
                    float a = S.c0.x, b = S.c0.y, c = S.c1.x, d = S.c1.y;
                    if (math.abs(b) > 1e-5f) {
                        float2 v1 = math.normalize(new float2(s1 * s1 - d * d, b));
                        float2 v2 = math.normalize(new float2(s2 * s2 - d * d, b));
                        V = new float2x2(v1, v2);
                    } else {
                        V = float2x2.identity;
                    }
                    float2x2 Dcl = new float2x2(new float2(clamped_s1, 0f), new float2(0f, clamped_s2));
                    float2x2 Sp = math.mul(math.mul(V, Dcl), math.transpose(V));

                    // Update plastic gradient: Fp := Sp * Fp
                    node.Fp = math.mul(Sp, node.Fp);
                }
            }
        }
    }
}