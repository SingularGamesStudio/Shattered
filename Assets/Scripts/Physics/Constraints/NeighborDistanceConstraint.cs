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

                    float restLen = cache.neighborDistances[k];

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
            const float yieldStrain = 0.03f; // Example threshold

            for (int i = 0; i < data.Count; ++i) {
                var node = data.nodes[i];
                var cache = data.caches[i];
                if (cache?.neighbors == null) continue;

                int n = cache.neighbors.Count;
                float2 plasticShift = float2.zero;
                int plasticSamples = 0;

                for (int k = 0; k < n; ++k) {
                    int j = cache.neighbors[k];
                    if (j < 0 || j >= data.Count || j == i) continue;

                    float2 xi0 = node.plasticReferencePos;
                    float2 xj0 = data.nodes[j].plasticReferencePos;
                    float restLen = math.distance(xi0, xj0);

                    float2 xi = node.predPos;
                    float2 xj = data.nodes[j].predPos;
                    float curLen = math.distance(xi, xj);

                    float strain = (curLen - restLen) / restLen;
                    if (math.abs(strain) > yieldStrain) {
                        // Shift plastic reference position toward current one—simple XPBI plastic update
                        plasticShift += (xi - xi0) * 0.3f; // Partial, can be tuned
                        plasticSamples++;
                    }
                }
                // For simple 2D model, average shift if plastic deformation is detected
                if (plasticSamples > 0) {
                    node.plasticReferencePos += plasticShift / plasticSamples;
                }
            }
        }
    }
}