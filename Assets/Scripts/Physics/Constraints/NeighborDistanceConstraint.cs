
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

                    if (wi + wj <= 0f) {
                        continue;
                    }

                    float2 oldPosI = node.predPos;
                    float2 r = node.predPos - data.nodes[j].predPos;
                    float len = math.length(r);

                    if (len < Const.Eps) {
                        debug.degenerateCount++;
                        continue;
                    }

                    float dLambda = -(len - cache.neighborDistances[k] + alphaTilde * cache.lambdas.neighborDistance[k]) /
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
    }
}