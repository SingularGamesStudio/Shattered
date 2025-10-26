using System.Collections.Generic;
using Unity.Mathematics;

namespace Physics {
    public class TensionConstraint : Constraint {
        private readonly float beta = 1.0f;

        public string GetConstraintType() => "TensionConstraint";

        public void Initialise(NodeBatch nodes) {
            nodes.CacheAvgRestLength();
            nodes.ResetDebugData(GetConstraintType());
        }

        public void Relax(NodeBatch data, float stiffness, float timeStep) {
            float aHat = stiffness / (timeStep * timeStep);
            string constraintType = GetConstraintType();

            for (int i = 0; i < data.Count; ++i) {
                var node = data.nodes[i];
                var cache = data.caches[i];
                var debug = data.GetOrCreateDebugData(i, constraintType);

                float2 oldPos = node.predPos;
                float2 grad_i = float2.zero;
                float avgLen = 0f;
                int samples = 0;

                for (int n = 0; n < cache.neighbors.Count; ++n) {
                    float2 d = node.predPos - data.nodes[cache.neighbors[n]].predPos;
                    float dist = math.length(d);
                    if (dist < Const.Eps) {
                        debug.degenerateCount++;
                        continue;
                    }
                    avgLen += dist;
                    grad_i += d / dist;
                    samples++;
                }

                if (samples == 0) {
                    debug.skippedIterations++;
                    continue;
                }

                if (cache.avgEdgeLen < Const.Eps) {
                    debug.degenerateCount++;
                    continue;
                }

                float C = (avgLen / (samples * cache.avgEdgeLen)) - math.pow(math.max(node.contraction, 1e-6f), -beta);
                grad_i /= cache.avgEdgeLen;
                float sumGradSq = node.invMass * math.lengthsq(grad_i);

                List<(int j, float2 grad_j, float wj)> neigh = new List<(int, float2, float)>(samples);
                for (int n = 0; n < cache.neighbors.Count; ++n) {
                    float2 d = node.predPos - data.nodes[cache.neighbors[n]].predPos;
                    float dist = math.length(d);
                    if (dist < Const.Eps) continue;

                    float wj = data.nodes[cache.neighbors[n]].invMass;
                    float2 grad_j = -(d / dist) / cache.avgEdgeLen;
                    sumGradSq += wj * math.lengthsq(grad_j);
                    neigh.Add((cache.neighbors[n], grad_j, wj));
                }

                float denom = sumGradSq + aHat;
                if (denom < Const.Eps) {
                    debug.skippedIterations++;
                    continue;
                }

                float dLambda = -(C + aHat * cache.lambdas.tension) / denom;

                if (float.IsNaN(dLambda) || float.IsInfinity(dLambda)) {
                    debug.nanInfCount++;
                    continue;
                }

                cache.lambdas.tension += dLambda;

                if (!node.isFixed) node.predPos -= (-node.invMass * dLambda) * grad_i;
                foreach (var t in neigh) {
                    if (!data.nodes[t.j].isFixed) data.nodes[t.j].predPos -= (-t.wj * dLambda) * t.grad_j;
                }

                debug.RecordPositionUpdate(node.predPos - oldPos);
            }

            data.FinalizeDebugData(constraintType);
        }
    }
}