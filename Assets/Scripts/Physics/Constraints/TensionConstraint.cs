using System.Collections.Generic;
using UnityEngine;
using Unity.Mathematics;

namespace Physics {
    // XPBD distance-like constraint with tension bias using previous-frame average strain.
    // Requires per-node caches: neighbors + neighborDistances prepared by another constraint.
    public class TensionConstraint : Constraint {
        private readonly float beta = 5.0f;

        const float eps = 1e-6f;

        // Prepare per-edge lambdas for this step using current neighbor caches
        public void Initialise(NodeBatch nodes) {
            nodes.CacheAvgRestLength();
        }

        public void Relax(NodeBatch data, float stiffness, float timeStep) {
            float aHat = stiffness / (timeStep * timeStep);

            for (int i = 0; i < data.Count; ++i) {
                var node = data.nodes[i];
                var cache = data.caches[i];

                // Compute current average length and gradient parts
                float avgLen = 0f;
                int samples = 0;
                float2 grad_i = float2.zero;

                for (int n = 0; n < cache.neighbors.Count; ++n) {
                    int j = cache.neighbors[n];
                    float2 d = node.predPos - data.nodes[j].predPos;
                    float dist = math.length(d);
                    if (dist < eps) { continue; }

                    float2 nij = d / dist;
                    avgLen += dist;
                    grad_i += nij;
                    samples++;
                }
                avgLen /= samples;


                float rHist = math.max(data.nodes[i].contraction, 1e-6f);
                float rStar = math.pow(rHist, -beta); // multiplicative target from history
                if (cache.avgEdgeLen < eps) continue;
                float C = (avgLen / cache.avgEdgeLen) - rStar;

                // Gradients scaled by 1/(N * avg0)
                grad_i /= cache.avgEdgeLen;

                float sumGradSq = node.invMass * math.lengthsq(grad_i);

                // accumulate neighbor contributions to denominator and apply later
                List<(int j, float2 grad_j, float wj)> neigh = new List<(int, float2, float)>(samples);
                for (int n = 0; n < cache.neighbors.Count; ++n) {
                    int j = cache.neighbors[n];
                    float2 d = node.predPos - data.nodes[j].predPos;
                    float dist = math.length(d);
                    if (dist < eps) { continue; }

                    float2 nij = d / dist;
                    float2 grad_j = -nij / cache.avgEdgeLen;
                    float wj = data.nodes[j].invMass;
                    sumGradSq += wj * math.lengthsq(grad_j);
                    neigh.Add((j, grad_j, wj));
                }

                float denom = sumGradSq + aHat;

                float dLambda = -(C + aHat * cache.lambdas.tension) / denom;
                cache.lambdas.tension += dLambda;

                if (!node.isFixed) node.predPos -= (-node.invMass * dLambda) * grad_i;
                foreach (var t in neigh) {
                    if (!data.nodes[t.j].isFixed) data.nodes[t.j].predPos -= (-t.wj * dLambda) * t.grad_j;
                }
            }
        }
    }
}