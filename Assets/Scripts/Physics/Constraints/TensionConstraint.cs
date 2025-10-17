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
        public void Initialise(List<Node> nodes) {
            for (int i = 0; i < nodes.Count; i++) {
                var ni = nodes[i];
                ni.constraintCache.tensionLambda = 0f;
                ni.constraintCache.avgEdgeLen = 0f;
                int count = 0;
                foreach (float f in ni.constraintCache.neighborDistances) {
                    if (f < eps) { continue; }
                    ni.constraintCache.avgEdgeLen += f;
                    count++;

                }
                ni.constraintCache.avgEdgeLen /= count;
            }
        }

        public void Relax(List<Node> nodes, float stiffness, float timeStep) {
            float aHat = stiffness / (timeStep * timeStep);

            int count = nodes.Count;
            for (int i = 0; i < count; ++i) {
                var A = nodes[i];
                var cache = A.constraintCache;
                if (cache == null || cache.neighbors == null || cache.neighbors.Count == 0) continue;

                // Compute current average length and gradient parts
                float avgLen = 0f;
                int samples = 0;
                float2 grad_i = float2.zero;

                for (int n = 0; n < cache.neighbors.Count; ++n) {
                    int j = cache.neighbors[n];
                    float2 d = A.predPos - nodes[j].predPos;
                    float dist = math.length(d);
                    if (dist < eps) { continue; }

                    float2 nij = d / dist;
                    avgLen += dist;
                    grad_i += nij;
                    samples++;
                }
                avgLen /= samples;


                float rHist = math.max(nodes[i].contraction, 1e-6f);
                float rStar = math.pow(rHist, -beta); // multiplicative target from history
                if (cache.avgEdgeLen < eps) continue;
                float C = (avgLen / cache.avgEdgeLen) - rStar;
                // Gradients scaled by 1/(N * avg0)
                grad_i /= cache.avgEdgeLen;

                float sumGradSq = A.invMass * math.lengthsq(grad_i);

                // accumulate neighbor contributions to denominator and apply later
                List<(int j, float2 grad_j, float wj)> neigh = new List<(int, float2, float)>(samples);
                for (int n = 0; n < cache.neighbors.Count; ++n) {
                    int j = cache.neighbors[n];
                    float2 d = A.predPos - nodes[j].predPos;
                    float dist = math.length(d);
                    if (dist < eps) { continue; }

                    float2 nij = d / dist;
                    float2 grad_j = -nij / cache.avgEdgeLen;
                    float wj = nodes[j].invMass;
                    sumGradSq += wj * math.lengthsq(grad_j);
                    neigh.Add((j, grad_j, wj));
                }

                float denom = sumGradSq + aHat;

                float dLambda = -(C + aHat * cache.tensionLambda) / denom;
                cache.tensionLambda += dLambda;

                if (!A.isFixed) A.predPos -= (-A.invMass * dLambda) * grad_i;
                foreach (var t in neigh) {
                    if (!nodes[t.j].isFixed) nodes[t.j].predPos -= (-t.wj * dLambda) * t.grad_j;
                }
            }
        }
    }
}