// Assets/Scripts/Physics/Constraints/NeighborDistanceConstraint.cs
using System.Collections.Generic;
using System.Diagnostics;
using Unity.Mathematics;

namespace Physics {
    // Directed XPBD neighbor-distance constraint without global edge caching.
    public class NeighborDistanceConstraint : Constraint {
        public float Compliance = 0f;              // α (0 => rigid)
        public float Damping = 0f;                 // γ ≥ 0 (optional)

        private const float Eps = 1e-6f;
        const int neighborCount = 6;

        public void Initialise(List<Node> nodes) {
            // Capture per-node neighbor lists and their rest distances; reset per-link lambdas.
            for (int i = 0; i < nodes.Count; i++) {
                var ni = nodes[i];
                ni.constraintCache ??= new ConstraintCache();
                ni.constraintCache.neighbors ??= new List<int>();
                ni.constraintCache.neighborDistances ??= new List<float>();
                ni.constraintCache.neighborLambdas ??= new List<float>();

                ni.constraintCache.neighbors = ni.parent.hnsw.SearchKnn(ni.predPos, neighborCount);
                ni.constraintCache.neighborDistances.Clear();
                ni.constraintCache.neighborLambdas.Clear();

                if (ni.HNSWNeighbors == null) continue;

                // Flatten all HNSW layers; do not deduplicate or assume symmetry.
                foreach (int j in ni.constraintCache.neighbors) {
                    ni.constraintCache.neighborDistances.Add(math.distance(ni.pos, nodes[j].pos));
                    ni.constraintCache.neighborLambdas.Add(0f);
                }
            }
        }

        public void Relax(List<Node> nodes, float stiffness, float timeStep) {
            float alphaTilde = stiffness / math.max(1e-6f, timeStep * timeStep);
            float gammaDt = Damping * timeStep;

            for (int i = 0; i < nodes.Count; i++) {
                var ni = nodes[i];
                var cache = ni.constraintCache;
                if (cache == null || cache.neighbors == null) continue;

                float wi = ni.isFixed ? 0f : ni.invMass;

                for (int k = 0; k < cache.neighbors.Count; k++) {
                    int j = cache.neighbors[k];
                    if (j < 0 || j >= nodes.Count || j == i) continue;

                    var nj = nodes[j];
                    float wj = nj.isFixed ? 0f : nj.invMass;
                    float wsum = wi + wj;
                    if (wsum <= 0f) continue;

                    float2 xi = ni.predPos;
                    float2 xj = nj.predPos;

                    float2 r = xi - xj;

                    float len = math.length(r);
                    if (len < Eps) continue;
                    float2 n = r / len;
                    float C = math.length(r) - cache.neighborDistances[k];

                    float denom = wsum + alphaTilde + gammaDt;
                    float lam = cache.neighborLambdas[k];
                    float dLambda = -(C + alphaTilde * lam) / math.max(denom, 1e-8f);

                    float2 corrI = -wi * dLambda * n;
                    float2 corrJ = +wj * dLambda * n;

                    ni.predPos += corrI;
                    nj.predPos += corrJ;

                    cache.neighborLambdas[k] = lam + dLambda;
                }
            }
        }
    }
}
