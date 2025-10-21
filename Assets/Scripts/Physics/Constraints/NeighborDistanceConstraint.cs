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


        public void Initialise(NodeBatch nodes) {
            nodes.CacheNeighbors();
        }

        public void Relax(NodeBatch data, float stiffness, float timeStep) {
            float alphaTilde = stiffness / math.max(1e-6f, timeStep * timeStep);
            float gammaDt = Damping * timeStep;

            for (int i = 0; i < data.Count; i++) {
                var node = data.nodes[i];
                var cache = data.caches[i];
                if (cache == null || cache.neighbors == null) continue;

                float wi = node.isFixed ? 0f : node.invMass;

                for (int k = 0; k < cache.neighbors.Count; k++) {
                    int j = cache.neighbors[k];
                    if (j < 0 || j >= data.Count || j == i) continue;

                    var neigh = data.nodes[j];
                    float wj = neigh.isFixed ? 0f : neigh.invMass;
                    float wsum = wi + wj;
                    if (wsum <= 0f) continue;

                    float2 xi = node.predPos;
                    float2 xj = neigh.predPos;

                    float2 r = xi - xj;

                    float len = math.length(r);
                    if (len < Eps) continue;
                    float2 n = r / len;
                    float C = math.length(r) - cache.neighborDistances[k];

                    float denom = wsum + alphaTilde + gammaDt;
                    float dLambda = -(C + alphaTilde * cache.lambdas.neighborDistance[k]) / math.max(denom, 1e-8f);

                    float2 corrI = -wi * dLambda * n;
                    float2 corrJ = +wj * dLambda * n;

                    node.predPos += corrI;
                    neigh.predPos += corrJ;

                    cache.lambdas.neighborDistance[k] += dLambda;
                }
            }
        }
    }
}
