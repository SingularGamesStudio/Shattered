// Assets/Scripts/Physics/Constraints/TensionConstraint.cs
using System.Collections.Generic;
using Unity.Mathematics;

namespace Physics {
    // Per-node tension constraint:
    // - Uses only predicted positions for solving to avoid mid-iteration physics commits.
    // - Never updates Node.contraction inside Relax; instead records pending strain and
    //   commits it once in the next Initialise call (once per physics step).
    // - Reuses Node.constraintCache.neighbors and neighborDistances (already initialized elsewhere).
    public class TensionConstraint : Constraint {
        // Tuning parameters
        public float TensionGain = 0.5f;           // scales how strongly contraction biases target length
        public float MaxTensionBias = 0.2f;        // clamps |bias| around 1 (e.g., ±20%)
        public float ContractionSmoothing = 0.5f;  // [0..1], how quickly contraction memory adapts per step

        // Runs many times per step:
        // - Applies mass-weighted PBD distance corrections toward a tension-biased target.
        // - Records pending average signed strain from the final iteration’s predicted positions,
        //   but does NOT update Node.contraction here.
        public float Relax(List<Node> nodes, float stiffness, float lagrangeMult, float timeStep) {
            for (int i = 0; i < nodes.Count; i++) {
                var ni = nodes[i];
                var cache = ni.constraintCache;
                if (cache == null || cache.neighbors == null || cache.neighborDistances == null) continue;

                // Compute bias from the node’s stored contraction memory
                float bias = math.clamp(1f + TensionGain * ni.contraction,
                                        1f - MaxTensionBias, 1f + MaxTensionBias);

                float strainAccum = 0f;
                int strainCount = 0;

                // Directed edges i -> j exactly as cached; no symmetry assumed
                int count = cache.neighbors.Count;
                for (int k = 0; k < count; k++) {
                    int j = cache.neighbors[k];
                    if (j < 0 || j >= nodes.Count || j == i) continue;

                    var nj = nodes[j];

                    // Always operate on predicted positions
                    float2 xi = ni.predPos;
                    float2 xj = nj.predPos;

                    float2 r = xi - xj;
                    float dist = math.length(r);

                    float rest = cache.neighborDistances[k]; // original per-step distance
                    float effectiveRest = rest * bias;

                    // Classic PBD correction along the edge
                    float constraint = dist - effectiveRest;
                    float wA = ni.isFixed ? 0f : ni.invMass;
                    float wB = nj.isFixed ? 0f : nj.invMass;
                    float wSum = wA + wB;
                    if (wSum <= 0f) continue;

                    float2 n = r / dist;
                    float2 corr = constraint * n;

                    if (!ni.isFixed) ni.predPos -= (wA / wSum) * corr;
                    if (!nj.isFixed) nj.predPos += (wB / wSum) * corr;

                    // Record signed strain relative to the original rest (not the biased target)
                    // Positive if contracted (shorter than rest), negative if expanded
                    float signedStrain = (rest - dist) / rest;
                    strainAccum += math.clamp(signedStrain, -1f, 1f);
                    strainCount++;

                    // Optional convergence tracker
                    lagrangeMult += math.abs(constraint);
                }
            }
            return lagrangeMult;
        }
    }
}