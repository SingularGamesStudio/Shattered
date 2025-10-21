// Assets/Scripts/Physics/Constraints/VolumeConstraint2D.cs
using System;
using System.Collections.Generic;
using System.Diagnostics;
using Unity.Mathematics;

namespace Physics {
    public class VolumeConstraint : Constraint {
        public float Compliance = 0f;     // α (0 => rigid)
        public float Damping = 0f;        // γ ≥ 0 (optional)

        const float Eps = 1e-6f;

        public void Initialise(NodeBatch data) {
            data.CacheVolume();
        }

        public void Relax(NodeBatch data, float stiffness, float timeStep) {
            int n = data.Count;
            float alphaTilde = stiffness / (timeStep * timeStep); // match project pattern
            float gammaDt = Damping * timeStep;

            for (int i = 0; i < n; i++) {
                var ni = data.nodes[i];
                var cache = data.caches[i];
                float wi = ni.isFixed ? 0f : ni.invMass;

                float2 a = ni.predPos;
                for (int k = 0; k < cache.neighbors.Count; k++) {

                    var nb = data.nodes[cache.neighbors[k]];
                    var nc = data.nodes[cache.neighbors[(k + 1) % cache.neighbors.Count]];

                    float wj = nb.isFixed ? 0f : nb.invMass;
                    float wk = nc.isFixed ? 0f : nc.invMass;

                    float2 b = nb.predPos;
                    float2 c = nc.predPos;

                    // Signed triangle area A = 0.5 * cross(b - a, c - a)
                    float A = 0.5f * Cross(b - a, c - a);
                    float C = A - cache.leafVolumes[k];
                    UnityEngine.Debug.Log(C);


                    // Gradients for triangle area:
                    // ∇_a A = 0.5 * perp(b - c), ∇_b A = 0.5 * perp(c - a), ∇_c A = 0.5 * perp(a - b)
                    float2 gA = 0.5f * Perp(b - c);
                    float2 gB = 0.5f * Perp(c - a);
                    float2 gC = 0.5f * Perp(a - b);

                    float denom = wi * math.lengthsq(gA) + wj * math.lengthsq(gB) + wk * math.lengthsq(gC) + alphaTilde + gammaDt;
                    if (denom <= Eps) continue;

                    float dLambda = -(C + alphaTilde * cache.lambdas.volume[k]) / denom;

                    ni.predPos += -wi * dLambda * gA;
                    nb.predPos += -wj * dLambda * gB;
                    nc.predPos += -wk * dLambda * gC;

                    cache.lambdas.volume[k] += dLambda;
                }
            }
        }


        static float Cross(in float2 a, in float2 b) => a.x * b.y - a.y * b.x;
        static float2 Perp(in float2 v) => new float2(-v.y, v.x);
    }
}