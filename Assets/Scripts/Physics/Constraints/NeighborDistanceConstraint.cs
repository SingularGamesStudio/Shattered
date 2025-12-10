using Unity.Mathematics;
using UnityEngine;

namespace Physics {
    public class NeighborDistanceConstraint : Constraint {
        public string GetConstraintType() => "NeighborDistanceConstraint";

        private int currentIteration = 0;

        public void Initialise(NodeBatch nodes) {
            nodes.CacheNeighbors();
            nodes.ComputeCorrectionMatrices();
            nodes.ResetDebugData(GetConstraintType());

            // Initialize Fp (plastic deformation gradient) to identity
            for (int i = 0; i < nodes.Count; i++) {
                nodes.nodes[i].Fp = float2x2.identity;
            }

            currentIteration = 0;
        }

        // XPBI Eq. 11: Velocity gradient estimation with kernel gradient correction
        private float2x2 EstimateVelocityGradient(NodeBatch batch, int particleIdx) {
            var node = batch.nodes[particleIdx];
            var cache = batch.caches[particleIdx];
            float2x2 velocitySum = float2x2.zero;

            for (int k = 0; k < cache.neighbors.Count; k++) {
                int j = cache.neighbors[k];
                if (j < 0 || j >= batch.Count) continue;

                float2 xij = batch.nodes[j].pos - node.pos;
                float dist = math.length(xij);
                if (dist < Const.Eps) continue;

                // Corrected gradient: L_p * (w * ∇W)
                float2 correctedGrad = math.mul(cache.L, (1.0f / dist) * math.normalize(xij));
                float2 velDiff = batch.nodes[j].vel - node.vel;

                velocitySum.c0 += velDiff * correctedGrad.x;
                velocitySum.c1 += velDiff * correctedGrad.y;
            }

            return velocitySum;
        }

        public void Relax(NodeBatch data, float compliance, float timeStep) {
            float alphaTilde = compliance / (timeStep * timeStep);

            for (int i = 0; i < data.Count; i++) {
                var node = data.nodes[i];
                var cache = data.caches[i];
                var debug = data.GetOrCreateDebugData(i, GetConstraintType());
                if (cache?.neighbors == null) continue;

                float wi = node.invMass;
                if (wi <= 0f && node.isFixed) continue;

                // XPBI Eq. 9: Update elastic deformation from velocity gradient
                float2x2 F_elastic_projected = ApplyPlasticityReturn(
                    float2x2.identity + EstimateVelocityGradient(data, i) * timeStep,
                    i,
                    ref debug
                );

                // Process each edge constraint
                for (int k = 0; k < cache.neighbors.Count; k++) {
                    int j = cache.neighbors[k];
                    if (j < 0 || j >= data.Count || j == i) continue;

                    var neighbor = data.nodes[j];
                    float wj = neighbor.invMass;
                    if (wi + wj <= 0f) continue;

                    float2 r = node.pos - neighbor.pos;
                    float len = math.length(r);

                    if (len < Const.Eps) {
                        debug.degenerateCount++;
                        continue;
                    }

                    // Target edge: F_elastic * Fp * (original edge)
                    float2 targetEdge = math.mul(F_elastic_projected,
                                                 math.mul(node.Fp, neighbor.originalPos - node.originalPos));
                    float targetLen = math.length(targetEdge);
                    if (targetLen < Const.Eps) continue;

                    // Constraint: C = |current| - |target|
                    float C = len - targetLen;
                    float2 n = r / len;

                    float denominator = (wi + wj) * math.lengthsq(n);
                    if (denominator < Const.Eps) {
                        debug.degenerateCount++;
                        continue;
                    }

                    // XPBD with compliance: Δλ = -(C + α̃λ) / (∇C·M⁻¹·∇C + α̃)
                    float lambdaAccum = k < cache.lambdas.neighborDistance.Count ?
                                       cache.lambdas.neighborDistance[k] : 0f;
                    float deltaLambda = -(C + alphaTilde * lambdaAccum) / (denominator + alphaTilde);

                    if (float.IsNaN(deltaLambda) || float.IsInfinity(deltaLambda)) {
                        debug.nanInfCount++;
                        continue;
                    }

                    // Update velocities: Δv = (1/m) * Δλ * ∇C / Δt
                    float velScale = deltaLambda / timeStep;
                    float2 deltaVel = velScale * n;
                    node.vel += wi * deltaVel;
                    neighbor.vel -= wj * deltaVel;

                    // Accumulate lambda
                    while (cache.lambdas.neighborDistance.Count <= k) {
                        cache.lambdas.neighborDistance.Add(0f);
                    }
                    cache.lambdas.neighborDistance[k] += deltaLambda;

                    debug.RecordPositionUpdate(wi * deltaVel * timeStep);
                    debug.constraintEnergy += math.abs(C);
                }

                debug.iterationsToConverge = currentIteration + 1;
            }

            data.FinalizeDebugData(GetConstraintType());
            currentIteration++;
        }

        // XPBI Eq. 14: Plasticity return mapping
        private float2x2 ApplyPlasticityReturn(float2x2 F_elastic, int nodeIdx, ref ConstraintDebugData debug) {
            const float yieldStretch = 1.05f;

            // Polar decomposition: F = R * S
            DeformationUtils.PolarDecompose2D(F_elastic, out float2x2 R, out float2x2 S,
                                              out float s1, out float s2, nodeIdx);

            // Check yield condition on principal stretches
            if (!(math.max(s1, s2) > yieldStretch || math.min(s1, s2) < 1f / yieldStretch)) {
                return F_elastic;
            }

            debug.plasticFlowCount++;

            float clamped_s1 = math.clamp(s1, 1f / yieldStretch, yieldStretch);
            float clamped_s2 = math.clamp(s2, 1f / yieldStretch, yieldStretch);

            // Reconstruct S with clamped principal stretches
            float2x2 V;
            if (math.abs(S.c0.y) > 1e-5f) {
                // Compute eigenvectors from symmetric S
                V = new float2x2(
                    math.normalize(new float2(clamped_s1 * clamped_s1 - S.c1.y, S.c0.y)),
                    math.normalize(new float2(clamped_s2 * clamped_s2 - S.c1.y, S.c0.y))
                );
            } else {
                V = float2x2.identity;
            }

            float2x2 S_clamped = math.mul(
                math.mul(V, new float2x2(new float2(clamped_s1, 0f), new float2(0f, clamped_s2))),
                math.transpose(V)
            );

            return math.mul(R, S_clamped);
        }

        // XPBI Eq. 22: Update plastic deformation after convergence
        public void UpdatePlasticDeformation(NodeBatch data, float timeStep) {
            const float yieldStretch = 1.05f;

            for (int i = 0; i < data.Count; i++) {
                var node = data.nodes[i];
                if (data.caches[i]?.neighbors == null) continue;

                float2x2 F_elastic = float2x2.identity + EstimateVelocityGradient(data, i) * timeStep;

                DeformationUtils.PolarDecompose2D(F_elastic, out float2x2 R, out float2x2 S,
                                                  out float s1, out float s2, i);

                if (!(math.max(s1, s2) > yieldStretch || math.min(s1, s2) < 1f / yieldStretch)) {
                    continue;
                }

                // Update Fp: accumulate plastic deformation
                // Note: Simplified multiplicative update
                node.Fp = math.mul(F_elastic, node.Fp);
            }

            currentIteration = 0;
        }
    }
}
