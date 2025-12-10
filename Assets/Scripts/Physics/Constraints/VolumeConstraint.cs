using Unity.Mathematics;
using UnityEngine;

namespace Physics {
    public class VolumeConstraint : Constraint {
        public string GetConstraintType() => "VolumeConstraint";

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
                if (cache?.neighbors == null || cache.neighbors.Count < 2) continue;

                float wi = node.invMass;
                if (wi <= 0f && node.isFixed) continue;

                // XPBI Eq. 9: Update elastic deformation from velocity gradient
                float2x2 F_elastic_projected = ApplyPlasticityReturnVolume(
                    float2x2.identity + EstimateVelocityGradient(data, i) * timeStep,
                    i,
                    ref debug
                );

                // Process each triangular area constraint (node i and two consecutive neighbors)
                for (int k = 0; k < cache.neighbors.Count; k++) {
                    int j = cache.neighbors[k];
                    int l = cache.neighbors[(k + 1) % cache.neighbors.Count];
                    if (j < 0 || j >= data.Count || l < 0 || l >= data.Count) continue;

                    var nb = data.nodes[j];
                    var nc = data.nodes[l];
                    float wj = nb.isFixed ? 0f : nb.invMass;
                    float wk = nc.isFixed ? 0f : nc.invMass;

                    // Current area (signed)
                    float currentArea = 0.5f * Cross(nb.pos - node.pos, nc.pos - node.pos);

                    // Target area: F_elastic * Fp * (original edges)
                    float2 targetEdge_j = math.mul(F_elastic_projected,
                                                   math.mul(node.Fp, nb.originalPos - node.originalPos));
                    float2 targetEdge_k = math.mul(F_elastic_projected,
                                                   math.mul(node.Fp, nc.originalPos - node.originalPos));
                    float targetArea = 0.5f * Cross(targetEdge_j, targetEdge_k);

                    if (math.abs(targetArea) < Const.Eps) continue;

                    // Constraint: C = currentArea - targetArea
                    float C = currentArea - targetArea;

                    // Constraint gradients (perpendicular to edges)
                    float2 gradC_i = 0.5f * Perp(nb.pos - nc.pos);
                    float2 gradC_j = 0.5f * Perp(nc.pos - node.pos);
                    float2 gradC_k = 0.5f * Perp(node.pos - nb.pos);

                    float denominator = wi * math.lengthsq(gradC_i) +
                                       wj * math.lengthsq(gradC_j) +
                                       wk * math.lengthsq(gradC_k);

                    if (denominator < Const.Eps) {
                        debug.degenerateCount++;
                        continue;
                    }

                    // XPBD with compliance: Δλ = -(C + α̃λ) / (∇C·M⁻¹·∇C + α̃)
                    float lambdaAccum = k < cache.lambdas.volume.Count ? cache.lambdas.volume[k] : 0f;
                    float deltaLambda = -(C + alphaTilde * lambdaAccum) / (denominator + alphaTilde);

                    if (float.IsNaN(deltaLambda) || float.IsInfinity(deltaLambda)) {
                        debug.nanInfCount++;
                        continue;
                    }

                    // Update velocities: Δv = (1/m) * Δλ * ∇C / Δt
                    float velScale = deltaLambda / timeStep;
                    node.vel += wi * velScale * gradC_i;
                    nb.vel += wj * velScale * gradC_j;
                    nc.vel += wk * velScale * gradC_k;

                    // Accumulate lambda
                    while (cache.lambdas.volume.Count <= k) {
                        cache.lambdas.volume.Add(0f);
                    }
                    cache.lambdas.volume[k] += deltaLambda;

                    debug.RecordPositionUpdate(wi * velScale * gradC_i * timeStep);
                    debug.constraintEnergy += math.abs(C);
                }

                debug.iterationsToConverge = currentIteration + 1;
            }

            data.FinalizeDebugData(GetConstraintType());
            currentIteration++;
        }

        // XPBI Eq. 14: Plasticity return mapping for area/volume preservation
        private float2x2 ApplyPlasticityReturnVolume(float2x2 F_elastic, int nodeIdx, ref ConstraintDebugData debug) {
            const float yieldAreaFrac = 1.05f;

            // Polar decomposition: F = R * S
            DeformationUtils.PolarDecompose2D(F_elastic, out float2x2 R, out float2x2 S,
                                              out float s1, out float s2, nodeIdx);

            // Area stretch = det(F) = s1 * s2
            float areaStretch = s1 * s2;

            if (!(areaStretch > yieldAreaFrac || areaStretch < 1f / yieldAreaFrac)) {
                return F_elastic;
            }

            debug.plasticFlowCount++;

            // Clamp area stretch while preserving aspect ratio
            float scaleFactor = math.sqrt(
                math.clamp(areaStretch, 1f / yieldAreaFrac, yieldAreaFrac) / areaStretch
            );
            float clamped_s1 = s1 * scaleFactor;
            float clamped_s2 = s2 * scaleFactor;

            // Reconstruct S with clamped stretches
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
            const float yieldAreaFrac = 1.05f;

            for (int i = 0; i < data.Count; i++) {
                var node = data.nodes[i];
                if (data.caches[i]?.neighbors == null || data.caches[i].neighbors.Count < 2) continue;

                float2x2 F_elastic = float2x2.identity + EstimateVelocityGradient(data, i) * timeStep;

                DeformationUtils.PolarDecompose2D(F_elastic, out float2x2 R, out float2x2 S,
                                                  out float s1, out float s2, i);

                float areaStretch = s1 * s2;
                if (!(areaStretch > yieldAreaFrac || areaStretch < 1f / yieldAreaFrac)) {
                    continue;
                }

                // Update Fp: accumulate plastic deformation
                // Note: Simplified multiplicative update, see XPBI paper for full derivation
                node.Fp = math.mul(F_elastic, node.Fp);
            }

            currentIteration = 0;
        }

        static float Cross(in float2 a, in float2 b) => a.x * b.y - a.y * b.x;
        static float2 Perp(in float2 v) => new float2(-v.y, v.x);
    }
}
