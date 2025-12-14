using Unity.Mathematics;

namespace Physics {
    public class XPBIConstraint {
        private int currentIteration;

        public void Initialise(NodeBatch batch) {
            batch.CacheNeighbors();
            batch.ComputeCorrectionMatrices();
            batch.ResetDebugData();
            currentIteration = 0;

            for (int i = 0; i < batch.Count; i++) {
                var cache = batch.caches[i];
                cache.F0 = batch.nodes[i].F;
            }
        }

        // XPBI Eq. (11) analogue: corrected neighbor-based velocity gradient estimate.
        private static float2x2 EstimateVelocityGradient(NodeBatch batch, int i) {
            var xi = batch.nodes[i];
            var ci = batch.caches[i];

            float2x2 gradV = float2x2.zero;
            var N = ci.neighbors;
            if (N == null || N.Count == 0) return gradV;

            for (int k = 0; k < N.Count; k++) {
                int j = N[k];
                if ((uint)j >= (uint)batch.Count) continue;

                float2 xij = batch.nodes[j].pos - xi.pos;
                float r2 = math.lengthsq(xij);
                if (r2 < Const.Eps * Const.Eps) continue;

                float w = 1.0f / r2;

                float2 correctedGrad = w * math.mul(ci.L, xij);
                float2 dv = batch.nodes[j].vel - xi.vel;

                gradV.c0 += dv * correctedGrad.x;
                gradV.c1 += dv * correctedGrad.y;
            }

            return gradV;
        }


        public void Relax(NodeBatch batch, float stiffness, float dt) {
            float invDt = 1.0f / math.max(dt, Const.Eps);

            for (int i = 0; i < batch.Count; i++) {
                var node = batch.nodes[i];
                var cache = batch.caches[i];
                var dbg = batch.debug[i];

                var N = cache?.neighbors;
                if (N == null || N.Count == 0) continue;

                if (node.invMass <= 0f && node.isFixed) continue;
                if (node.restVolume <= Const.Eps) continue;

                // XPBI Eq. (9): F_{n+1} = (I + ∇v dt) F_n  (updated-Lagrangian style).
                float2x2 Ftrial = math.mul(float2x2.identity + EstimateVelocityGradient(batch, i) * dt, cache.F0);

                // Plastic projection (implementation-specific, but driven by XPBI’s projected update loop idea).
                float2x2 Fel = DeformationUtils.ApplyPlasticityReturn(Ftrial, i, ref dbg);

                float C = DeformationUtils.XPBIConstraint(Fel, i);
                if (math.abs(C) < Const.Eps) {
                    dbg.RecordConstraintEval(C, 0f, cache.lambda, 0f, 0f, currentIteration);
                    continue;
                }
                dbg.constraintEnergy += math.abs(C);

                // XPBD compliance: α~ = α / dt^2, and Δλ is XPBD Eq. (18).
                float alphaTilde = ((1.0f / node.restVolume) * stiffness) * (invDt * invDt);

                float2x2 dCdF = DeformationUtils.ComputeGradient(Fel, i);

                float2 gradC_vi = float2.zero;
                var gradC_vj = new float2[N.Count];
                float2x2 FT = math.transpose(Fel);

                for (int k = 0; k < N.Count; k++) {
                    int j = N[k];
                    if ((uint)j >= (uint)batch.Count) continue;

                    float2 xij = batch.nodes[j].pos - node.pos;
                    float r2 = math.lengthsq(xij);
                    if (r2 < Const.Eps * Const.Eps) continue;

                    float w = 1.0f / r2;

                    float2 correctedGrad = w * math.mul(cache.L, xij);
                    float2 q = math.mul(dCdF, math.mul(FT, correctedGrad));

                    float2 g = q;
                    gradC_vi -= g;
                    gradC_vj[k] += g;
                }

                float gradCViLenSq = math.lengthsq(gradC_vi);

                float denom = node.invMass * gradCViLenSq;
                for (int k = 0; k < N.Count; k++) {
                    int j = N[k];
                    if ((uint)j >= (uint)batch.Count) continue;

                    if (batch.nodes[j].invMass <= 0f && batch.nodes[j].isFixed) continue;
                    denom += batch.nodes[j].invMass * math.lengthsq(gradC_vj[k]);
                }

                if (denom < Const.Eps) {
                    dbg.degenerateCount++;
                    dbg.RecordConstraintEval(C, denom, cache.lambda, 0f, gradCViLenSq, currentIteration);
                    continue;
                }

                // XPBD Eq. (18): Δλ = -(C + α~ λ) / (∑ w |∇C|^2 + α~).
                float lambdaBefore = cache.lambda;
                float dLambda = -(C + alphaTilde * lambdaBefore) / (denom + alphaTilde);

                if (float.IsNaN(dLambda) || float.IsInfinity(dLambda)) {
                    dbg.nanInfCount++;
                    dbg.RecordConstraintEval(C, denom, cache.lambda, dLambda, gradCViLenSq, currentIteration);
                    continue;
                }

                float velScale = dLambda * invDt;

                float2 dVi = node.invMass * velScale * gradC_vi;
                node.vel += dVi;

                dbg.RecordPositionUpdate(dVi * dt);
                dbg.iterationsToConverge = currentIteration + 1;

                for (int k = 0; k < N.Count; k++) {
                    int j = N[k];
                    if ((uint)j >= (uint)batch.Count) continue;

                    if (batch.nodes[j].invMass <= 0f && batch.nodes[j].isFixed) continue;

                    float2 dVj = batch.nodes[j].invMass * velScale * gradC_vj[k];
                    batch.nodes[j].vel += dVj;

                    var dbgJ = batch.debug[j];
                    dbgJ.RecordPositionUpdate(dVj * dt);
                    dbgJ.iterationsToConverge = math.max(dbgJ.iterationsToConverge, currentIteration + 1);
                }

                cache.lambda = lambdaBefore + dLambda;

                dbg.RecordConstraintEval(C, denom, cache.lambda, dLambda, gradCViLenSq, currentIteration);
            }

            batch.FinalizeDebugData();
            currentIteration++;
        }

        public void CommitDeformation(NodeBatch batch, float dt) {
            for (int i = 0; i < batch.Count; i++) {
                var node = batch.nodes[i];
                var cache = batch.caches[i];
                var dbg = batch.debug[i];

                var N = cache?.neighbors;
                if (N == null || N.Count == 0) continue;

                if (node.invMass <= 0f && node.isFixed) continue;

                float2x2 Ftrial = math.mul(float2x2.identity + EstimateVelocityGradient(batch, i) * dt, cache.F0);
                float2x2 Fel = DeformationUtils.ApplyPlasticityReturn(Ftrial, i, ref dbg);

                float det = Fel.c0.x * Fel.c1.y - Fel.c0.y * Fel.c1.x;
                if (math.abs(det) > Const.Eps) {
                    float invDet = 1.0f / det;
                    node.Fp = math.mul(
                        math.mul(new float2x2(
                            new float2(Fel.c1.y * invDet, -Fel.c0.y * invDet),
                            new float2(-Fel.c1.x * invDet, Fel.c0.x * invDet)
                        ), Ftrial),
                        node.Fp
                    );
                }

                node.F = Fel;
            }
        }
    }
}
