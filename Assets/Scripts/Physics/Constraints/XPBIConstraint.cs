using Unity.Mathematics;

namespace Physics {
    public static class XPBIConstraint {

        private static float2x2 EstimateVelocityGradient(NodeBatch batch, int i) {
            // XPBI Eq. (11): ∂v/∂x = Σ_b V_b^n (v_b - v_p) (L_p ∇W_b(x_p))^T
            // In 2D we store the 2x2 gradient with columns corresponding to x/y partials.
            var xi = batch.nodes[i];
            var ci = batch.caches[i];

            float2x2 gradV = float2x2.zero;
            var N = ci.neighbors;
            if (N == null || N.Count == 0) return gradV;

            float h = ci.kernelH;
            if (h <= Const.Eps) return gradV;

            for (int k = 0; k < N.Count; k++) {
                int j = N[k];
                if ((uint)j >= (uint)batch.Count) continue;

                float2 xij = batch.nodes[j].pos - xi.pos;

                float2 gradW = SPHKernels.GradWendlandC2(xij, h);
                if (math.lengthsq(gradW) <= Const.Eps * Const.Eps) continue;

                float Vb = batch.caches[j].currentVolume;
                if (Vb <= Const.Eps) continue;

                // XPBI Eq. (10): corrected kernel gradient uses L_p:
                // \tilde{∇W} = L_p ∇W
                float2 correctedGrad = math.mul(ci.L, gradW);

                float2 dv = batch.nodes[j].vel - xi.vel;

                gradV.c0 += dv * (Vb * correctedGrad.x);
                gradV.c1 += dv * (Vb * correctedGrad.y);
            }

            return gradV;
        }

        public static void Relax(NodeBatch batch, float compliance, float dt, int currentIteration) {
            float invDt = 1.0f / math.max(dt, Const.Eps);

            for (int i = 0; i < batch.Count; i++) {
                var node = batch.nodes[i];
                var cache = batch.caches[i];
                var dbg = batch.debug[i];

                var N = cache.neighbors;
                if (N == null || N.Count == 0) continue;

                if (node.isFixed || node.invMass <= 0f) continue;
                if (node.restVolume <= Const.Eps) continue;
                if (cache.kernelH <= Const.Eps) continue;

                // Updated Lagrangian form used by XPBI: F_trial = (I + dt ∇v) F0.
                long t = LoopProfiler.Stamp();
                float2x2 gradV = EstimateVelocityGradient(batch, i);
                LoopProfiler.Add(LoopProfiler.Section.RelaxEstimateGradV, t);

                t = LoopProfiler.Stamp();
                float2x2 Ftrial = math.mul(float2x2.identity + gradV * dt, cache.F0);
                LoopProfiler.Add(LoopProfiler.Section.RelaxFtrial, t);

                // Plastic projection in Hencky space is used for evaluation (implicit-like), but Fp is committed later.
                t = LoopProfiler.Stamp();
                float2x2 Fel = DeformationUtils.ApplyPlasticityReturn(Ftrial, i, ref dbg);
                LoopProfiler.Add(LoopProfiler.Section.RelaxPlasticityReturn, t);

                t = LoopProfiler.Stamp();
                float C = DeformationUtils.XPBIConstraint(Fel, i);
                LoopProfiler.Add(LoopProfiler.Section.RelaxConstraintEval, t);

                if (math.abs(C) < Const.Eps) {
                    dbg.RecordConstraintEval(C, 0f, cache.lambda, 0f, 0f, currentIteration);
                    continue;
                }
                dbg.constraintEnergy += math.abs(C);

                // XPBD softness: α~ = (compliance / V_rest) / dt^2.
                float alphaTilde = (compliance / math.max(node.restVolume, Const.Eps)) * (invDt * invDt);

                t = LoopProfiler.Stamp();
                float2x2 dCdF = DeformationUtils.ComputeGradient(Fel, i);
                LoopProfiler.Add(LoopProfiler.Section.RelaxComputeGradient, t);

                float2 gradC_vi = float2.zero;
                float2[] gradC_vj = cache.gradC_vj;
                for (int k = 0; k < Const.NeighborCount; k++) gradC_vj[k] = float2.zero;

                float2x2 FT = math.transpose(Fel);

                // XPBI Eq. (12): ∇_{x_b} C_p = V_b^n (∂C_p/∂F_p) F_p^T (L_p ∇W_b(x_p)).
                t = LoopProfiler.Stamp();
                for (int k = 0; k < N.Count && k < Const.NeighborCount; k++) {
                    int j = N[k];
                    if ((uint)j >= (uint)batch.Count) continue;

                    float2 xij = batch.nodes[j].pos - node.pos;

                    float2 gradW = SPHKernels.GradWendlandC2(xij, cache.kernelH);
                    if (math.lengthsq(gradW) <= Const.Eps * Const.Eps) continue;

                    float Vb = batch.caches[j].currentVolume;
                    if (Vb <= Const.Eps) continue;

                    float2 correctedGrad = math.mul(cache.L, gradW);
                    float2 q = Vb * math.mul(dCdF, math.mul(FT, correctedGrad));

                    gradC_vi -= q;
                    gradC_vj[k] = q;
                }
                LoopProfiler.Add(LoopProfiler.Section.RelaxGradCAccum, t);

                float gradCViLenSq = math.lengthsq(gradC_vi);

                t = LoopProfiler.Stamp();
                float denom = node.invMass * gradCViLenSq;
                for (int k = 0; k < N.Count && k < Const.NeighborCount; k++) {
                    int j = N[k];
                    if ((uint)j >= (uint)batch.Count) continue;

                    if (batch.nodes[j].isFixed || batch.nodes[j].invMass <= 0f) continue;
                    denom += batch.nodes[j].invMass * math.lengthsq(gradC_vj[k]);
                }
                LoopProfiler.Add(LoopProfiler.Section.RelaxDenomAccum, t);

                if (denom < Const.Eps) {
                    dbg.degenerateCount++;
                    dbg.RecordConstraintEval(C, denom, cache.lambda, 0f, gradCViLenSq, currentIteration);
                    continue;
                }

                t = LoopProfiler.Stamp();
                // XPBD update (XPBI Eq. (13) style): Δλ = -(C + α~ λ) / (Σ w||∇C||^2 + α~).
                float lambdaBefore = cache.lambda;
                float dLambda = -(C + alphaTilde * lambdaBefore) / (denom + alphaTilde);
                LoopProfiler.Add(LoopProfiler.Section.RelaxLambdaUpdate, t);

                if (float.IsNaN(dLambda) || float.IsInfinity(dLambda)) {
                    dbg.nanInfCount++;
                    dbg.RecordConstraintEval(C, denom, cache.lambda, dLambda, gradCViLenSq, currentIteration);
                    continue;
                }

                // Velocity-first form: Δv = (w Δλ / dt) ∇C.
                float velScale = dLambda * invDt;

                t = LoopProfiler.Stamp();

                float2 dVi = node.invMass * velScale * gradC_vi;
                node.vel += dVi;

                dbg.RecordPositionUpdate(dVi * dt);
                dbg.iterationsToConverge = currentIteration + 1;

                for (int k = 0; k < N.Count && k < Const.NeighborCount; k++) {
                    int j = N[k];
                    if ((uint)j >= (uint)batch.Count) continue;

                    if (batch.nodes[j].isFixed || batch.nodes[j].invMass <= 0f) continue;

                    float2 dVj = batch.nodes[j].invMass * velScale * gradC_vj[k];
                    batch.nodes[j].vel += dVj;

                    var dbgJ = batch.debug[j];
                    dbgJ.RecordPositionUpdate(dVj * dt);
                    dbgJ.iterationsToConverge = math.max(dbgJ.iterationsToConverge, currentIteration + 1);
                }

                cache.lambda = lambdaBefore + dLambda;

                dbg.RecordConstraintEval(C, denom, cache.lambda, dLambda, gradCViLenSq, currentIteration);

                LoopProfiler.Add(LoopProfiler.Section.RelaxApplyVelocities, t);
            }
        }

        public static void CommitDeformation(NodeBatch batch, float dt) {
            for (int i = 0; i < batch.Count; i++) {
                var node = batch.nodes[i];
                var cache = batch.caches[i];
                var dbg = batch.debug[i];

                var N = cache.neighbors;
                if (N == null || N.Count == 0) continue;

                if (node.isFixed || node.invMass <= 0f) continue;
                if (cache.kernelH <= Const.Eps) continue;

                long t = LoopProfiler.Stamp();
                float2x2 gradV = EstimateVelocityGradient(batch, i);
                LoopProfiler.Add(LoopProfiler.Section.CommitEstimateGradV, t);

                t = LoopProfiler.Stamp();
                float2x2 Ftrial = math.mul(float2x2.identity + gradV * dt, cache.F0);
                LoopProfiler.Add(LoopProfiler.Section.CommitFtrial, t);

                t = LoopProfiler.Stamp();
                float2x2 Fel = DeformationUtils.ApplyPlasticityReturn(Ftrial, i, ref dbg);
                LoopProfiler.Add(LoopProfiler.Section.CommitPlasticityReturn, t);

                t = LoopProfiler.Stamp();
                // Multiplicative plasticity update:
                // F = Fel * Fp, and we want Fp_{n+1} such that Fel^{-1} F_trial updates plastic part.
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
                LoopProfiler.Add(LoopProfiler.Section.CommitFpUpdate, t);
            }
        }
    }
}
