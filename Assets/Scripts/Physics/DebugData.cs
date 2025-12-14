using System.Collections.Generic;
using Unity.Mathematics;

namespace Physics {

    /// <summary>
    /// Debug data for the universal XPBI constitutive constraint per node.
    /// Collected across solver iterations within a frame.
    /// </summary>
    public class DebugData {
        // ---- Position update statistics (Δx on i, approximated from Δv * dt) ----
        public float maxPositionDelta = 0f;
        public float sumPositionDelta = 0f;
        public float avgPositionDelta = 0f;
        public int positionUpdateCount = 0;

        // ---- Constraint residual (C(F) = sqrt(2 Psi(F))) ----
        public float lastC = 0f;
        public float maxAbsC = 0f;
        public float sumAbsC = 0f;
        public float avgAbsC = 0f;
        public int constraintEvalCount = 0;

        // ---- Lagrange multiplier tracking ----
        public float lastLambda = 0f;          // lambda before update (accumulated)
        public float lastDeltaLambda = 0f;     // delta applied this step
        public float maxAbsLambda = 0f;
        public float maxAbsDeltaLambda = 0f;
        public float sumAbsDeltaLambda = 0f;

        // ---- Denominator / conditioning ----
        public float lastDenominator = 0f;
        public float minDenominator = float.PositiveInfinity;
        public float maxDenominator = 0f;

        // ---- Gradient magnitude proxy (for sensitivity / stiffness hotspots) ----
        public float lastGradCViLenSq = 0f;
        public float maxGradCViLenSq = 0f;

        // ---- Convergence / iteration ----
        public int lastIterationUpdated = 0;   // most recent iteration that changed the node
        public int iterationsToConverge = 0;   // keeps your previous meaning: last iteration+1 that updated

        // ---- Energy-ish aggregate (you previously used |C| as "energy") ----
        public float constraintEnergy = 0f;

        // ---- Error and issue counters ----
        public int degenerateCount = 0;
        public int nanInfCount = 0;
        public int plasticFlowCount = 0;

        public void Reset() {
            maxPositionDelta = 0f;
            sumPositionDelta = 0f;
            avgPositionDelta = 0f;
            positionUpdateCount = 0;

            lastC = 0f;
            maxAbsC = 0f;
            sumAbsC = 0f;
            avgAbsC = 0f;
            constraintEvalCount = 0;

            lastLambda = 0f;
            lastDeltaLambda = 0f;
            maxAbsLambda = 0f;
            maxAbsDeltaLambda = 0f;
            sumAbsDeltaLambda = 0f;

            lastDenominator = 0f;
            minDenominator = float.PositiveInfinity;
            maxDenominator = 0f;

            lastGradCViLenSq = 0f;
            maxGradCViLenSq = 0f;

            lastIterationUpdated = 0;
            iterationsToConverge = 0;

            constraintEnergy = 0f;

            degenerateCount = 0;
            nanInfCount = 0;
            plasticFlowCount = 0;
        }

        public void RecordConstraintEval(float C, float denom, float lambdaAccum, float deltaLambda, float gradCViLenSq, int iteration) {
            lastC = C;
            float absC = math.abs(C);
            maxAbsC = math.max(maxAbsC, absC);
            sumAbsC += absC;
            constraintEvalCount++;

            lastDenominator = denom;
            minDenominator = math.min(minDenominator, denom);
            maxDenominator = math.max(maxDenominator, denom);

            lastLambda = lambdaAccum;
            lastDeltaLambda = deltaLambda;
            maxAbsLambda = math.max(maxAbsLambda, math.abs(lambdaAccum));
            maxAbsDeltaLambda = math.max(maxAbsDeltaLambda, math.abs(deltaLambda));
            sumAbsDeltaLambda += math.abs(deltaLambda);

            lastGradCViLenSq = gradCViLenSq;
            maxGradCViLenSq = math.max(maxGradCViLenSq, gradCViLenSq);

            lastIterationUpdated = iteration;
        }

        public void RecordPositionUpdate(float2 deltaX) {
            float magnitude = math.length(deltaX);
            maxPositionDelta = math.max(maxPositionDelta, magnitude);
            sumPositionDelta += magnitude;
            positionUpdateCount++;
        }

        public void FinalizeAverages() {
            if (positionUpdateCount > 0) avgPositionDelta = sumPositionDelta / positionUpdateCount;
            if (constraintEvalCount > 0) avgAbsC = sumAbsC / constraintEvalCount;
        }
    }
}