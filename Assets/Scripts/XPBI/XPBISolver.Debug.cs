using UnityEngine;

namespace GPU.Solver {
    public sealed partial class XPBISolver {
        private const int ConvergenceDebugIterBufSize = 8;
        private const float ConvergenceDebugScaleC = 1_000_000f;
        private const float ConvergenceDebugScaleDLambda = 1_000_000f;
        private void LogConvergenceStatsFromData(uint[] data, int maxSolveLayer, int maxIter) {
            if (data == null || data.Length == 0)
                return;

            float invScaleC = 1f / ConvergenceDebugScaleC;
            float invScaleDL = 1f / ConvergenceDebugScaleDLambda;

            for (int layer = maxSolveLayer; layer >= 0; layer--) {
                int iterations = GetIterationsForLayer(layer, maxSolveLayer);
                int baseIter = layer * maxIter;
                int gsSplitIter = (layer == 0 && Const.EnableTwoStageGS2)
                    ? Mathf.Min(Const.TwoStagePreGsItersL0, iterations)
                    : -1;

                var table = new System.Text.StringBuilder();
                table.AppendLine($"Layer {layer} convergence stats:");
                table.AppendLine("Iter | Marker | Count | Avg|C|       | Max|C|       | Avg|dLambda| | Max|dLambda|");
                table.AppendLine("-----|--------|-------|--------------|--------------|---------------|---------------");

                bool hasData = false;

                for (int iter = 0; iter < iterations; iter++) {
                    if (iter == gsSplitIter) {
                        table.AppendLine("-----|--------|-------|--------------|--------------|---------------|---------------");
                    }

                    int baseU = (baseIter + iter) * ConvergenceDebugIterBufSize;
                    if (baseU + 7 >= data.Length)
                        continue;

                    uint sumAbsC = data[baseU + 0];
                    uint maxAbsC = data[baseU + 1];
                    uint sumAbsDL = data[baseU + 2];
                    uint maxAbsDL = data[baseU + 3];
                    uint cnt = data[baseU + 4];
                    uint marker = data[baseU + 7];

                    if (marker == 0 && cnt == 0 && sumAbsC == 0 && maxAbsC == 0 && sumAbsDL == 0 && maxAbsDL == 0)
                        continue;

                    hasData = true;

                    float avgAbsC = cnt > 0 ? (sumAbsC * invScaleC) / cnt : 0f;
                    float mxAbsC = maxAbsC * invScaleC;
                    float avgAbsDL = cnt > 0 ? (sumAbsDL * invScaleDL) / cnt : 0f;
                    float mxAbsDL = maxAbsDL * invScaleDL;

                    table.AppendLine(
                        $"{iter,4} | {marker,6} | {cnt,5} | {avgAbsC,12:G6} | {mxAbsC,12:G6} | {avgAbsDL,13:G6} | {mxAbsDL,13:G6}");
                }

                if (hasData) {
                    Debug.LogError(table.ToString());
                }
            }
        }

        void ClearDebugBuffer(int layer, int iterations) {
            asyncCb.SetComputeIntParam(shader, "_ConvergenceDebugEnable", 1);
            asyncCb.SetComputeFloatParam(shader, "_ConvergenceDebugScaleC", ConvergenceDebugScaleC);
            asyncCb.SetComputeFloatParam(shader, "_ConvergenceDebugScaleDLambda", ConvergenceDebugScaleDLambda);
            asyncCb.SetComputeIntParam(shader, "_ConvergenceDebugOffset", layer * convergenceDebugMaxIter);
            asyncCb.SetComputeIntParam(shader, "_ConvergenceDebugIterCount", iterations);

            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_ConvergenceDebug", convergenceDebug);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_ConvergenceDebug", convergenceDebug);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_ConvergenceDebug", convergenceDebug);
            asyncCb.SetComputeBufferParam(shader, kClearConvergenceDebugStats, "_ConvergenceDebug", convergenceDebug);

            Dispatch("XPBI.ClearConvergenceDebugStats", shader, kClearConvergenceDebugStats, (iterations + 255) / 256, 1, 1);
        }
    }
}