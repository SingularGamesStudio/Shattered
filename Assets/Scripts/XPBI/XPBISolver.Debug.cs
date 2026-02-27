using System.Collections.Generic;
using System.Globalization;
using System.IO;
using UnityEngine;
using UnityEngine.Rendering;

namespace GPU.Solver {
    public sealed partial class XPBISolver {
        private const int ConvergenceDebugIterBufSize = 8;
        private const float ConvergenceDebugScaleC = 1_000_000f;
        private const float ConvergenceDebugScaleDLambda = 1_000_000f;

        private readonly struct ProlongationConstraintProbe {
            public readonly int Tick;
            public readonly int Layer;
            public readonly int PreEntry;
            public readonly int PostEntry;

            public ProlongationConstraintProbe(int tick, int layer, int preEntry, int postEntry) {
                Tick = tick;
                Layer = layer;
                PreEntry = preEntry;
                PostEntry = postEntry;
            }
        }

        private bool[] coloringUpdatedByLayerForDebugLog;
        private int convergenceDebugStepCursor;
        private string convergenceDebugCsvPath;

        private (float avgAbsC, float maxAbsC, uint count, uint marker) ReadConstraintErrorStat(uint[] data, int entryIndex) {
            if (data == null || entryIndex < 0)
                return (0f, 0f, 0u, 0u);

            int baseU = entryIndex * ConvergenceDebugIterBufSize;
            if (baseU + 4 >= data.Length)
                return (0f, 0f, 0u, 0u);

            uint sumAbsC = data[baseU + 0];
            uint maxAbsC = data[baseU + 1];
            uint cnt = data[baseU + 4];
            uint marker = baseU + 7 < data.Length ? data[baseU + 7] : 0u;
            float invScaleC = 1f / ConvergenceDebugScaleC;
            float avgAbsC = cnt > 0 ? (sumAbsC * invScaleC) / cnt : 0f;
            float mxAbsC = maxAbsC * invScaleC;
            return (avgAbsC, mxAbsC, cnt, marker);
        }

        private void LogProlongationConstraintStatsFromData(uint[] data, IReadOnlyList<ProlongationConstraintProbe> probes) {
            if (data == null || probes == null || probes.Count == 0)
                return;

            for (int i = 0; i < probes.Count; i++) {
                var probe = probes[i];
                var pre = ReadConstraintErrorStat(data, probe.PreEntry);
                var post = ReadConstraintErrorStat(data, probe.PostEntry);
                float deltaAvg = post.avgAbsC - pre.avgAbsC;

                Debug.LogError(
                    $"Prolongation constraint error L{probe.Layer} T{probe.Tick}: pre(avg|max)={pre.avgAbsC:G6}|{pre.maxAbsC:G6} post(avg|max)={post.avgAbsC:G6}|{post.maxAbsC:G6} deltaAvg={deltaAvg:G6} counts={pre.count}->{post.count} markers={pre.marker}->{post.marker}");
            }
        }

        private void CaptureCurrentConstraintErrorSample(int entryIndex, int originalActiveCount, int probeActiveCount, bool restoreOwnerFilter) {
            if (entryIndex < 0 || probeActiveCount <= 0 || prolongationConstraintDebug == null)
                return;

            ComputeBuffer restoredConvergenceDebug = convergenceDebug ?? convergenceDebugFallback ?? prolongationConstraintDebug;

            asyncCb.SetComputeIntParam(shader, "_ConvergenceDebugEnable", 1);
            asyncCb.SetComputeFloatParam(shader, "_ConvergenceDebugScaleC", ConvergenceDebugScaleC);
            asyncCb.SetComputeFloatParam(shader, "_ConvergenceDebugScaleDLambda", ConvergenceDebugScaleDLambda);
            asyncCb.SetComputeIntParam(shader, "_ConvergenceDebugOffset", entryIndex);
            asyncCb.SetComputeIntParam(shader, "_ConvergenceDebugIter", 0);
            asyncCb.SetComputeIntParam(shader, "_ConvergenceDebugIterCount", 1);
            asyncCb.SetComputeIntParam(shader, "_ActiveCount", probeActiveCount);
            asyncCb.SetComputeIntParam(shader, "_UseDtOwnerFilter", 0);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_ConvergenceDebug", prolongationConstraintDebug);
            asyncCb.SetComputeBufferParam(shader, kClearConvergenceDebugStats, "_ConvergenceDebug", prolongationConstraintDebug);

            Dispatch("XPBI.ClearConvergenceDebugStats.ProlongationProbe", shader, kClearConvergenceDebugStats, 1, 1, 1);
            Dispatch("XPBI.JR.SavePrevAndClear.ProlongationProbe", shader, kJRSavePrevAndClear, Groups256(probeActiveCount), 1, 1);
            Dispatch("XPBI.JR.ComputeDeltas.ProlongationProbe", shader, kJRComputeDeltas, Groups256(probeActiveCount), 1, 1);

            asyncCb.SetComputeIntParam(shader, "_ActiveCount", originalActiveCount);
            asyncCb.SetComputeIntParam(shader, "_UseDtOwnerFilter", restoreOwnerFilter ? 1 : 0);
            asyncCb.SetComputeIntParam(shader, "_ConvergenceDebugEnable", 0);

            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_ConvergenceDebug", restoredConvergenceDebug);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_ConvergenceDebug", restoredConvergenceDebug);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_ConvergenceDebug", restoredConvergenceDebug);
            asyncCb.SetComputeBufferParam(shader, kClearConvergenceDebugStats, "_ConvergenceDebug", restoredConvergenceDebug);
        }

        private void LogConvergenceStatsFromData(uint[] data, int maxSolveLayer, int maxIter, int tickCount, int baseStep) {
            if (data == null || data.Length == 0)
                return;

            float invScaleC = 1f / ConvergenceDebugScaleC;
            float invScaleDL = 1f / ConvergenceDebugScaleDLambda;

            AppendLayer0ConvergenceCsv(data, maxSolveLayer, maxIter, tickCount, baseStep, invScaleC, invScaleDL);

            int logTick = Mathf.Clamp(tickCount - 1, 0, int.MaxValue);

            for (int layer = maxSolveLayer; layer >= 0; layer--) {
                int JRiterations = GetJRIterationsForLayer(layer, maxSolveLayer);
                int baseIter = ((logTick * convergenceDebugLayers) + layer) * maxIter;
                int GSiterations = (layer == 0) ? Const.GSIterationsL0 : 1;

                var table = new System.Text.StringBuilder();
                table.AppendLine($"Layer {layer} convergence stats:");
                table.AppendLine("Iter | Marker | Count | Avg|C|       | Max|C|       | Avg|dLambda| | Max|dLambda|");
                table.AppendLine("-----|--------|-------|--------------|--------------|---------------|---------------");

                for (int iter = 0; iter < JRiterations + GSiterations; iter++) {
                    if (iter == GSiterations) {
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

                    float avgAbsC = cnt > 0 ? (sumAbsC * invScaleC) / cnt : 0f;
                    float mxAbsC = maxAbsC * invScaleC;
                    float avgAbsDL = cnt > 0 ? (sumAbsDL * invScaleDL) / cnt : 0f;
                    float mxAbsDL = maxAbsDL * invScaleDL;

                    table.AppendLine(
                        $"{iter,4} | {marker,6} | {cnt,5} | {avgAbsC,12:G6} | {mxAbsC,12:G6} | {avgAbsDL,13:G6} | {mxAbsDL,13:G6}");
                }

                Debug.LogError(table.ToString());
            }
        }

        private void AppendLayer0ConvergenceCsv(uint[] data, int maxSolveLayer, int maxIter, int tickCount, int baseStep, float invScaleC, float invScaleDL) {
            int layer = 0;
            int gsIterations = Const.GSIterationsL0;
            int jrIterations = GetJRIterationsForLayer(layer, maxSolveLayer);
            int totalIterations = gsIterations + jrIterations;
            if (totalIterations <= 0)
                return;

            if (string.IsNullOrEmpty(convergenceDebugCsvPath))
                convergenceDebugCsvPath = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "convergence_stats.csv"));

            bool writeHeader = !File.Exists(convergenceDebugCsvPath) || new FileInfo(convergenceDebugCsvPath).Length == 0;
            using var writer = new StreamWriter(convergenceDebugCsvPath, append: true);
            if (writeHeader)
                writer.WriteLine("step,in_step_iteration,marker,count,avg_abs_c,max_abs_c,avg_abs_dlambda,max_abs_dlambda");

            for (int tick = 0; tick < tickCount; tick++) {
                int step = baseStep + tick;
                int baseIter = ((tick * convergenceDebugLayers) + layer) * maxIter;
                for (int iter = 0; iter < totalIterations; iter++) {
                    int baseU = (baseIter + iter) * ConvergenceDebugIterBufSize;
                    if (baseU + 7 >= data.Length)
                        continue;

                    uint sumAbsC = data[baseU + 0];
                    uint maxAbsC = data[baseU + 1];
                    uint sumAbsDL = data[baseU + 2];
                    uint maxAbsDL = data[baseU + 3];
                    uint cnt = data[baseU + 4];
                    uint marker = data[baseU + 7];

                    float avgAbsC = cnt > 0 ? (sumAbsC * invScaleC) / cnt : 0f;
                    float mxAbsC = maxAbsC * invScaleC;
                    float avgAbsDL = cnt > 0 ? (sumAbsDL * invScaleDL) / cnt : 0f;
                    float mxAbsDL = maxAbsDL * invScaleDL;

                    writer.Write(step.ToString(CultureInfo.InvariantCulture));
                    writer.Write(',');
                    writer.Write(iter.ToString(CultureInfo.InvariantCulture));
                    writer.Write(',');
                    writer.Write(marker.ToString(CultureInfo.InvariantCulture));
                    writer.Write(',');
                    writer.Write(cnt.ToString(CultureInfo.InvariantCulture));
                    writer.Write(',');
                    writer.Write(avgAbsC.ToString("G9", CultureInfo.InvariantCulture));
                    writer.Write(',');
                    writer.Write(mxAbsC.ToString("G9", CultureInfo.InvariantCulture));
                    writer.Write(',');
                    writer.Write(avgAbsDL.ToString("G9", CultureInfo.InvariantCulture));
                    writer.Write(',');
                    writer.WriteLine(mxAbsDL.ToString("G9", CultureInfo.InvariantCulture));
                }
            }
        }

        void ClearDebugBuffer(int layer, int iterations, int tickIndex) {
            asyncCb.SetComputeIntParam(shader, "_ConvergenceDebugEnable", 1);
            asyncCb.SetComputeFloatParam(shader, "_ConvergenceDebugScaleC", ConvergenceDebugScaleC);
            asyncCb.SetComputeFloatParam(shader, "_ConvergenceDebugScaleDLambda", ConvergenceDebugScaleDLambda);
            asyncCb.SetComputeIntParam(shader, "_ConvergenceDebugOffset", ((tickIndex * convergenceDebugLayers) + layer) * convergenceDebugMaxIter);
            asyncCb.SetComputeIntParam(shader, "_ConvergenceDebugIterCount", iterations);

            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_ConvergenceDebug", convergenceDebug);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_ConvergenceDebug", convergenceDebug);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_ConvergenceDebug", convergenceDebug);
            asyncCb.SetComputeBufferParam(shader, kClearConvergenceDebugStats, "_ConvergenceDebug", convergenceDebug);

            Dispatch("XPBI.ClearConvergenceDebugStats", shader, kClearConvergenceDebugStats, (iterations + 255) / 256, 1, 1);
        }

        private void RequestColoringConflictHistoryDebugLog(int maxSolveLayer) {
            for (int layer = maxSolveLayer; layer >= 0; layer--) {
                bool[] coloringUpdatedByLayer = coloringUpdatedByLayerForDebugLog;
                if (coloringUpdatedByLayer == null || layer < 0 || layer >= coloringUpdatedByLayer.Length || !coloringUpdatedByLayer[layer])
                    continue;

                ulong key = 0xFFFFFFFF00000000UL | (uint)layer;
                if (!coloringByMeshLayer.TryGetValue(key, out var coloring) || coloring == null)
                    continue;

                int recordedIterations = coloring.GetRecordedConflictIterationCount();
                if (recordedIterations <= 0)
                    continue;

                int capturedLayer = layer;
                coloring.ReadConflictHistoryAsync(conflicts => {
                    if (conflicts == null || conflicts.Length == 0) {
                        Debug.LogError($"Coloring conflicts per iteration L{capturedLayer}: unavailable");
                        return;
                    }

                    var line = new System.Text.StringBuilder();
                    line.Append("Coloring conflicts per iteration L")
                        .Append(capturedLayer)
                        .Append(": ");

                    for (int i = 0; i < conflicts.Length; i++) {
                        if (i > 0)
                            line.Append(", ");
                        line.Append("i").Append(i).Append("=").Append(conflicts[i]);
                    }

                    Debug.LogError(line.ToString());
                });
            }
        }
    }
}