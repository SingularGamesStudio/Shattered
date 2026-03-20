using System.Globalization;
using System.IO;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using ProlongationConstraintProbe = GPU.Solver.XPBISolver.ProlongationConstraintProbe;
using SolveSession = GPU.Solver.XPBISolver.SolveSession;

namespace GPU.Solver {
    internal sealed partial class SolverDebug {
        private readonly XPBISolver solver;
        private readonly ComputeShader shader;

        public SolverDebug(XPBISolver solver) {
            this.solver = solver;
            shader = solver.SolverDebugShader;
        }

        /// <summary>
        /// Prepares debug buffers and per-session debug carriers prior to command recording.
        /// </summary>
        public void PrepareSession(SolveSession session) {
            EnsureConvergenceDebugCapacity(session.ConvergenceDebugLayerCount, session.ConvergenceDebugMaxIterations, session.Request.TickCount);

            if (session.EnableProlongationConstraintProbeDebug && session.MaxProlongationProbeSamples > 0)
                EnsureProlongationConstraintDebugCapacity(session.MaxProlongationProbeSamples);

            session.ColoringUpdatedByLayer = SimulationParamSource.Current.uiAndReadback.convergenceDebugEnabled ? new bool[session.ConvergenceDebugLayerCount] : null;
            session.ProlongationConstraintProbes = session.MaxProlongationProbeSamples > 0
                ? new List<ProlongationConstraintProbe>(Mathf.Max(1, session.MaxProlongationProbeSamples / 2))
                : null;
            session.ProlongationProbeCursor = 0;
        }

        /// <summary>
        /// Records blocking waits (when required) and async GPU readbacks for debug output.
        /// </summary>
        public void RecordReadbacksAndFence(SolveSession session, GraphicsFence fence) {
            bool hasProlongationProbeData =
                session.EnableProlongationConstraintProbeDebug &&
                session.ProlongationConstraintProbes != null &&
                session.ProlongationConstraintProbes.Count > 0;

            if (hasProlongationProbeData)
                Graphics.WaitOnAsyncGraphicsFence(fence);

            if (hasProlongationProbeData && prolongationConstraintDebug != null) {
                var capturedProbes = new List<ProlongationConstraintProbe>(session.ProlongationConstraintProbes);
                AsyncGPUReadback.Request(prolongationConstraintDebug, req => {
                    if (req.hasError)
                        return;

                    var data = req.GetData<uint>();
                    LogProlongationConstraintStatsFromData(data.ToArray(), capturedProbes);
                });
            }

            if (SimulationParamSource.Current.uiAndReadback.convergenceDebugEnabled && convergenceDebug != null && convergenceDebugRequiredUInts > 0) {
                Graphics.WaitOnAsyncGraphicsFence(fence);

                int dbgMaxLayer = session.ConvergenceDebugMaxLayer;
                int dbgMaxIter = convergenceDebugMaxIter;
                int dbgTickCount = session.Request.TickCount;
                int dbgBaseStep = convergenceDebugStepCursor;
                coloringUpdatedByLayerForDebugLog = session.ColoringUpdatedByLayer;
                convergenceDebugStepCursor += session.Request.TickCount;

                AsyncGPUReadback.Request(convergenceDebug, req => {
                    if (req.hasError)
                        return;

                    var data = req.GetData<uint>();
                    LogConvergenceStatsFromData(data.ToArray(), dbgMaxLayer, dbgMaxIter, dbgTickCount, dbgBaseStep);
                });

                RequestColoringConflictHistoryDebugLog(dbgMaxLayer);
            }

            coloringUpdatedByLayerForDebugLog = null;
        }

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

        internal void LogProlongationConstraintStatsFromData(uint[] data, IReadOnlyList<ProlongationConstraintProbe> probes) {
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

        internal void LogConvergenceStatsFromData(uint[] data, int maxSolveLayer, int maxIter, int tickCount, int baseStep) {
            if (data == null || data.Length == 0)
                return;

            float invScaleC = 1f / ConvergenceDebugScaleC;
            float invScaleDL = 1f / ConvergenceDebugScaleDLambda;

            AppendLayer0ConvergenceCsv(data, maxSolveLayer, maxIter, tickCount, baseStep, invScaleC, invScaleDL);

            int logTick = Mathf.Clamp(tickCount - 1, 0, int.MaxValue);

            for (int layer = maxSolveLayer; layer >= 0; layer--) {
                int jrIterations = GetJRIterationsForLayer(layer, maxSolveLayer);
                int baseIter = ((logTick * ConvergenceDebugLayers) + layer) * maxIter;
                int gsIterations = layer == 0 ? Const.GSIterationsL0 : 1;

                var table = new System.Text.StringBuilder();
                table.AppendLine($"Layer {layer} convergence stats:");
                table.AppendLine("Iter | Marker | Count | Avg|C|       | Max|C|       | Avg|dLambda| | Max|dLambda|");
                table.AppendLine("-----|--------|-------|--------------|--------------|---------------|---------------");

                for (int iter = 0; iter < jrIterations + gsIterations; iter++) {
                    if (iter == gsIterations)
                        table.AppendLine("-----|--------|-------|--------------|--------------|---------------|---------------");

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
                int baseIter = ((tick * ConvergenceDebugLayers) + layer) * maxIter;
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

        private static int GetJRIterationsForLayer(int layer, int maxSolveLayer) {
            int iterations = Const.JRIterationsLMid;
            if (layer == maxSolveLayer)
                iterations = Const.JRIterationsLMax;
            if (layer == 0)
                iterations = Const.JRIterationsL0;

            return Mathf.Max(1, iterations);
        }

        internal void RequestColoringConflictHistoryDebugLog(int maxSolveLayer) {
            solver.coloring.RequestConflictHistoryDebugLog(maxSolveLayer, coloringUpdatedByLayerForDebugLog);
        }

    }
}
