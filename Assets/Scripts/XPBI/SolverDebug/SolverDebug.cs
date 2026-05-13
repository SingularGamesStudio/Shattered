using System.Globalization;
using System.IO;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;
using ProlongationConstraintProbe = GPU.Solver.XPBISolver.ProlongationConstraintProbe;
using ProlongateDebugSample = GPU.Solver.XPBISolver.ProlongateDebugSample;
using VelDebugSample = GPU.Solver.XPBISolver.VelDebugSample;
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

            if (session.EnableProlongateDebug && session.MaxProlongateDebugEntries > 0)
                EnsureProlongateDebugCapacity(session.MaxProlongateDebugEntries);

            if (session.EnableVelDebug && session.MaxVelDebugEntries > 0)
                EnsureVelDebugCapacity(session.MaxVelDebugEntries);

            session.ColoringUpdatedByLayer = SimulationParamSource.Current.uiAndReadback.convergenceDebugEnabled ? new bool[session.ConvergenceDebugLayerCount] : null;
            session.ProlongationConstraintProbes = session.MaxProlongationProbeSamples > 0
                ? new List<ProlongationConstraintProbe>(Mathf.Max(1, session.MaxProlongationProbeSamples / 2))
                : null;
            session.ProlongationProbeCursor = 0;

            session.ProlongateDebugSamples = session.MaxProlongateDebugEntries > 0
                ? new List<ProlongateDebugSample>(Mathf.Max(1, session.MaxProlongateDebugEntries / 2))
                : null;
            session.ProlongateDebugCursor = 0;

            session.VelDebugSamples = session.MaxVelDebugEntries > 0
                ? new List<VelDebugSample>(Mathf.Max(1, session.MaxVelDebugEntries / 2))
                : null;
            session.VelDebugCursor = 0;
        }

        /// <summary>
        /// Records blocking waits (when required) and async GPU readbacks for debug output.
        /// </summary>
        public void RecordReadbacksAndFence(SolveSession session, GraphicsFence fence) {
            bool hasProlongationProbeData =
                session.EnableProlongationConstraintProbeDebug &&
                session.ProlongationConstraintProbes != null &&
                session.ProlongationConstraintProbes.Count > 0;

            bool hasProlongateDebugData =
                session.EnableProlongateDebug &&
                session.ProlongateDebugSamples != null &&
                session.ProlongateDebugSamples.Count > 0;

            bool hasVelDebugData =
                session.EnableVelDebug &&
                session.VelDebugSamples != null &&
                session.VelDebugSamples.Count > 0;

            if (hasProlongationProbeData || hasProlongateDebugData || hasVelDebugData)
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

            /*if (hasProlongateDebugData && prolongateDebug != null && prolongateDebugCpu != null) {
                int required = Mathf.Min(prolongateDebug.count, prolongateDebugCpu.Length);
                prolongateDebug.GetData(prolongateDebugCpu, 0, 0, required);
                LogProlongateDebugFromData(prolongateDebugCpu, session.ProlongateDebugSamples);
            }

            if (hasVelDebugData && velDebug != null && velDebugCpu != null) {
                int required = Mathf.Min(velDebug.count, velDebugCpu.Length);
                velDebug.GetData(velDebugCpu, 0, 0, required);
                LogVelDebugFromData(velDebugCpu, session.VelDebugSamples);
            }*/

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

        internal void LogProlongateDebugFromData(uint[] data, IReadOnlyList<ProlongateDebugSample> samples) {
            if (data == null || samples == null || samples.Count == 0)
                return;

            for (int i = 0; i < samples.Count; i++) {
                var sample = samples[i];
                int baseU = sample.Entry * ProlongateDebugStride;
                if (baseU < 0 || baseU + ProlongateDebugStride > data.Length)
                    continue;

                uint total = data[baseU + 0];
                uint earlyFine = data[baseU + 1];
                uint giInvalid = data[baseU + 2];
                uint fixedVertex = data[baseU + 3];
                uint liActive = data[baseU + 4];
                uint noWeight = data[baseU + 5];
                uint weightZero = data[baseU + 6];
                uint applied = data[baseU + 7];
                uint parentOutOfRange = data[baseU + 8];
                uint parentWeightZero = data[baseU + 9];
                uint parentUsedSum = data[baseU + 10];
                uint parentFixed = data[baseU + 11];
                uint parentRpLenZero = data[baseU + 12];
                uint affineUsed = data[baseU + 13];
                uint rawWeightSumU = data[baseU + 14];
                uint rawWeightMaxU = data[baseU + 15];
                uint weightSumU = data[baseU + 16];
                uint weightMaxU = data[baseU + 17];
                uint blendDvSumU = data[baseU + 18];
                uint blendDvMaxU = data[baseU + 19];
                uint prolongDvSumU = data[baseU + 20];
                uint prolongDvMaxU = data[baseU + 21];
                uint parentUsedMax = data[baseU + 22];
                uint parentCountSum = data[baseU + 23];
                uint parentCountMax = data[baseU + 24];

                int weightCountInt = (int)total - (int)earlyFine - (int)giInvalid - (int)fixedVertex - (int)liActive - (int)noWeight;
                int dvCountInt = (int)applied;
                uint weightCount = (uint)Mathf.Max(0, weightCountInt);
                uint dvCount = (uint)Mathf.Max(0, dvCountInt);
                float invWeightScale = 1f / ProlongateDebugScaleWeight;
                float invDvScale = 1f / ProlongateDebugScaleDv;
                float rawWeightSum = rawWeightSumU * invWeightScale;
                float weightSum = weightSumU * invWeightScale;
                float blendDvSum = blendDvSumU * invDvScale;
                float prolongDvSum = prolongDvSumU * invDvScale;

                float rawWeightAvg = weightCount > 0 ? rawWeightSum / weightCount : 0f;
                float weightAvg = dvCount > 0 ? weightSum / dvCount : 0f;
                float blendDvAvg = dvCount > 0 ? blendDvSum / dvCount : 0f;
                float prolongDvAvg = dvCount > 0 ? prolongDvSum / dvCount : 0f;
                float parentUsedAvg = weightCount > 0 ? (float)parentUsedSum / weightCount : 0f;
                float parentCountAvg = weightCount > 0 ? (float)parentCountSum / weightCount : 0f;
                Debug.Log(prolongDvAvg+" "+weightAvg+" "+blendDvAvg);
                /*Debug.Log(
                    $"Prolongate debug T{sample.Tick} L{sample.Layer} active={sample.ActiveCount} fine={sample.FineCount} " +
                    $"threads={total} applied={applied} early(li>=fine|gi|fixed|li<active|noWeight|wSum0)=" +
                    $"{earlyFine}|{giInvalid}|{fixedVertex}|{liActive}|{noWeight}|{weightZero} " +
                    $"parents(avg|max used/count)={parentUsedAvg:G3}|{parentUsedMax}/{parentCountAvg:G3}|{parentCountMax} " +
                    $"rawW(avg|max)={rawWeightAvg:G6}|{rawWeightMaxU * invWeightScale:G6} " +
                    $"wSum(avg|max)={weightAvg:G6}|{weightMaxU * invWeightScale:G6} " +
                    $"blendDv(avg|max)={blendDvAvg:G6}|{blendDvMaxU * invDvScale:G6} " +
                    $"prolongDv(avg|max)={prolongDvAvg:G6}|{prolongDvMaxU * invDvScale:G6} " +
                    $"parentOut={parentOutOfRange} parentW0={parentWeightZero} fixedParent={parentFixed} rp0={parentRpLenZero} affine={affineUsed}");*/
            }
        }

        internal void LogVelDebugFromData(uint[] data, IReadOnlyList<VelDebugSample> samples) {
            if (data == null || samples == null || samples.Count == 0)
                return;

            for (int i = 0; i < samples.Count; i++) {
                var sample = samples[i];
                int baseU = sample.Entry * VelDebugStride;
                if (baseU < 0 || baseU + VelDebugStride > data.Length)
                    continue;

                uint sumVelU = data[baseU + 0];
                uint maxVelU = data[baseU + 1];
                uint sumSavedU = data[baseU + 2];
                uint maxSavedU = data[baseU + 3];
                uint count = data[baseU + 4];
                uint velZero = data[baseU + 5];
                uint savedZero = data[baseU + 6];

                float invScale = 1f / VelDebugScale;
                float sumVel = sumVelU * invScale;
                float sumSaved = sumSavedU * invScale;
                float avgVel = count > 0 ? sumVel / count : 0f;
                float avgSaved = count > 0 ? sumSaved / count : 0f;
                float maxVel = maxVelU * invScale;
                float maxSaved = maxSavedU * invScale;

                string stage = sample.Stage switch {
                    (int)VelDebugStage.AfterRelax => "AfterRelax",
                    (int)VelDebugStage.AfterCopyVelToPrev => "AfterCopyVelToPrev",
                    (int)VelDebugStage.AfterApplyXsph => "AfterApplyXsph",
                    _ => "Unknown",
                };

                Debug.Log(
                    $"VelDebug T{sample.Tick} L{sample.Layer} {stage} active={sample.ActiveCount} " +
                    $"avgVel={avgVel:G6} maxVel={maxVel:G6} avgSaved={avgSaved:G6} maxSaved={maxSaved:G6} " +
                    $"zeroVel={velZero} zeroSaved={savedZero}");
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
