using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace GPU.Solver {
    internal sealed partial class SolverDebug {
        private const int ConvergenceDebugIterBufSize = 8;
        private const float ConvergenceDebugScaleC = 1_000_000f;
        private const float ConvergenceDebugScaleDLambda = 1_000_000f;

        internal ComputeBuffer convergenceDebug;
        internal ComputeBuffer convergenceDebugFallback;
        internal ComputeBuffer prolongationConstraintDebug;
        private uint[] convergenceDebugCpu;
        private uint[] prolongationConstraintDebugCpu;
        private bool[] coloringUpdatedByLayerForDebugLog;
        private int convergenceDebugStepCursor;
        private string convergenceDebugCsvPath;
        internal int convergenceDebugRequiredUInts;
        internal int convergenceDebugMaxIter;
        private int convergenceDebugLayers;
        private int convergenceDebugTicks;
        internal int prolongationConstraintDebugEntries;

        internal int kClearConvergenceDebugStats;

        internal ComputeBuffer ConvergenceDebug => convergenceDebug;
        internal ComputeBuffer ConvergenceDebugFallback => convergenceDebugFallback;
        internal ComputeBuffer ProlongationConstraintDebug => prolongationConstraintDebug;
        internal int ConvergenceDebugRequiredUInts => convergenceDebugRequiredUInts;
        internal int ConvergenceDebugMaxIter => convergenceDebugMaxIter;
        internal int ConvergenceDebugLayers => convergenceDebugLayers;
        internal int ProlongationConstraintDebugEntries => prolongationConstraintDebugEntries;
        internal int ClearConvergenceDebugStatsKernel => kClearConvergenceDebugStats;

        private static void Dispatch(CommandBuffer cb, string marker, ComputeShader shader, int kernel, int groupsX, int groupsY, int groupsZ) {
            cb.BeginSample(marker);
            cb.DispatchCompute(shader, kernel, groupsX, groupsY, groupsZ);
            cb.EndSample(marker);
        }

        internal void CaptureCurrentConstraintErrorSample(CommandBuffer cb, int entryIndex, int originalActiveCount, int probeActiveCount, bool restoreOwnerFilter) {
            if (entryIndex < 0 || probeActiveCount <= 0 || ProlongationConstraintDebug == null)
                return;

            ComputeBuffer restoredConvergenceDebug = ConvergenceDebug ?? ConvergenceDebugFallback ?? ProlongationConstraintDebug;

            cb.SetComputeIntParam(shader, "_ConvergenceDebugEnable", 1);
            cb.SetComputeFloatParam(shader, "_ConvergenceDebugScaleC", ConvergenceDebugScaleC);
            cb.SetComputeFloatParam(shader, "_ConvergenceDebugScaleDLambda", ConvergenceDebugScaleDLambda);
            cb.SetComputeIntParam(shader, "_ConvergenceDebugOffset", entryIndex);
            cb.SetComputeIntParam(shader, "_ConvergenceDebugIter", 0);
            cb.SetComputeIntParam(shader, "_ConvergenceDebugIterCount", 1);
            cb.SetComputeIntParam(shader, "_ActiveCount", probeActiveCount);
            cb.SetComputeIntParam(shader, "_UseDtOwnerFilter", 0);
            cb.SetComputeBufferParam(shader, solver.LayerSolveRuntime.KJRComputeDeltas, "_ConvergenceDebug", ProlongationConstraintDebug);
            cb.SetComputeBufferParam(shader, ClearConvergenceDebugStatsKernel, "_ConvergenceDebug", ProlongationConstraintDebug);

            Dispatch(cb, "XPBI.ClearConvergenceDebugStats.ProlongationProbe", shader, ClearConvergenceDebugStatsKernel, 1, 1, 1);
            Dispatch(cb, "XPBI.JR.SavePrevAndClear.ProlongationProbe", shader, solver.LayerSolveRuntime.KJRSavePrevAndClear, XPBISolver.Groups256(probeActiveCount), 1, 1);
            Dispatch(cb, "XPBI.JR.ComputeDeltas.ProlongationProbe", shader, solver.LayerSolveRuntime.KJRComputeDeltas, XPBISolver.Groups256(probeActiveCount), 1, 1);

            cb.SetComputeIntParam(shader, "_ActiveCount", originalActiveCount);
            cb.SetComputeIntParam(shader, "_UseDtOwnerFilter", restoreOwnerFilter ? 1 : 0);
            cb.SetComputeIntParam(shader, "_ConvergenceDebugEnable", 0);

            cb.SetComputeBufferParam(shader, solver.LayerSolveRuntime.KRelaxColored, "_ConvergenceDebug", restoredConvergenceDebug);
            cb.SetComputeBufferParam(shader, solver.LayerSolveRuntime.KRelaxColoredPersistentCoarse, "_ConvergenceDebug", restoredConvergenceDebug);
            cb.SetComputeBufferParam(shader, solver.LayerSolveRuntime.KJRComputeDeltas, "_ConvergenceDebug", restoredConvergenceDebug);
            cb.SetComputeBufferParam(shader, ClearConvergenceDebugStatsKernel, "_ConvergenceDebug", restoredConvergenceDebug);
        }

        internal void ClearDebugBuffer(CommandBuffer cb, int layer, int iterations, int tickIndex) {
            cb.SetComputeIntParam(shader, "_ConvergenceDebugEnable", 1);
            cb.SetComputeFloatParam(shader, "_ConvergenceDebugScaleC", ConvergenceDebugScaleC);
            cb.SetComputeFloatParam(shader, "_ConvergenceDebugScaleDLambda", ConvergenceDebugScaleDLambda);
            cb.SetComputeIntParam(shader, "_ConvergenceDebugOffset", ((tickIndex * ConvergenceDebugLayers) + layer) * ConvergenceDebugMaxIter);
            cb.SetComputeIntParam(shader, "_ConvergenceDebugIterCount", iterations);

            cb.SetComputeBufferParam(shader, solver.LayerSolveRuntime.KRelaxColored, "_ConvergenceDebug", ConvergenceDebug);
            cb.SetComputeBufferParam(shader, solver.LayerSolveRuntime.KRelaxColoredPersistentCoarse, "_ConvergenceDebug", ConvergenceDebug);
            cb.SetComputeBufferParam(shader, solver.LayerSolveRuntime.KJRComputeDeltas, "_ConvergenceDebug", ConvergenceDebug);
            cb.SetComputeBufferParam(shader, ClearConvergenceDebugStatsKernel, "_ConvergenceDebug", ConvergenceDebug);

            Dispatch(cb, "XPBI.ClearConvergenceDebugStats", shader, ClearConvergenceDebugStatsKernel, (iterations + 255) / 256, 1, 1);
        }

        internal void AllocateRuntimeBuffers() {
            convergenceDebugFallback = new ComputeBuffer(ConvergenceDebugIterBufSize, sizeof(uint), ComputeBufferType.Structured);
            convergenceDebugFallback.SetData(new uint[ConvergenceDebugIterBufSize]);
        }

        internal void ReleaseRuntimeBuffers() {
            convergenceDebug?.Dispose(); convergenceDebug = null;
            convergenceDebugFallback?.Dispose(); convergenceDebugFallback = null;
            prolongationConstraintDebug?.Dispose(); prolongationConstraintDebug = null;
            convergenceDebugCpu = null;
            prolongationConstraintDebugCpu = null;
            convergenceDebugRequiredUInts = 0;
            convergenceDebugMaxIter = 0;
            convergenceDebugLayers = 0;
            convergenceDebugTicks = 0;
            prolongationConstraintDebugEntries = 0;
            coloringUpdatedByLayerForDebugLog = null;
            convergenceDebugStepCursor = 0;
            convergenceDebugCsvPath = null;
        }

        internal void EnsureConvergenceDebugCapacity(int layers, int maxIter, int ticks) {
            int safeTicks = math.max(1, ticks);
            int requiredUInts = safeTicks * layers * maxIter * ConvergenceDebugIterBufSize;

            if (convergenceDebug != null &&
                convergenceDebug.IsValid() &&
                convergenceDebug.count == requiredUInts &&
                convergenceDebugMaxIter == maxIter &&
                convergenceDebugLayers == layers &&
                convergenceDebugTicks == safeTicks &&
                convergenceDebugCpu != null &&
                convergenceDebugCpu.Length == requiredUInts)
                return;

            convergenceDebug?.Dispose();

            convergenceDebug = new ComputeBuffer(requiredUInts, sizeof(uint), ComputeBufferType.Structured);
            convergenceDebugCpu = new uint[requiredUInts];
            convergenceDebugRequiredUInts = requiredUInts;
            convergenceDebugMaxIter = maxIter;
            convergenceDebugLayers = layers;
            convergenceDebugTicks = safeTicks;
        }

        internal void EnsureProlongationConstraintDebugCapacity(int entryCount) {
            int safeEntries = math.max(1, entryCount);
            int requiredUInts = safeEntries * ConvergenceDebugIterBufSize;

            if (prolongationConstraintDebug != null &&
                prolongationConstraintDebug.IsValid() &&
                prolongationConstraintDebugEntries == safeEntries &&
                prolongationConstraintDebug.count == requiredUInts &&
                prolongationConstraintDebugCpu != null &&
                prolongationConstraintDebugCpu.Length == requiredUInts)
                return;

            prolongationConstraintDebug?.Dispose();

            prolongationConstraintDebug = new ComputeBuffer(requiredUInts, sizeof(uint), ComputeBufferType.Structured);
            prolongationConstraintDebugCpu = new uint[requiredUInts];
            prolongationConstraintDebugEntries = safeEntries;
        }

        internal void CacheRuntimeKernels() {
            kClearConvergenceDebugStats = shader.FindKernel("ClearConvergenceDebugStats");
        }
    }
}
