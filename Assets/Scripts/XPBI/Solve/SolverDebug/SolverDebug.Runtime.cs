using Unity.Mathematics;
using UnityEngine;

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

        internal void CaptureCurrentConstraintErrorSample(int entryIndex, int originalActiveCount, int probeActiveCount, bool restoreOwnerFilter) {
            if (entryIndex < 0 || probeActiveCount <= 0 || ProlongationConstraintDebug == null)
                return;

            ComputeBuffer restoredConvergenceDebug = ConvergenceDebug ?? ConvergenceDebugFallback ?? ProlongationConstraintDebug;

            solver.asyncCb.SetComputeIntParam(solver.shader, "_ConvergenceDebugEnable", 1);
            solver.asyncCb.SetComputeFloatParam(solver.shader, "_ConvergenceDebugScaleC", ConvergenceDebugScaleC);
            solver.asyncCb.SetComputeFloatParam(solver.shader, "_ConvergenceDebugScaleDLambda", ConvergenceDebugScaleDLambda);
            solver.asyncCb.SetComputeIntParam(solver.shader, "_ConvergenceDebugOffset", entryIndex);
            solver.asyncCb.SetComputeIntParam(solver.shader, "_ConvergenceDebugIter", 0);
            solver.asyncCb.SetComputeIntParam(solver.shader, "_ConvergenceDebugIterCount", 1);
            solver.asyncCb.SetComputeIntParam(solver.shader, "_ActiveCount", probeActiveCount);
            solver.asyncCb.SetComputeIntParam(solver.shader, "_UseDtOwnerFilter", 0);
            solver.asyncCb.SetComputeBufferParam(solver.shader, solver.LayerSolve.KJRComputeDeltas, "_ConvergenceDebug", ProlongationConstraintDebug);
            solver.asyncCb.SetComputeBufferParam(solver.shader, ClearConvergenceDebugStatsKernel, "_ConvergenceDebug", ProlongationConstraintDebug);

            solver.Dispatch("XPBI.ClearConvergenceDebugStats.ProlongationProbe", solver.shader, ClearConvergenceDebugStatsKernel, 1, 1, 1);
            solver.Dispatch("XPBI.JR.SavePrevAndClear.ProlongationProbe", solver.shader, solver.LayerSolve.KJRSavePrevAndClear, XPBISolver.Groups256(probeActiveCount), 1, 1);
            solver.Dispatch("XPBI.JR.ComputeDeltas.ProlongationProbe", solver.shader, solver.LayerSolve.KJRComputeDeltas, XPBISolver.Groups256(probeActiveCount), 1, 1);

            solver.asyncCb.SetComputeIntParam(solver.shader, "_ActiveCount", originalActiveCount);
            solver.asyncCb.SetComputeIntParam(solver.shader, "_UseDtOwnerFilter", restoreOwnerFilter ? 1 : 0);
            solver.asyncCb.SetComputeIntParam(solver.shader, "_ConvergenceDebugEnable", 0);

            solver.asyncCb.SetComputeBufferParam(solver.shader, solver.LayerSolve.KRelaxColored, "_ConvergenceDebug", restoredConvergenceDebug);
            solver.asyncCb.SetComputeBufferParam(solver.shader, solver.LayerSolve.KRelaxColoredPersistentCoarse, "_ConvergenceDebug", restoredConvergenceDebug);
            solver.asyncCb.SetComputeBufferParam(solver.shader, solver.LayerSolve.KJRComputeDeltas, "_ConvergenceDebug", restoredConvergenceDebug);
            solver.asyncCb.SetComputeBufferParam(solver.shader, ClearConvergenceDebugStatsKernel, "_ConvergenceDebug", restoredConvergenceDebug);
        }

        internal void ClearDebugBuffer(int layer, int iterations, int tickIndex) {
            solver.asyncCb.SetComputeIntParam(solver.shader, "_ConvergenceDebugEnable", 1);
            solver.asyncCb.SetComputeFloatParam(solver.shader, "_ConvergenceDebugScaleC", ConvergenceDebugScaleC);
            solver.asyncCb.SetComputeFloatParam(solver.shader, "_ConvergenceDebugScaleDLambda", ConvergenceDebugScaleDLambda);
            solver.asyncCb.SetComputeIntParam(solver.shader, "_ConvergenceDebugOffset", ((tickIndex * ConvergenceDebugLayers) + layer) * ConvergenceDebugMaxIter);
            solver.asyncCb.SetComputeIntParam(solver.shader, "_ConvergenceDebugIterCount", iterations);

            solver.asyncCb.SetComputeBufferParam(solver.shader, solver.LayerSolve.KRelaxColored, "_ConvergenceDebug", ConvergenceDebug);
            solver.asyncCb.SetComputeBufferParam(solver.shader, solver.LayerSolve.KRelaxColoredPersistentCoarse, "_ConvergenceDebug", ConvergenceDebug);
            solver.asyncCb.SetComputeBufferParam(solver.shader, solver.LayerSolve.KJRComputeDeltas, "_ConvergenceDebug", ConvergenceDebug);
            solver.asyncCb.SetComputeBufferParam(solver.shader, ClearConvergenceDebugStatsKernel, "_ConvergenceDebug", ConvergenceDebug);

            solver.Dispatch("XPBI.ClearConvergenceDebugStats", solver.shader, ClearConvergenceDebugStatsKernel, (iterations + 255) / 256, 1, 1);
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
            kClearConvergenceDebugStats = solver.shader.FindKernel("ClearConvergenceDebugStats");
        }
    }
}
