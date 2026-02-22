using System;
using GPU.Delaunay;
using Unity.Mathematics;
using Unity.VisualScripting.FullSerializer;
using Unity.Collections;
using UnityEngine;
using UnityEngine.Rendering;

namespace GPU.Solver {
    public sealed partial class XPBISolver : IDisposable {
        private int initializedCount = -1;
        private Meshless meshless;
        private CommandBuffer asyncCb;

        private readonly ComputeShader shader;
        private readonly ComputeShader coloringShader;
        private DTColoring[] coloringPerLevel;

        [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
        public struct ForceEvent {
            /// <summary>
            /// Node index to apply the force to.
            /// </summary>
            public uint node;

            /// <summary>
            /// Force vector in simulation space.
            /// </summary>
            public float2 force;
        }

        /// <summary>
        /// Creates a new solver instance.
        /// </summary>
        /// <param name="solverShader">Compute shader implementing the XPBI pipeline.</param>
        /// <param name="coloringShader">Compute shader used to build/maintain graph coloring for colored relaxation.</param>
        public XPBISolver(ComputeShader solverShader, ComputeShader coloringShader) {
            this.shader = solverShader;
            this.coloringShader = coloringShader;
        }

        private static int Groups256(int count) {
            return (count + 255) / 256;
        }

        private void EnsureAsyncCommandBufferForRecording() {
            if (asyncCb == null)
                asyncCb = new CommandBuffer { name = "XPBI Async Batch" };

            asyncCb.Clear();
            asyncCb.SetExecutionFlags(CommandBufferExecutionFlags.AsyncCompute);
        }

        private void EnsurePerLevelStateCapacity(int maxLevel) {
            int required = maxLevel + 1;

            if (coloringPerLevel == null || coloringPerLevel.Length < required)
                Array.Resize(ref coloringPerLevel, required);

            // relaxArgsByLevel is defined in another partial; ensure it is safe to index here.
            if (relaxArgsByLevel == null || relaxArgsByLevel.Length < required)
                Array.Resize(ref relaxArgsByLevel, required);
        }

        private void Dispatch(ComputeShader shader, int kernel, int x, int y, int z) {
            asyncCb.DispatchCompute(shader, kernel, x, y, z);
        }

        /// <summary>
        /// Uploads gameplay forces to the GPU for use in subsequent <see cref="SubmitSolve"/> calls.
        /// </summary>
        /// <param name="events">Source array of force events.</param>
        /// <param name="count">Number of valid elements in <paramref name="events"/>.</param>
        /// <remarks>
        /// Passing <c>count == 0</c> is allowed and results in no events being applied.
        /// </remarks>
        public void SetGameplayForces(ForceEvent[] events, int count) {
            if (events == null) throw new ArgumentNullException(nameof(events));
            if (count < 0 || count > events.Length) throw new ArgumentOutOfRangeException(nameof(count));

            forceEventsCount = count;
            if (forceEventsCount <= 0) return;

            EnsureForceEventCapacity(forceEventsCount);

            Array.Copy(events, 0, forceEventsCpu, 0, forceEventsCount);
            forceEvents.SetData(forceEventsCpu, 0, 0, forceEventsCount);
        }

        /// <summary>
        /// Clears gameplay forces (no force events will be applied until new ones are set).
        /// </summary>
        public void ClearGameplayForces() {
            forceEventsCount = 0;
        }

        /// <summary>
        /// Records and submits a GPU solve for the provided meshless system.
        /// </summary>
        /// <param name="m">Meshless system to solve.</param>
        /// <param name="dtPerTick">Time step per tick.</param>
        /// <param name="tickCount">Number of ticks to execute in this submission.</param>
        /// <param name="useHierarchical">Whether to use hierarchical (multi-level) solve.</param>
        /// <param name="queueType">Async compute queue type to submit to.</param>
        /// <param name="writeSlot">Destination slot for cycled buffers.</param>
        /// <returns>
        /// A <see cref="GraphicsFence"/> inserted after the command buffer workload.
        /// </returns>
        /// <remarks>
        /// If convergence debugging is enabled, this method performs a GPU readback which blocks the CPU until the GPU has completed the work.
        /// </remarks>
        public GraphicsFence SubmitSolve(
    Meshless m,
    float dtPerTick,
    int tickCount,
    bool useHierarchical,
    bool ConvergenceDebugEnabled,
    ComputeQueueType queueType,
    int writeSlot
) {
            EnsureKernelsCached();

            int total = m.nodes.Count;
            if (total == 0 || tickCount <= 0)
                return default;

            EnsureCapacity(total);

            if (initializedCount != total)
                InitializeFromMeshless(m);

            EnsureAsyncCommandBufferForRecording();

            int maxSolveLevel = (useHierarchical && m.levelEndIndex != null) ? m.maxLayer : 0;
            EnsurePerLevelStateCapacity(maxSolveLevel);
            EnsureConvergenceDebugCapacity(maxSolveLevel + 1, Const.IterationsL0);

            for (int tick = 0; tick < tickCount; tick++) {
                SetCommonShaderParams(dtPerTick, Const.Gravity, Const.Compliance, total);

                // 1) Build parent relationships on first tick (TODO: each tick?).
                if (tick == 0 && useHierarchical && m.maxLayer > 0)
                    RebuildAllParents(m, total);

                // 2) Apply gameplay events + continuous external forces.
                ApplyForces(total, forceEventsCount);

                // 3) Solve constraints level-by-level, from coarse to fine .
                for (int level = maxSolveLevel; level >= 0; level--) {
                    if (!m.TryGetLevelDt(level, out DT dtLevel) || dtLevel == null)
                        continue;
                    if (m.NodeCount(level) < 3)
                        continue;

                    ProcessLevel(m, level, dtLevel, total, tick, forceEventsCount, convergenceDebug);
                }

                // 4) Integrate positions on the full set of nodes.
                PrepareIntegratePosParams();
                Dispatch(shader, kIntegratePositions, Groups256(total), 1, 1);

                // 5) Push integrated positions back into DT, then run DT maintenance (fix/legalize).
                for (int level = m.maxLayer; level >= 0; level--) {
                    if (!m.TryGetLevelDt(level, out DT dtLevel) || dtLevel == null)
                        continue;

                    int activeCount = (level > 0) ? m.NodeCount(level) : total;
                    if (activeCount < 3)
                        continue;

                    PrepareUpdateDtPosParams(level, dtLevel, activeCount, m, writeSlot);
                    Dispatch(shader, kUpdateDtPositions, Groups256(activeCount), 1, 1);

                    dtLevel.EnqueueMaintain(asyncCb, dtLevel.GetPositionsBuffer(writeSlot),
                        writeSlot, Const.DTFixIterations, Const.DTLegalizeIterations);
                }
            }

            // 6) Insert a fence and submit to the selected async compute queue.
            var fence = asyncCb.CreateGraphicsFence(GraphicsFenceType.AsyncQueueSynchronisation,
                SynchronisationStageFlags.ComputeProcessing);

            Graphics.ExecuteCommandBufferAsync(asyncCb, queueType);

            // 7) Optional convergence debug readback (CPU-blocking).
            if (ConvergenceDebugEnabled && convergenceDebug != null && convergenceDebugCpu != null && convergenceDebugRequiredUInts > 0) {
                convergenceDebug.GetData(convergenceDebugCpu);
                LogConvergenceStatsFromData(convergenceDebugCpu, maxSolveLevel, convergenceDebugMaxIter);
            }

            return fence;
        }


        /// <summary>
        /// Rebuilds parent relationships for all hierarchical levels where applicable.
        /// </summary>
        private void RebuildAllParents(Meshless m, int total) {
            for (int level = m.maxLayer; level >= 1; level--) {
                if (!m.TryGetLevelDt(level, out DT dtLevel) || dtLevel == null)
                    continue;

                int activeCount = m.NodeCount(level);
                int fineCount = level > 1 ? m.NodeCount(level - 1) : total;
                if (fineCount <= activeCount)
                    continue;

                PrepareParentRebuildBuffers(dtLevel, activeCount, fineCount);
                Dispatch(shader, kRebuildParentsAtLevel, Groups256(fineCount - activeCount), 1, 1);
            }
        }

        /// <summary>
        /// Applies gameplay and external forces for this tick.
        /// </summary>
        private void ApplyForces(int total, int gameplayCountThisTick) {
            PrepareApplyForcesParams();

            if (gameplayCountThisTick > 0) {
                asyncCb.SetComputeIntParam(shader, "_ForceEventCount", gameplayCountThisTick);
                asyncCb.SetComputeBufferParam(shader, kApplyGameplayForces, "_ForceEvents", forceEvents);
                Dispatch(shader, kApplyGameplayForces, Groups256(gameplayCountThisTick), 1, 1);
            } else {
                asyncCb.SetComputeIntParam(shader, "_ForceEventCount", 0);
            }

            Dispatch(shader, kExternalForces, Groups256(total), 1, 1);
        }

        /// <summary>
        /// Runs the hierarchical/colored relaxation and inter-level transfers for a single level.
        /// </summary>
        private void ProcessLevel(
    Meshless m,
    int level,
    DT dtLevel,
    int total,
    int tickIndex,
    int gameplayCountThisTick,
    ComputeBuffer debugBuffer
) {
            int activeCount = level > 0 ? NodeCount(level) : total;
            int fineCount = level > 1 ? NodeCount(level - 1) : total;

            PrepareRelaxBuffers(dtLevel, activeCount, fineCount, tickIndex);

            // 1) Snapshot prefix state / clear and cache hierarchical statistics used for correction.
            Dispatch(shader, kSaveVelPrefix, Groups256(activeCount), 1, 1);

            Dispatch(shader, kClearHierarchicalStats, Groups256(activeCount), 1, 1);
            Dispatch(shader, kCacheHierarchicalStats, Groups256(total), 1, 1);

            // 2) Optionally inject restricted gameplay forces into upper levels.
            bool injectRestricted =
                level > 0 &&
                fineCount > activeCount &&
                gameplayCountThisTick > 0 &&
                Const.RestrictedDeltaVScale > 0f;

            if (injectRestricted) {
                asyncCb.SetComputeIntParam(shader, "_ForceEventCount", gameplayCountThisTick);
                asyncCb.SetComputeBufferParam(shader, kRestrictGameplayDeltaVFromEvents, "_ForceEvents", forceEvents);

                Dispatch(shader, kClearRestrictedDeltaV, Groups256(activeCount), 1, 1);
                Dispatch(shader, kRestrictGameplayDeltaVFromEvents, Groups256(gameplayCountThisTick), 1, 1);

                asyncCb.SetComputeFloatParam(shader, "_RestrictedDeltaVScale", Const.RestrictedDeltaVScale);
                Dispatch(shader, kApplyRestrictedDeltaVToActiveAndPrefix, Groups256(activeCount), 1, 1);
            } else {
                asyncCb.SetComputeIntParam(shader, "_ForceEventCount", 0);
            }

            // 3) Cache per-node constants and initialize/clear per-iteration accumulators for this level.
            Dispatch(shader, kCacheKernelH, Groups256(activeCount), 1, 1);
            Dispatch(shader, kComputeCorrectionL, Groups256(activeCount), 1, 1);
            Dispatch(shader, kCacheF0AndResetLambda, Groups256(activeCount), 1, 1);

            // 4) Rebuild coloring used by the colored Gauss-Seidel.
            RebuildColoringForLevel(level, dtLevel, activeCount, dtLevel.NeighborCount);

            var coloring = coloringPerLevel[level];
            if (coloring != null) {
                asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_ColorOrder", coloring.OrderBuffer);
                asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_ColorStarts", coloring.StartsBuffer);
                asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_ColorCounts", coloring.CountsBuffer);
            }

            int iterations = Const.IterationsLMid;
            if (level == m.maxLayer)
                iterations = Const.IterationsLMax;
            if (level == 0)
                iterations = Const.IterationsL0;

            // 5) Optional convergence debug (first tick only).
            bool dbg = debugBuffer != null && tickIndex == 0;
            if (dbg) {
                ClearDebugBuffer(level, iterations);
            } else {
                asyncCb.SetComputeIntParam(shader, "_ConvergenceDebugEnable", 0);
            }

            // 6) Main solve: run multiple iterations; each iteration performs colored relaxation passes.
            for (int iter = 0; iter < iterations; iter++) {
                if (dbg)
                    asyncCb.SetComputeIntParam(shader, "_ConvergenceDebugIter", iter);

                for (int c = 0; c < 16; c++) {
                    asyncCb.SetComputeIntParam(shader, "_ColorIndex", c);
                    asyncCb.DispatchCompute(shader, kRelaxColored, relaxArgsByLevel[level], (uint)c * 12);
                }
            }

            // 7) Transfer results across hierarchy and finalize per-level outputs.
            if (level > 0 && fineCount > activeCount)
                Dispatch(shader, kProlongate, Groups256(fineCount - activeCount), 1, 1);

            if (injectRestricted)
                Dispatch(shader, kRemoveRestrictedDeltaVFromActive, Groups256(activeCount), 1, 1);

            if (level == 0)
                Dispatch(shader, kCommitDeformation, Groups256(activeCount), 1, 1);
        }


        /// <summary>
        /// Ensures graph coloring exists and enqueues coloring maintenance/rebuild steps for this level.
        /// </summary>
        private void RebuildColoringForLevel(int level, DT dtLevel, int activeCount, int dtNeighborCount) {
            if (coloringShader == null) {
                Debug.LogError("XPBISolver: No coloring shader provided. Cannot rebuild coloring.");
                return;
            }

            if (coloringPerLevel == null || level < 0 || level >= coloringPerLevel.Length) {
                Debug.LogError("XPBISolver: coloringPerLevel is not sized. Cannot rebuild coloring.");
                return;
            }

            if (coloringPerLevel[level] == null)
                coloringPerLevel[level] = new DTColoring(coloringShader);

            var coloring = coloringPerLevel[level];
            uint seed = 12345u + (uint)level; // deterministic seed per level

            if (coloring.ColorBuffer == null) {
                coloring.Init(activeCount, dtNeighborCount, seed);
                coloring.EnqueueInitTriGrid(asyncCb, pos, dtLevel, levelCellSizeCpu[level]);
                dtLevel.MarkAllDirty(asyncCb);
            }

            coloring.EnqueueUpdateAfterMaintain(asyncCb, pos, dtLevel, levelCellSizeCpu[level], Const.ColoringConflictRounds);
            coloring.EnqueueRebuildOrderAndArgs(asyncCb);

            relaxArgsByLevel[level] = coloring.RelaxArgsBuffer;
        }

        private int NodeCount(int level) {
            return meshless.NodeCount(level);
        }

        /// <summary>
        /// Releases GPU buffers and other resources owned by this solver.
        /// </summary>
        public void Dispose() {
            Release();
        }

        private void Release() {
            ReleaseBuffers();
            capacity = 0;

            posCpu = null;
            velCpu = null;
            invMassCpu = null;
            flagsCpu = null;
            restVolumeCpu = null;
            parentIndexCpu = null;
            FCpu = null;
            FpCpu = null;

            forceEventsCpu = null;
            forceEventsCapacity = 0;
            forceEventsCount = 0;

            if (asyncCb != null) {
                asyncCb.Dispose();
                asyncCb = null;
            }

            if (coloringPerLevel != null) {
                foreach (var coloring in coloringPerLevel)
                    coloring?.Dispose();
                coloringPerLevel = null;
            }

            kernelsCached = false;
        }
    }
}
