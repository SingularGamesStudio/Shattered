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
        private DTColoring[] coloringPerLayer;

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

        private void EnsurePerLayerStateCapacity(int maxLayer) {
            int required = maxLayer + 1;

            if (coloringPerLayer == null || coloringPerLayer.Length < required)
                Array.Resize(ref coloringPerLayer, required);

            // relaxArgsByLayer is defined in another partial; ensure it is safe to index here.
            if (relaxArgsByLayer == null || relaxArgsByLayer.Length < required)
                Array.Resize(ref relaxArgsByLayer, required);
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
        /// <param name="useHierarchical">Whether to use hierarchical (multi-layer) solve.</param>
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

            int maxSolveLayer = (useHierarchical && m.layerEndIndex != null) ? m.maxLayer : 0;
            EnsurePerLayerStateCapacity(maxSolveLayer);
            EnsureConvergenceDebugCapacity(maxSolveLayer + 1, GetMaxIterationsForSolve(maxSolveLayer));

            for (int tick = 0; tick < tickCount; tick++) {
                SetCommonShaderParams(dtPerTick, Const.Gravity, Const.Compliance, total);

                // 1) Build parent relationships on first tick (TODO: each tick?).
                RebuildAllParents(m, total);

                // 2) Apply gameplay events + continuous external forces.
                ApplyForces(total, forceEventsCount);

                // 3) Solve constraints by V-cycle sweeps, each from coarse to fine.
                int vCycles = useHierarchical ? Mathf.Max(1, Const.HierarchyVCyclesPerTick) : 1;
                for (int cycle = 0; cycle < vCycles; cycle++) {
                    for (int layer = maxSolveLayer; layer >= 0; layer--) {
                        if (!m.TryGetLayerDt(layer, out DT dtLayer) || dtLayer == null)
                            continue;
                        if (m.NodeCount(layer) < 3)
                            continue;

                        ProcessLayer(m, layer, dtLayer, total, tick, forceEventsCount, convergenceDebug, maxSolveLayer);
                    }
                }

                // 4) Integrate positions on the full set of nodes.
                PrepareIntegratePosParams();
                Dispatch(shader, kIntegratePositions, Groups256(total), 1, 1);

                // 5) Push integrated positions back into DT, then run DT maintenance (fix/legalize).
                for (int layer = m.maxLayer; layer >= 0; layer--) {
                    if (!m.TryGetLayerDt(layer, out DT dtLayer) || dtLayer == null)
                        continue;

                    int activeCount = (layer > 0) ? m.NodeCount(layer) : total;
                    if (activeCount < 3)
                        continue;

                    PrepareUpdateDtPosParams(layer, dtLayer, activeCount, m, writeSlot);
                    Dispatch(shader, kUpdateDtPositions, Groups256(activeCount), 1, 1);

                    dtLayer.EnqueueMaintain(asyncCb, dtLayer.GetPositionsBuffer(writeSlot),
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
                LogConvergenceStatsFromData(convergenceDebugCpu, maxSolveLayer, convergenceDebugMaxIter);
            }

            return fence;
        }


        /// <summary>
        /// Rebuilds parent relationships for all hierarchical layers where applicable.
        /// </summary>
        private void RebuildAllParents(Meshless m, int total) {
            for (int layer = m.maxLayer; layer >= 1; layer--) {
                if (!m.TryGetLayerDt(layer, out DT dtLayer) || dtLayer == null)
                    continue;

                int activeCount = m.NodeCount(layer);
                int fineCount = layer > 1 ? m.NodeCount(layer - 1) : total;
                if (fineCount <= activeCount)
                    continue;

                PrepareParentRebuildBuffers(dtLayer, activeCount, fineCount);
                Dispatch(shader, kRebuildParentsAtLayer, Groups256(fineCount - activeCount), 1, 1);
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
        /// Runs the hierarchical/colored relaxation and inter-layer transfers for a single layer.
        /// </summary>
        private void ProcessLayer(
    Meshless m,
    int layer,
    DT dtLayer,
    int total,
    int tickIndex,
    int gameplayCountThisTick,
    ComputeBuffer debugBuffer,
    int maxSolveLayer
) {
            int activeCount = layer > 0 ? NodeCount(layer) : total;
            int fineCount = layer > 1 ? NodeCount(layer - 1) : total;

            PrepareRelaxBuffers(dtLayer, activeCount, fineCount, tickIndex);

            // 1) Rebuild hierarchical stats (CoarseFixed, volumes, fixed-child anchors) for this layer.
            Dispatch(shader, kClearHierarchicalStats, Groups256(activeCount), 1, 1);
            Dispatch(shader, kCacheHierarchicalStats, Groups256(fineCount), 1, 1);
            Dispatch(shader, kFinalizeHierarchicalStats, Groups256(activeCount), 1, 1);

            // 2) Snapshot prefix state.
            Dispatch(shader, kSaveVelPrefix, Groups256(activeCount), 1, 1);

            // 3) XPBI-aware coarse pre-correction via restricted residual + optional restricted gameplay events.
            bool useHierarchyTransfer = layer > 0 && fineCount > activeCount;
            bool injectRestrictedGameplay =
                useHierarchyTransfer &&
                gameplayCountThisTick > 0 &&
                Const.RestrictedDeltaVScale > 0f;
            bool injectRestrictedResidual =
                useHierarchyTransfer &&
                Const.UseResidualVCycle &&
                Const.RestrictResidualDeltaVScale > 0f;

            if (injectRestrictedResidual) {
                Dispatch(shader, kClearRestrictedDeltaV, Groups256(activeCount), 1, 1);
                Dispatch(shader, kRestrictFineVelocityResidualToActive, Groups256(fineCount - activeCount), 1, 1);
                asyncCb.SetComputeFloatParam(shader, "_RestrictedDeltaVScale", Const.RestrictResidualDeltaVScale);
                Dispatch(shader, kApplyRestrictedDeltaVToActiveAndPrefix, Groups256(activeCount), 1, 1);
            }

            if (injectRestrictedGameplay) {
                Dispatch(shader, kClearRestrictedDeltaV, Groups256(activeCount), 1, 1);
                asyncCb.SetComputeIntParam(shader, "_ForceEventCount", gameplayCountThisTick);
                asyncCb.SetComputeBufferParam(shader, kRestrictGameplayDeltaVFromEvents, "_ForceEvents", forceEvents);
                Dispatch(shader, kRestrictGameplayDeltaVFromEvents, Groups256(gameplayCountThisTick), 1, 1);
                asyncCb.SetComputeFloatParam(shader, "_RestrictedDeltaVScale", Const.RestrictedDeltaVScale);
                Dispatch(shader, kApplyRestrictedDeltaVToActiveAndPrefix, Groups256(activeCount), 1, 1);
            }

            if (!injectRestrictedGameplay) {
                asyncCb.SetComputeIntParam(shader, "_ForceEventCount", 0);
            }

            // 4) Cache per-node constants and initialize/clear per-iteration accumulators for this layer.
            Dispatch(shader, kCacheKernelH, Groups256(activeCount), 1, 1);
            Dispatch(shader, kComputeCorrectionL, Groups256(activeCount), 1, 1);
            Dispatch(shader, kCacheF0AndResetLambda, Groups256(activeCount), 1, 1);

            // 5) Rebuild coloring used by the colored Gauss-Seidel.
            RebuildColoringForLayer(layer, dtLayer, activeCount, dtLayer.NeighborCount);

            var coloring = coloringPerLayer[layer];
            if (coloring != null) {
                asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_ColorOrder", coloring.OrderBuffer);
                asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_ColorStarts", coloring.StartsBuffer);
                asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_ColorCounts", coloring.CountsBuffer);
            }

            int iterations = GetIterationsForLayer(layer, maxSolveLayer);

            // 6) Optional convergence debug (first tick only).
            bool dbg = debugBuffer != null && tickIndex == 0;
            if (dbg) {
                ClearDebugBuffer(layer, iterations);
            } else {
                asyncCb.SetComputeIntParam(shader, "_ConvergenceDebugEnable", 0);
            }

            // 7) Main solve: run multiple iterations; each iteration performs colored relaxation passes.
            for (int iter = 0; iter < iterations; iter++) {
                if (dbg)
                    asyncCb.SetComputeIntParam(shader, "_ConvergenceDebugIter", iter);

                for (int c = 0; c < 16; c++) {
                    asyncCb.SetComputeIntParam(shader, "_ColorIndex", c);
                    asyncCb.DispatchCompute(shader, kRelaxColored, relaxArgsByLayer[layer], (uint)c * 12);
                }
            }

            // 8) Transfer results across hierarchy and finalize per-layer outputs.
            if (layer > 0 && fineCount > activeCount) {
                Dispatch(shader, kProlongate, Groups256(fineCount - activeCount), 1, 1);
                if (Const.PostProlongSmoothing > 0f)
                    Dispatch(shader, kSmoothProlongatedFineVel, Groups256(fineCount - activeCount), 1, 1);
            }

            if (injectRestrictedGameplay || injectRestrictedResidual)
                Dispatch(shader, kRemoveRestrictedDeltaVFromActive, Groups256(activeCount), 1, 1);

            if (layer == 0)
                Dispatch(shader, kCommitDeformation, Groups256(activeCount), 1, 1);
        }

        private int GetIterationsForLayer(int layer, int maxSolveLayer) {
            int iterations = Const.IterationsLMid;
            if (layer == maxSolveLayer)
                iterations = Const.IterationsLMax;
            if (layer == 0)
                iterations = Const.IterationsL0;

            return Mathf.Max(1, iterations);
        }

        private int GetMaxIterationsForSolve(int maxSolveLayer) {
            int maxIterations = 1;
            for (int layer = 0; layer <= maxSolveLayer; layer++)
                maxIterations = Mathf.Max(maxIterations, GetIterationsForLayer(layer, maxSolveLayer));
            return maxIterations;
        }


        /// <summary>
        /// Ensures graph coloring exists and enqueues coloring maintenance/rebuild steps for this layer.
        /// </summary>
        private void RebuildColoringForLayer(int layer, DT dtLayer, int activeCount, int dtNeighborCount) {
            if (coloringShader == null) {
                Debug.LogError("XPBISolver: No coloring shader provided. Cannot rebuild coloring.");
                return;
            }

            if (coloringPerLayer == null || layer < 0 || layer >= coloringPerLayer.Length) {
                Debug.LogError("XPBISolver: coloringPerLayer is not sized. Cannot rebuild coloring.");
                return;
            }

            if (coloringPerLayer[layer] == null)
                coloringPerLayer[layer] = new DTColoring(coloringShader);

            var coloring = coloringPerLayer[layer];
            uint seed = 12345u + (uint)layer; // deterministic seed per layer

            if (coloring.ColorBuffer == null) {
                coloring.Init(activeCount, dtNeighborCount, seed);
                coloring.EnqueueInitTriGrid(asyncCb, pos, dtLayer, layerCellSizeCpu[layer]);
                dtLayer.MarkAllDirty(asyncCb);
            }

            coloring.EnqueueUpdateAfterMaintain(asyncCb, pos, dtLayer, layerCellSizeCpu[layer], Const.ColoringConflictRounds);
            coloring.EnqueueRebuildOrderAndArgs(asyncCb);

            relaxArgsByLayer[layer] = coloring.RelaxArgsBuffer;
        }

        private int NodeCount(int layer) {
            return meshless.NodeCount(layer);
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

            if (coloringPerLayer != null) {
                foreach (var coloring in coloringPerLayer)
                    coloring?.Dispose();
                coloringPerLayer = null;
            }

            kernelsCached = false;
        }
    }
}
