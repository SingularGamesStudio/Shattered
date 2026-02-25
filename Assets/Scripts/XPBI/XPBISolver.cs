using System;
using System.Collections.Generic;
using GPU.Delaunay;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace GPU.Solver {
    public sealed partial class XPBISolver : IDisposable {
        private int initializedCount = -1;
        private bool layoutInitialized;
        private CommandBuffer asyncCb;

        private struct MeshRange {
            public Meshless meshless;
            public int baseIndex;
            public int totalCount;
        }

        private readonly List<MeshRange> solveRanges = new List<MeshRange>(64);
        private readonly Dictionary<ulong, DTColoring> coloringByMeshLayer = new Dictionary<ulong, DTColoring>(128);

        private readonly ComputeShader shader;
        private readonly ComputeShader coloringShader;

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
        /// Records and submits one global solve for all active meshless systems.
        /// </summary>
        public GraphicsFence SubmitSolve(
    IReadOnlyList<Meshless> meshes,
    float dtPerTick,
    int tickCount,
    bool useHierarchical,
    bool ConvergenceDebugEnabled,
    ComputeQueueType queueType,
    int readSlot,
    int writeSlot,
    GlobalDTHierarchy globalDTHierarchy
) {
            EnsureKernelsCached();

            if (tickCount <= 0)
                return default;

            if (globalDTHierarchy == null || globalDTHierarchy.MaxLayer < 0)
                return default;

            int totalCount;
            int maxSolveLayer;
            if (!EnsureLayout(meshes, out totalCount, out maxSolveLayer))
                return default;

            maxSolveLayer = Mathf.Max(maxSolveLayer, globalDTHierarchy.MaxLayer);

            EnsureCapacity(totalCount);

            if (!layoutInitialized)
                InitializeFromMeshless((System.Collections.Generic.List<MeshRange>)solveRanges, totalCount);

            EnsureAsyncCommandBufferForRecording();
            EnsureConvergenceDebugCapacity(maxSolveLayer + 1, GetMaxIterationsForSolve(maxSolveLayer));

            for (int tick = 0; tick < tickCount; tick++) {
                SetCommonShaderParams(dtPerTick, Const.Gravity, Const.Compliance, totalCount, 0);

                for (int layer = globalDTHierarchy.MaxLayer; layer >= 1; layer--) {
                    if (!globalDTHierarchy.TryGetLayerDt(layer, out DT dtLayer) || dtLayer == null)
                        continue;

                    if (!globalDTHierarchy.TryGetLayerMappings(layer, out _, out _, out int[] globalFineNodeByLocal, out int activeCount, out int fineCount))
                        continue;
                    if (fineCount <= activeCount)
                        continue;

                    bool useMappedIndices = !IsIdentityMapping(globalFineNodeByLocal, fineCount);
                    if (useMappedIndices) {
                        ComputeBuffer globalNodeMap = EnsureGlobalLayerNodeMapBuffer(layer, globalFineNodeByLocal, fineCount);
                        ComputeBuffer globalToLocalMap = EnsureGlobalLayerGlobalToLocalBufferCached(layer, globalFineNodeByLocal, fineCount, totalCount);
                        PrepareParentRebuildBuffers(dtLayer, 0, activeCount, fineCount, true, 0, globalNodeMap, globalToLocalMap);
                    } else {
                        PrepareParentRebuildBuffers(dtLayer, 0, activeCount, fineCount, false, 0, null, null);
                    }
                    Dispatch(shader, kRebuildParentsAtLayer, Groups256(fineCount - activeCount), 1, 1);
                }

                ApplyForces(totalCount, forceEventsCount);

                int vCycles = useHierarchical ? Mathf.Max(1, Const.HierarchyVCyclesPerTick) : 1;
                for (int cycle = 0; cycle < vCycles; cycle++) {
                    for (int layer = maxSolveLayer; layer >= 0; layer--) {
                        if (!globalDTHierarchy.TryGetLayerDt(layer, out DT globalLayerDt) || globalLayerDt == null)
                            continue;

                        if (!globalDTHierarchy.TryGetLayerMappings(layer, out int[] ownerBodyByLocal, out _, out int[] globalFineNodeByLocal, out int globalActiveCount, out int globalFineCount))
                            continue;
                        if (globalActiveCount < 3)
                            continue;

                        if (!globalDTHierarchy.TryGetLayerExecutionContext(layer, out int execActiveCount, out int execFineCount, out float layerKernelH))
                            continue;

                        bool useMappedIndices = !IsIdentityMapping(globalFineNodeByLocal, globalFineCount);
                        ComputeBuffer globalNodeMap = null;
                        ComputeBuffer globalToLocalMap = null;
                        if (useMappedIndices) {
                            globalNodeMap = EnsureGlobalLayerNodeMapBuffer(layer, globalFineNodeByLocal, globalFineCount);
                            globalToLocalMap = EnsureGlobalLayerGlobalToLocalBufferCached(layer, globalFineNodeByLocal, globalFineCount, totalCount);
                        }

                        ComputeBuffer ownerByLocalBuffer = null;
                        if (ownerBodyByLocal != null && ownerBodyByLocal.Length >= globalActiveCount)
                            ownerByLocalBuffer = EnsureGlobalLayerOwnerByLocalBuffer(layer, ownerBodyByLocal, globalActiveCount);

                        ProcessGlobalLayer(
                            layer,
                            globalLayerDt,
                            execActiveCount,
                            execFineCount,
                            layerKernelH,
                            tick,
                            forceEventsCount,
                            convergenceDebug,
                            maxSolveLayer,
                            useMappedIndices,
                            globalNodeMap,
                            globalToLocalMap,
                            ownerByLocalBuffer
                        );
                    }
                }

                asyncCb.SetComputeIntParam(shader, "_Base", 0);
                asyncCb.SetComputeIntParam(shader, "_TotalCount", totalCount);
                PrepareIntegratePosParams();
                Dispatch(shader, kClampVelocities, Groups256(totalCount), 1, 1);
                Dispatch(shader, kIntegratePositions, Groups256(totalCount), 1, 1);

                for (int layer = globalDTHierarchy.MaxLayer; layer >= 0; layer--) {
                    if (!globalDTHierarchy.TryGetLayerDt(layer, out DT dtLayer) || dtLayer == null)
                        continue;

                    if (!globalDTHierarchy.TryGetLayerMappings(layer, out _, out _, out int[] globalFineNodeByLocal, out int activeCount, out int fineCount))
                        continue;
                    if (activeCount < 3)
                        continue;

                    bool useMappedIndices = !IsIdentityMapping(globalFineNodeByLocal, fineCount);
                    if (useMappedIndices) {
                        ComputeBuffer globalNodeMap = EnsureGlobalLayerNodeMapBuffer(layer, globalFineNodeByLocal, fineCount);
                        ComputeBuffer globalToLocalMap = EnsureGlobalLayerGlobalToLocalBufferCached(layer, globalFineNodeByLocal, fineCount, totalCount);
                        PrepareUpdateDtPosParamsMapped(dtLayer, globalNodeMap, globalToLocalMap, activeCount, globalDTHierarchy.NormCenter, globalDTHierarchy.NormInvHalfExtent, writeSlot);
                        Dispatch(shader, kUpdateDtPositionsMapped, Groups256(activeCount), 1, 1);
                    } else {
                        PrepareUpdateDtPosParamsUnmapped(dtLayer, 0, activeCount, globalDTHierarchy.NormCenter, globalDTHierarchy.NormInvHalfExtent, writeSlot);
                        Dispatch(shader, kUpdateDtPositions, Groups256(activeCount), 1, 1);
                    }

                    dtLayer.EnqueueMaintain(asyncCb, dtLayer.GetPositionsBuffer(writeSlot),
                        readSlot, writeSlot, Const.DTFixIterations, Const.DTLegalizeIterations);
                }
            }

            var fence = asyncCb.CreateGraphicsFence(GraphicsFenceType.AsyncQueueSynchronisation,
                SynchronisationStageFlags.ComputeProcessing);

            Graphics.ExecuteCommandBufferAsync(asyncCb, queueType);

            if (ConvergenceDebugEnabled && convergenceDebug != null && convergenceDebugCpu != null && convergenceDebugRequiredUInts > 0) {
                convergenceDebug.GetData(convergenceDebugCpu);
                LogConvergenceStatsFromData(convergenceDebugCpu, maxSolveLayer, convergenceDebugMaxIter);
            }

            return fence;
        }

        static bool IsIdentityMapping(int[] globalNodeByLocal, int count) {
            if (globalNodeByLocal == null || globalNodeByLocal.Length < count)
                return false;

            for (int i = 0; i < count; i++) {
                if (globalNodeByLocal[i] != i)
                    return false;
            }

            return true;
        }

        private bool EnsureLayout(IReadOnlyList<Meshless> meshes, out int totalCount, out int maxSolveLayer) {
            totalCount = 0;
            maxSolveLayer = 0;

            if (meshes == null || meshes.Count == 0)
                return false;

            bool changed = false;
            int validIndex = 0;

            for (int i = 0; i < meshes.Count; i++) {
                Meshless m = meshes[i];
                if (m == null || m.nodes == null || m.nodes.Count <= 0)
                    continue;

                int count = m.nodes.Count;
                if (validIndex >= solveRanges.Count) {
                    changed = true;
                } else {
                    MeshRange existing = solveRanges[validIndex];
                    if (existing.meshless != m || existing.baseIndex != totalCount || existing.totalCount != count)
                        changed = true;
                }

                totalCount += count;
                maxSolveLayer = Mathf.Max(maxSolveLayer, m.maxLayer);
                validIndex++;
            }

            if (validIndex == 0 || totalCount <= 0)
                return false;

            if (validIndex != solveRanges.Count)
                changed = true;

            if (changed) {
                solveRanges.Clear();

                int baseIndex = 0;
                for (int i = 0; i < meshes.Count; i++) {
                    Meshless m = meshes[i];
                    if (m == null || m.nodes == null || m.nodes.Count <= 0)
                        continue;

                    int count = m.nodes.Count;
                    solveRanges.Add(new MeshRange {
                        meshless = m,
                        baseIndex = baseIndex,
                        totalCount = count,
                    });

                    baseIndex += count;
                }

                layoutInitialized = false;
            }

            return true;
        }
        /// <summary>
        /// Applies gameplay and external forces for this tick.
        /// </summary>
        private void ApplyForces(int total, int gameplayCountThisTick) {
            PrepareApplyForcesParams();

            if (gameplayCountThisTick > 0 && forceEvents != null) {
                asyncCb.SetComputeIntParam(shader, "_ForceEventCount", gameplayCountThisTick);
                asyncCb.SetComputeBufferParam(shader, kApplyGameplayForces, "_ForceEvents", forceEvents);
                Dispatch(shader, kApplyGameplayForces, Groups256(gameplayCountThisTick), 1, 1);
            } else {
                asyncCb.SetComputeIntParam(shader, "_ForceEventCount", 0);
            }

            Dispatch(shader, kExternalForces, Groups256(total), 1, 1);
        }


        private void ProcessGlobalLayer(
            int layer,
            DT dtLayer,
            int activeCount,
            int fineCount,
            float layerKernelH,
            int tickIndex,
            int gameplayCountThisTick,
            ComputeBuffer debugBuffer,
            int maxSolveLayer,
            bool useDtGlobalNodeMap,
            ComputeBuffer dtGlobalNodeMap,
            ComputeBuffer dtGlobalToLayerLocalMap,
            ComputeBuffer dtOwnerByLocal
        ) {
            PrepareRelaxBuffers(dtLayer, 0, activeCount, fineCount, tickIndex, layerKernelH, useDtGlobalNodeMap, 0, dtGlobalNodeMap, dtGlobalToLayerLocalMap, dtOwnerByLocal);

            Dispatch(shader, kClearHierarchicalStats, Groups256(activeCount), 1, 1);
            Dispatch(shader, kCacheHierarchicalStats, Groups256(fineCount), 1, 1);
            Dispatch(shader, kFinalizeHierarchicalStats, Groups256(activeCount), 1, 1);

            Dispatch(shader, kSaveVelPrefix, Groups256(activeCount), 1, 1);

            bool useHierarchyTransfer = layer > 0 && fineCount > activeCount;
            bool injectRestrictedGameplay =
                useHierarchyTransfer &&
                gameplayCountThisTick > 0 &&
                forceEvents != null &&
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

            if (!injectRestrictedGameplay)
                asyncCb.SetComputeIntParam(shader, "_ForceEventCount", 0);

            Dispatch(shader, kComputeCorrectionL, Groups256(activeCount), 1, 1);
            Dispatch(shader, kCacheF0AndResetLambda, Groups256(activeCount), 1, 1);
            Dispatch(shader, kResetCollisionLambda, Groups256(activeCount), 1, 1);

            var coloring = RebuildGlobalColoringForLayer(layer, dtLayer, activeCount, dtLayer.NeighborCount, layerKernelH);
            if (coloring != null) {
                asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_ColorOrder", coloring.OrderBuffer);
                asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_ColorStarts", coloring.StartsBuffer);
                asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_ColorCounts", coloring.CountsBuffer);
            }

            int iterations = GetIterationsForLayer(layer, maxSolveLayer);

            bool dbg = debugBuffer != null && tickIndex == 0;
            if (dbg) {
                ClearDebugBuffer(layer, iterations);
            } else {
                asyncCb.SetComputeIntParam(shader, "_ConvergenceDebugEnable", 0);
            }

            for (int iter = 0; iter < iterations; iter++) {
                if (dbg)
                    asyncCb.SetComputeIntParam(shader, "_ConvergenceDebugIter", iter);

                for (int c = 0; c < 16; c++) {
                    asyncCb.SetComputeIntParam(shader, "_ColorIndex", c);
                    if (coloring != null && coloring.RelaxArgsBuffer != null)
                        asyncCb.DispatchCompute(shader, kRelaxColored, coloring.RelaxArgsBuffer, (uint)c * 12);
                }
            }

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


        private DTColoring RebuildGlobalColoringForLayer(int layer, DT dtLayer, int activeCount, int dtNeighborCount, float layerCellSize) {
            if (coloringShader == null) {
                Debug.LogError("XPBISolver: No coloring shader provided. Cannot rebuild global coloring.");
                return null;
            }

            ulong key = 0xFFFFFFFF00000000UL | (uint)layer;
            if (!coloringByMeshLayer.TryGetValue(key, out DTColoring coloring) || coloring == null) {
                coloring = new DTColoring(coloringShader);
                coloringByMeshLayer[key] = coloring;
            }

            uint seed = 12345u + (uint)layer;
            if (coloring.ColorBuffer == null) {
                coloring.Init(activeCount, dtNeighborCount, seed);
                coloring.EnqueueInitTriGrid(asyncCb, pos, dtLayer, layerCellSize);
                dtLayer.MarkAllDirty(asyncCb);
            }

            coloring.EnqueueUpdateAfterMaintain(asyncCb, pos, dtLayer, layerCellSize, Const.ColoringConflictRounds);
            coloring.EnqueueRebuildOrderAndArgs(asyncCb);
            return coloring;
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

            foreach (var kv in coloringByMeshLayer)
                kv.Value?.Dispose();
            coloringByMeshLayer.Clear();

            solveRanges.Clear();
            layoutInitialized = false;

            kernelsCached = false;
        }
    }
}
