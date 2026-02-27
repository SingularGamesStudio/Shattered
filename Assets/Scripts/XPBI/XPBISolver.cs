using System;
using System.Collections.Generic;
using GPU.Delaunay;
using GPU.Neighbors;
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
        private readonly Dictionary<int, string[]> relaxDispatchMarkersByLayer = new Dictionary<int, string[]>(8);
        private readonly Dictionary<int, LayerColoringMaskCache> coloringMaskCacheByLayer = new Dictionary<int, LayerColoringMaskCache>(8);

        private sealed class LayerColoringMaskCache {
            public uint[] mask;
            public int[] ownerRef;
            public int activeCount;
            public int fixedSignature;
        }

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

        private void Dispatch(string marker, ComputeShader shader, int kernel, int x, int y, int z) {
            asyncCb.BeginSample(marker);
            asyncCb.DispatchCompute(shader, kernel, x, y, z);
            asyncCb.EndSample(marker);
        }

        private void DispatchIndirect(string marker, ComputeShader shader, int kernel, ComputeBuffer args, uint argsOffset) {
            asyncCb.BeginSample(marker);
            asyncCb.DispatchCompute(shader, kernel, args, argsOffset);
            asyncCb.EndSample(marker);
        }

        private string GetRelaxDispatchMarker(int layer, int color) {
            if (!relaxDispatchMarkersByLayer.TryGetValue(layer, out string[] markersByColor) || markersByColor == null) {
                markersByColor = new string[16];
                relaxDispatchMarkersByLayer[layer] = markersByColor;
            }

            string marker = markersByColor[color];
            if (marker == null) {
                marker = $"XPBI.RelaxColored.L{layer}.C{color}";
                markersByColor[color] = marker;
            }

            return marker;
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
    GlobalDTHierarchy globalDTHierarchy,
    INeighborSearch layer0NeighborSearch = null,
    float2 layer0NeighborBoundsMin = default,
    float2 layer0NeighborBoundsMax = default
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
            bool useOverrideLayer0NeighborSearch = layer0NeighborSearch != null;
            if (useOverrideLayer0NeighborSearch) {
                useHierarchical = false;
                maxSolveLayer = 0;
            }

            EnsureCapacity(totalCount);

            if (!layoutInitialized)
                InitializeFromMeshless((System.Collections.Generic.List<MeshRange>)solveRanges, totalCount);

            EnsureAsyncCommandBufferForRecording();
            EnsureConvergenceDebugCapacity(maxSolveLayer + 1, Const.JRIterationsL0 + Const.GSIterationsL0);
            bool[] coloringUpdatedByLayer = ConvergenceDebugEnabled ? new bool[maxSolveLayer + 1] : null;
            int fixedObjectSignature = ComputeFixedObjectSignature();

            for (int tick = 0; tick < tickCount; tick++) {
                SetCommonShaderParams(dtPerTick, Const.Gravity, Const.Compliance, totalCount, 0);

                if (useHierarchical) {
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
                            PrepareParentRebuildBuffers(
                                dtLayer,
                                0,
                                activeCount,
                                fineCount,
                                useDtGlobalNodeMap: true,
                                dtLocalBase: 0,
                                dtGlobalNodeMap: globalNodeMap,
                                dtGlobalToLayerLocalMap: globalToLocalMap);
                        } else {
                            PrepareParentRebuildBuffers(
                                dtLayer,
                                0,
                                activeCount,
                                fineCount,
                                useDtGlobalNodeMap: false,
                                dtLocalBase: 0,
                                dtGlobalNodeMap: null,
                                dtGlobalToLayerLocalMap: null);
                        }
                        Dispatch("XPBI.RebuildParentsAtLayer", shader, kRebuildParentsAtLayer, Groups256(fineCount - activeCount), 1, 1);
                    }
                }

                ApplyForces(totalCount, forceEventsCount);

                if (useOverrideLayer0NeighborSearch) {
                    if (globalDTHierarchy.TryGetLayerExecutionContext(0, out int activeCount, out _, out float layerKernelH)) {
                        float supportRadius = Mathf.Max(1e-5f, Const.WendlandSupport * layerKernelH);
                        float cellSize = supportRadius;
                        layer0NeighborSearch.EnqueueBuild(
                            asyncCb,
                            pos,
                            activeCount,
                            cellSize,
                            supportRadius,
                            layer0NeighborBoundsMin,
                            layer0NeighborBoundsMax,
                            readSlot,
                            writeSlot,
                            Const.DTFixIterations,
                            Const.DTLegalizeIterations,
                            true);
                    }
                }

                INeighborSearch layer0NeighborForCollision = null;
                int layer0ActiveForCollision = 0;
                int layer0FineForCollision = 0;
                float layer0KernelForCollision = 0f;
                bool layer0UseMappedForCollision = false;
                ComputeBuffer layer0NodeMapForCollision = null;
                ComputeBuffer layer0GlobalToLocalForCollision = null;
                ComputeBuffer layer0OwnerByLocalForCollision = null;

                if (globalDTHierarchy.TryGetLayerDt(0, out DT layer0DtForCollision) && layer0DtForCollision != null) {
                    layer0NeighborForCollision = useOverrideLayer0NeighborSearch ? layer0NeighborSearch : layer0DtForCollision;
                    if (layer0NeighborForCollision != null &&
                        globalDTHierarchy.TryGetLayerMappings(0, out int[] ownerBodyByLocalLayer0, out _, out int[] globalFineNodeByLocalLayer0, out int globalActiveCountLayer0, out int globalFineCountLayer0) &&
                        globalDTHierarchy.TryGetLayerExecutionContext(0, out int execActiveCountLayer0, out int execFineCountLayer0, out float layerKernelHLayer0)) {
                        layer0ActiveForCollision = execActiveCountLayer0;
                        layer0FineForCollision = execFineCountLayer0;
                        layer0KernelForCollision = layerKernelHLayer0;

                        layer0UseMappedForCollision = !IsIdentityMapping(globalFineNodeByLocalLayer0, globalFineCountLayer0);
                        if (layer0UseMappedForCollision) {
                            layer0NodeMapForCollision = EnsureGlobalLayerNodeMapBuffer(0, globalFineNodeByLocalLayer0, globalFineCountLayer0);
                            layer0GlobalToLocalForCollision = EnsureGlobalLayerGlobalToLocalBufferCached(0, globalFineNodeByLocalLayer0, globalFineCountLayer0, totalCount);
                        }

                        if (ownerBodyByLocalLayer0 != null && ownerBodyByLocalLayer0.Length >= globalActiveCountLayer0)
                            layer0OwnerByLocalForCollision = EnsureGlobalLayerOwnerByLocalBuffer(0, ownerBodyByLocalLayer0, globalActiveCountLayer0);
                    }
                }

                BuildLayer0CollisionEventsPerTick(
                    layer0NeighborForCollision,
                    layer0ActiveForCollision,
                    layer0FineForCollision,
                    layer0KernelForCollision,
                    tick,
                    layer0UseMappedForCollision,
                    layer0NodeMapForCollision,
                    layer0GlobalToLocalForCollision,
                    layer0OwnerByLocalForCollision
                );

                for (int layer = maxSolveLayer; layer >= 0; layer--) {
                    if (!globalDTHierarchy.TryGetLayerDt(layer, out DT globalLayerDt) || globalLayerDt == null)
                        continue;

                    INeighborSearch layerNeighborSearch = useOverrideLayer0NeighborSearch && layer == 0
                        ? layer0NeighborSearch
                        : globalLayerDt;
                    if (layerNeighborSearch == null)
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
                        layerNeighborSearch,
                        ownerBodyByLocal,
                        fixedObjectSignature,
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
                        ownerByLocalBuffer,
                        coloringUpdatedByLayer
                    );
                }

                asyncCb.SetComputeIntParam(shader, "_Base", 0);
                asyncCb.SetComputeIntParam(shader, "_TotalCount", totalCount);
                PrepareIntegratePosParams();
                Dispatch("XPBI.ClampVelocities", shader, kClampVelocities, Groups256(totalCount), 1, 1);
                Dispatch("XPBI.IntegratePositions", shader, kIntegratePositions, Groups256(totalCount), 1, 1);

                for (int layer = maxSolveLayer; layer >= 0; layer--) {
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
                        Dispatch("XPBI.UpdateDtPositionsMapped", shader, kUpdateDtPositionsMapped, Groups256(activeCount), 1, 1);
                    } else {
                        PrepareUpdateDtPosParamsUnmapped(dtLayer, 0, activeCount, globalDTHierarchy.NormCenter, globalDTHierarchy.NormInvHalfExtent, writeSlot);
                        Dispatch("XPBI.UpdateDtPositions", shader, kUpdateDtPositions, Groups256(activeCount), 1, 1);
                    }

                    if (!(useOverrideLayer0NeighborSearch && layer == 0)) {
                        float layerSupportRadius = 1f;
                        if (globalDTHierarchy.TryGetLayerExecutionContext(layer, out _, out _, out float dtLayerKernelH))
                            layerSupportRadius = Mathf.Max(1e-5f, Const.WendlandSupport * dtLayerKernelH);

                        float normalizedLayerSupportRadius = Mathf.Max(1e-8f, layerSupportRadius * globalDTHierarchy.NormInvHalfExtent);

                        dtLayer.EnqueueBuild(asyncCb, dtLayer.GetPositionsBuffer(writeSlot),
                            activeCount, normalizedLayerSupportRadius, normalizedLayerSupportRadius, float2.zero, float2.zero,
                            readSlot, writeSlot, Const.DTFixIterations, Const.DTLegalizeIterations);
                    }
                }
            }

            var fence = asyncCb.CreateGraphicsFence(GraphicsFenceType.AsyncQueueSynchronisation,
                SynchronisationStageFlags.ComputeProcessing);

            Graphics.ExecuteCommandBufferAsync(asyncCb, queueType);

            if (ConvergenceDebugEnabled && convergenceDebug != null && convergenceDebugRequiredUInts > 0) {
                Graphics.WaitOnAsyncGraphicsFence(fence);

                int dbgMaxLayer = maxSolveLayer;
                int dbgMaxIter = convergenceDebugMaxIter;
                coloringUpdatedByLayerForDebugLog = coloringUpdatedByLayer;

                AsyncGPUReadback.Request(convergenceDebug, req => {
                    if (req.hasError) return;
                    var data = req.GetData<uint>();
                    LogConvergenceStatsFromData(data.ToArray(), dbgMaxLayer, dbgMaxIter);
                });

                RequestColoringConflictHistoryDebugLog(dbgMaxLayer);
            }
            coloringUpdatedByLayerForDebugLog = null;

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
                Dispatch("XPBI.ApplyGameplayForces", shader, kApplyGameplayForces, Groups256(gameplayCountThisTick), 1, 1);
            } else {
                asyncCb.SetComputeIntParam(shader, "_ForceEventCount", 0);
            }

            Dispatch("XPBI.ExternalForces", shader, kExternalForces, Groups256(total), 1, 1);
        }

        private void BuildLayer0CollisionEventsPerTick(
            INeighborSearch layer0NeighborSearch,
            int layer0ActiveCount,
            int layer0FineCount,
            float layer0KernelH,
            int tickIndex,
            bool useDtGlobalNodeMap,
            ComputeBuffer dtGlobalNodeMap,
            ComputeBuffer dtGlobalToLayerLocalMap,
            ComputeBuffer dtOwnerByLocal
        ) {
            if (layer0NeighborSearch == null || layer0ActiveCount < 3)
                return;

            PrepareRelaxBuffers(layer0NeighborSearch, 0, layer0ActiveCount, layer0FineCount, tickIndex, layer0KernelH, useDtGlobalNodeMap, 0, dtGlobalNodeMap, dtGlobalToLayerLocalMap, dtOwnerByLocal);
            asyncCb.SetComputeIntParam(shader, "_UseTransferredCollisions", 0);
            Dispatch("XPBI.ClearCollisionEventCount", shader, kClearCollisionEventCount, 1, 1, 1);
            Dispatch("XPBI.BuildCollisionEventsL0", shader, kBuildCollisionEventsL0, Groups256(layer0ActiveCount), 1, 1);
        }


        private void ProcessGlobalLayer(
            int layer,
            INeighborSearch neighborSearch,
            int[] ownerBodyByLocal,
            int fixedObjectSignature,
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
            ComputeBuffer dtOwnerByLocal,
            bool[] coloringUpdatedByLayer
        ) {
            PrepareRelaxBuffers(neighborSearch, 0, activeCount, fineCount, tickIndex, layerKernelH, useDtGlobalNodeMap, 0, dtGlobalNodeMap, dtGlobalToLayerLocalMap, dtOwnerByLocal);

            Dispatch("XPBI.ClearHierarchicalStats", shader, kClearHierarchicalStats, Groups256(activeCount), 1, 1);
            Dispatch("XPBI.CacheHierarchicalStats", shader, kCacheHierarchicalStats, Groups256(fineCount), 1, 1);
            Dispatch("XPBI.FinalizeHierarchicalStats", shader, kFinalizeHierarchicalStats, Groups256(activeCount), 1, 1);

            Dispatch("XPBI.SaveVelPrefix", shader, kSaveVelPrefix, Groups256(activeCount), 1, 1);

            bool useHierarchyTransfer = layer > 0 && fineCount > activeCount;
            bool injectRestrictedGameplay =
                useHierarchyTransfer &&
                gameplayCountThisTick > 0 &&
                forceEvents != null &&
                Const.RestrictedDeltaVScale > 0f;
            bool injectRestrictedResidual =
                useHierarchyTransfer &&
                Const.RestrictResidualDeltaVScale > 0f;

            if (injectRestrictedResidual) {
                Dispatch("XPBI.ClearRestrictedDeltaV", shader, kClearRestrictedDeltaV, Groups256(activeCount), 1, 1);
                Dispatch("XPBI.RestrictFineVelocityResidualToActive", shader, kRestrictFineVelocityResidualToActive, Groups256(fineCount - activeCount), 1, 1);
                asyncCb.SetComputeFloatParam(shader, "_RestrictedDeltaVScale", Const.RestrictResidualDeltaVScale);
                Dispatch("XPBI.ApplyRestrictedDeltaVToActiveAndPrefix", shader, kApplyRestrictedDeltaVToActiveAndPrefix, Groups256(activeCount), 1, 1);
            }

            if (injectRestrictedGameplay) {
                Dispatch("XPBI.ClearRestrictedDeltaV", shader, kClearRestrictedDeltaV, Groups256(activeCount), 1, 1);
                asyncCb.SetComputeIntParam(shader, "_ForceEventCount", gameplayCountThisTick);
                asyncCb.SetComputeBufferParam(shader, kRestrictGameplayDeltaVFromEvents, "_ForceEvents", forceEvents);
                Dispatch("XPBI.RestrictGameplayDeltaVFromEvents", shader, kRestrictGameplayDeltaVFromEvents, Groups256(gameplayCountThisTick), 1, 1);
                asyncCb.SetComputeFloatParam(shader, "_RestrictedDeltaVScale", Const.RestrictedDeltaVScale);
                Dispatch("XPBI.ApplyRestrictedDeltaVToActiveAndPrefix", shader, kApplyRestrictedDeltaVToActiveAndPrefix, Groups256(activeCount), 1, 1);
            }

            if (!injectRestrictedGameplay)
                asyncCb.SetComputeIntParam(shader, "_ForceEventCount", 0);

            Dispatch("XPBI.ComputeCorrectionL", shader, kComputeCorrectionL, Groups256(activeCount), 1, 1);
            Dispatch("XPBI.CacheF0AndResetLambda", shader, kCacheF0AndResetLambda, Groups256(activeCount), 1, 1);
            Dispatch("XPBI.ResetCollisionLambda", shader, kResetCollisionLambda, Groups256(activeCount), 1, 1);

            bool useTransferredCollisions = layer > 0;
            asyncCb.SetComputeIntParam(shader, "_UseTransferredCollisions", useTransferredCollisions ? 1 : 0);
            if (useTransferredCollisions) {
                Dispatch("XPBI.ClearTransferredCollision", shader, kClearTransferredCollision, Groups256(activeCount), 1, 1);
                Dispatch("XPBI.RestrictCollisionEventsToActivePairs", shader, kRestrictCollisionEventsToActivePairs, Groups256(collisionEvents != null ? collisionEvents.count : 0), 1, 1);
            }

            int JRIterations = GetJRIterationsForLayer(layer, maxSolveLayer);
            int GSIterations = layer == 0 ? Const.GSIterationsL0 : 1;

            DTColoring coloring = null;
            if (GSIterations > 0) {
                coloring = RebuildGlobalColoringForLayer(layer, neighborSearch, ownerBodyByLocal, fixedObjectSignature, activeCount, layerKernelH);
                if (coloring != null) {
                    if (coloringUpdatedByLayer != null && layer >= 0 && layer < coloringUpdatedByLayer.Length)
                        coloringUpdatedByLayer[layer] = true;

                    asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_ColorOrder", coloring.OrderBuffer);
                    asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_ColorStarts", coloring.StartsBuffer);
                    asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_ColorCounts", coloring.CountsBuffer);
                    asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_ColorOrder", coloring.OrderBuffer);
                    asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_ColorStarts", coloring.StartsBuffer);
                    asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_ColorCounts", coloring.CountsBuffer);
                }
            }

            bool usePersistentCoarseGs =
                GSIterations > 0 &&
                coloring != null &&
                activeCount <= Const.PersistentCoarseMaxNodes;

            int debugIterations = GSIterations + JRIterations;
            bool dbg = debugBuffer != null && tickIndex == 0;
            if (dbg && debugIterations > 0) {
                ClearDebugBuffer(layer, debugIterations);
            } else {
                asyncCb.SetComputeIntParam(shader, "_ConvergenceDebugEnable", 0);
            }

            // ----- Stage 1: GS solve -----
            if (GSIterations > 0 && coloring != null) {
                asyncCb.SetComputeIntParam(shader, "_CollisionEnable", 1);

                if (usePersistentCoarseGs) {
                    asyncCb.SetComputeIntParam(shader, "_PersistentIters", GSIterations);
                    asyncCb.SetComputeIntParam(shader, "_PersistentBaseDebugIter", 0);
                    Dispatch("XPBI.RelaxColoredPersistentCoarse", shader, kRelaxColoredPersistentCoarse, 1, 1, 1);
                } else {
                    for (int iter = 0; iter < GSIterations; iter++) {
                        if (dbg) asyncCb.SetComputeIntParam(shader, "_ConvergenceDebugIter", iter);

                        for (int c = 0; c < 16; c++) {
                            asyncCb.SetComputeIntParam(shader, "_ColorIndex", c);
                            if (coloring.RelaxArgsBuffer != null)
                                DispatchIndirect(GetRelaxDispatchMarker(layer, c), shader, kRelaxColored,
                                                 coloring.RelaxArgsBuffer, (uint)c * 12);
                        }
                    }
                }
            }

            // ----- Stage 2: JR solve (uncolored) -----
            if (JRIterations > 0) {
                asyncCb.SetComputeIntParam(shader, "_CollisionEnable", 0);

                asyncCb.SetComputeFloatParam(shader, "_JROmegaV", Const.JROmegaV);
                asyncCb.SetComputeFloatParam(shader, "_JROmegaL", Const.JROmegaL);

                for (int iter = 0; iter < JRIterations; iter++) {
                    if (dbg) asyncCb.SetComputeIntParam(shader, "_ConvergenceDebugIter", GSIterations + iter);

                    Dispatch("XPBI.JR.SavePrevAndClear", shader, kJRSavePrevAndClear, Groups256(activeCount), 1, 1);

                    Dispatch("XPBI.JR.ComputeDeltas", shader, kJRComputeDeltas, Groups256(activeCount), 1, 1);

                    Dispatch("XPBI.JR.Apply", shader, kJRApply, Groups256(activeCount), 1, 1);
                }
            }

            if (layer > 0 && fineCount > activeCount) {
                Dispatch("XPBI.Prolongate", shader, kProlongate, Groups256(fineCount - activeCount), 1, 1);
                if (Const.PostProlongSmoothing > 0f)
                    Dispatch("XPBI.SmoothProlongatedFineVel", shader, kSmoothProlongatedFineVel, Groups256(fineCount - activeCount), 1, 1);
            }

            if (injectRestrictedGameplay || injectRestrictedResidual)
                Dispatch("XPBI.RemoveRestrictedDeltaVFromActive", shader, kRemoveRestrictedDeltaVFromActive, Groups256(activeCount), 1, 1);

            if (layer == 0)
                Dispatch("XPBI.CommitDeformation", shader, kCommitDeformation, Groups256(activeCount), 1, 1);
        }

        private int GetJRIterationsForLayer(int layer, int maxSolveLayer) {
            int iterations = Const.JRIterationsLMid;
            if (layer == maxSolveLayer)
                iterations = Const.JRIterationsLMax;
            if (layer == 0)
                iterations = Const.JRIterationsL0;

            return Mathf.Max(1, iterations);
        }

        private int ComputeFixedObjectSignature() {
            int signature = 17;
            for (int i = 0; i < solveRanges.Count; i++) {
                Meshless meshless = solveRanges[i].meshless;
                signature = unchecked(signature * 31 + ((meshless != null && meshless.fixedObject) ? 1 : 0));
            }

            return signature;
        }

        private uint[] BuildLayerColoringActiveMask(int layer, int[] ownerBodyByLocal, int fixedObjectSignature, int activeCount) {
            if (!coloringMaskCacheByLayer.TryGetValue(layer, out LayerColoringMaskCache cache) || cache == null) {
                cache = new LayerColoringMaskCache();
                coloringMaskCacheByLayer[layer] = cache;
            }

            if (cache.mask == null || cache.mask.Length < activeCount)
                cache.mask = new uint[Mathf.Max(1, activeCount)];

            bool needsRebuild =
                cache.ownerRef != ownerBodyByLocal ||
                cache.activeCount != activeCount ||
                cache.fixedSignature != fixedObjectSignature;

            if (needsRebuild) {
                bool hasOwners = ownerBodyByLocal != null && ownerBodyByLocal.Length >= activeCount;
                for (int i = 0; i < activeCount; i++) {
                    bool include = true;

                    if (hasOwners) {
                        int owner = ownerBodyByLocal[i];
                        if (owner >= 0 && owner < solveRanges.Count) {
                            Meshless ownerMesh = solveRanges[owner].meshless;
                            include = ownerMesh != null && !ownerMesh.fixedObject;
                        }
                    }

                    cache.mask[i] = include ? 1u : 0u;
                }

                cache.ownerRef = ownerBodyByLocal;
                cache.activeCount = activeCount;
                cache.fixedSignature = fixedObjectSignature;
            }

            return cache.mask;
        }


        private DTColoring RebuildGlobalColoringForLayer(int layer, INeighborSearch neighborSearch, int[] ownerBodyByLocal, int fixedObjectSignature, int activeCount, float layerCellSize) {
            if (coloringShader == null) {
                Debug.LogError("XPBISolver: No coloring shader provided. Cannot rebuild global coloring.");
                return null;
            }

            ulong key = 0xFFFFFFFF00000000UL | (uint)layer;
            if (!coloringByMeshLayer.TryGetValue(key, out DTColoring coloring) || coloring == null) {
                coloring = new DTColoring(coloringShader);
                coloringByMeshLayer[key] = coloring;
            }

            uint[] activeMask = BuildLayerColoringActiveMask(layer, ownerBodyByLocal, fixedObjectSignature, activeCount);

            uint seed = 12345u + (uint)layer;
            if (coloring.ColorBuffer == null) {
                coloring.Init(activeCount, neighborSearch.NeighborCount, seed);
                coloring.UpdateActiveMask(activeMask, activeCount);
                coloring.EnqueueInitTriGrid(asyncCb, pos, layerCellSize);
                neighborSearch.MarkAllDirty(asyncCb);
                coloring.EnqueueUpdateAfterMaintain(asyncCb, pos, neighborSearch, layerCellSize, Const.InitColoringConflictRounds);
            } else {
                coloring.UpdateActiveMask(activeMask, activeCount);
                coloring.EnqueueUpdateAfterMaintain(asyncCb, pos, neighborSearch, layerCellSize, Const.ColoringConflictRounds);
            }
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
            coloringMaskCacheByLayer.Clear();
            relaxDispatchMarkersByLayer.Clear();

            solveRanges.Clear();
            layoutInitialized = false;

            kernelsCached = false;
        }
    }
}
