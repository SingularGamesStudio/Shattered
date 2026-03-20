using System;
using System.Collections.Generic;
using GPU.Delaunay;
using GPU.Neighbors;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace GPU.Solver {
    public sealed partial class XPBISolver : IDisposable {
        private int initializedCount = -1;
        internal bool layoutInitialized;
        internal CommandBuffer asyncCb;

        internal struct MeshRange {
            public Meshless meshless;
            public int baseIndex;
            public int totalCount;
        }

        internal readonly List<MeshRange> solveRanges = new List<MeshRange>(64);
        private readonly Dictionary<int, string[]> relaxDispatchMarkersByLayer = new Dictionary<int, string[]>(8);

        internal readonly GameplayForces gameplayForce;
        internal readonly LayerMappingCache layerMappingCache;
        private readonly HierarchySync hierarchySync;
        internal readonly CollisionEvents collisionEvent;
        private readonly LayerCacheRuntime layerCacheRuntime;
        private readonly LayerSolveRuntime layerSolveRuntime;
        private readonly LayerCachePass layerCachePass;
        private readonly LayerSolvePass layerSolvePass;
        internal readonly Coloring coloring;
        internal readonly SolverDebug solverDebug;

        internal LayerCacheRuntime LayerCacheRuntime => layerCacheRuntime;
        internal LayerSolveRuntime LayerSolveRuntime => layerSolveRuntime;
        internal ComputeShader LayerCacheShader => layerCacheShader;
        internal ComputeShader LayerSolveShader => layerSolveShader;
        internal ComputeShader GameplayForcesShader => gameplayShader;
        internal ComputeShader HierarchySyncShader => hierarchyShader;
        internal ComputeShader CollisionEventsShader => collisionShader;
        internal ComputeShader SolverDebugShader => solverDebugShader;
        internal IReadOnlyList<ComputeShader> CommonParamShaders => commonParamShaders;

        internal static int Groups256(int count) {
            return (count + 255) / 256;
        }

        internal void EnsureAsyncCommandBufferForRecording() {
            if (asyncCb == null)
                asyncCb = new CommandBuffer { name = "XPBI Async Batch" };

            asyncCb.Clear();
            asyncCb.SetExecutionFlags(CommandBufferExecutionFlags.AsyncCompute);
        }

        internal void Dispatch(string marker, ComputeShader shader, int kernel, int x, int y, int z) {
            asyncCb.BeginSample(marker);
            asyncCb.DispatchCompute(shader, kernel, x, y, z);
            asyncCb.EndSample(marker);
        }

        internal void DispatchIndirect(string marker, ComputeShader shader, int kernel, ComputeBuffer args, uint argsOffset) {
            asyncCb.BeginSample(marker);
            asyncCb.DispatchCompute(shader, kernel, args, argsOffset);
            asyncCb.EndSample(marker);
        }

        internal string GetRelaxDispatchMarker(int layer, int color) {
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
            gameplayForce.SetForces(events, count);
        }

        /// <summary>
        /// Clears gameplay forces (no force events will be applied until new ones are set).
        /// </summary>
        public void ClearGameplayForces() {
            gameplayForce.ClearForces();
        }

        /// <summary>
        /// Records and submits one global solve for all active meshless systems.
        /// </summary>
        public GraphicsFence SubmitSolve(
    IReadOnlyList<Meshless> meshes,
    float dtPerTick,
    int tickCount,
    int readSlot,
    int writeSlot,
            DTHierarchy globalDTHierarchy,
    INeighborSearch layer0NeighborSearch = null,
    float2 layer0NeighborBoundsMin = default,
    float2 layer0NeighborBoundsMax = default
) {
            SolveRequest request = new SolveRequest(
                meshes,
                dtPerTick,
                tickCount,
                readSlot,
                writeSlot,
                globalDTHierarchy,
                layer0NeighborSearch,
                layer0NeighborBoundsMin,
                layer0NeighborBoundsMax);
            EnsureKernelsCached();

            if (!TryBuildSession(request, out SolveSession session))
                return default;

            EnsureAsyncCommandBufferForRecording();
            session.AsyncCb = asyncCb;
            solverDebug.PrepareSession(session);

            for (int tick = 0; tick < session.Request.TickCount; tick++) {
                TickContext tickContext = new TickContext(tick, gameplayForce.EventCount, gameplayForce.HasEventsBuffer);
                SetCommonShaderParams(session.Request.DtPerTick, session.TotalCount, 0);

                hierarchySync.RecordPreSolveParentRebuild(session);
                gameplayForce.RecordApplyForces(session, tickContext);
                collisionEvent.RecordLayer0Build(session, tickContext);

                for (int layer = session.MaxSolveLayer; layer >= 0; layer--) {
                    if (!layerMappingCache.TryBuildLayerContext(session, layer, out LayerContext layerContext))
                        continue;

                    layerCachePass.RecordCache(session, tickContext, layerContext, out bool injectRestrictedGameplay, out bool injectRestrictedResidual);
                    collisionEvent.RecordTransferredRestriction(session, layerContext);
                    layerSolvePass.RecordSolve(session, tickContext, layerContext);
                    layerCachePass.RecordRestrictionCleanup(session.AsyncCb, layerContext.ActiveCount, injectRestrictedGameplay, injectRestrictedResidual);
                }

                hierarchySync.RecordIntegrate(session);
                hierarchySync.RecordPostIntegrateDtSync(session);
            }

            GraphicsFence fence = asyncCb.CreateGraphicsFence(
                GraphicsFenceType.AsyncQueueSynchronisation,
                SynchronisationStageFlags.ComputeProcessing);
            Graphics.ExecuteCommandBufferAsync(asyncCb, ComputeQueueType.Urgent);

            solverDebug.RecordReadbacksAndFence(session, fence);
            return fence;
        }

        /// <summary>
        /// Returns whether a local-to-global mapping is an identity mapping for the requested prefix.
        /// </summary>
        internal static bool IsIdentityMapping(int[] globalNodeByLocal, int count) {
            if (globalNodeByLocal == null || globalNodeByLocal.Length < count)
                return false;

            for (int i = 0; i < count; i++) {
                if (globalNodeByLocal[i] != i)
                    return false;
            }

            return true;
        }

        /// <summary>
        /// Validates request input, ensures capacities, and builds a complete solve session.
        /// </summary>
        private bool TryBuildSession(in SolveRequest request, out SolveSession session) {
            session = null;

            if (request.TickCount <= 0)
                return false;

            if (request.GlobalDTHierarchy == null || request.GlobalDTHierarchy.MaxLayer < 0)
                return false;

            if (!EnsureLayout(request.Meshes, out int totalCount, out int maxSolveLayer))
                return false;

            maxSolveLayer = Mathf.Max(maxSolveLayer, request.GlobalDTHierarchy.MaxLayer);

            bool useOverrideLayer0NeighborSearch = request.Layer0NeighborSearch != null;
            bool useHierarchical = SimulationParamSource.Current.interaction.useHierarchicalSolver && !useOverrideLayer0NeighborSearch;
            if (!useHierarchical)
                maxSolveLayer = 0;

            EnsureCapacity(totalCount);
            if (!layoutInitialized)
                InitializeFromMeshless(solveRanges, totalCount);

            int convergenceDebugMaxLayer = useHierarchical ? maxSolveLayer : 0;
            int convergenceDebugLayerCount = convergenceDebugMaxLayer + 1;
            int convergenceDebugMaxIterations = Mathf.Max(
                Const.GSIterationsL0 + Const.JRIterationsL0,
                1 + Mathf.Max(Const.JRIterationsLMax, Const.JRIterationsLMid));

            session = new SolveSession {
                Request = request,
                TotalCount = totalCount,
                Pos = pos,
                Vel = vel,
                InvMass = invMass,
                SolveRanges = solveRanges,
                MaxSolveLayer = maxSolveLayer,
                UseHierarchical = useHierarchical,
                UseOverrideLayer0NeighborSearch = useOverrideLayer0NeighborSearch,
                EnableProlongationConstraintProbeDebug = Const.ProlongationConstraintDebugEnabled,
                FixedObjectSignature = ComputeFixedObjectSignature(),
                ConvergenceDebugMaxLayer = convergenceDebugMaxLayer,
                ConvergenceDebugLayerCount = convergenceDebugLayerCount,
                ConvergenceDebugMaxIterations = convergenceDebugMaxIterations,
            };

            session.MaxProlongationProbeSamples = session.EnableProlongationConstraintProbeDebug && session.UseHierarchical
                ? Mathf.Max(0, request.TickCount * Mathf.Max(0, maxSolveLayer) * 2)
                : 0;

            return true;
        }

        /// <summary>
        /// Builds or validates the active solve layout and derives aggregate node/layer counts.
        /// </summary>
        private bool EnsureLayout(IReadOnlyList<Meshless> meshes, out int totalCount, out int maxSolveLayer) {
            totalCount = 0;
            maxSolveLayer = 0;

            if (meshes == null || meshes.Count == 0)
                return false;

            bool changed = false;
            int validIndex = 0;

            for (int i = 0; i < meshes.Count; i++) {
                Meshless meshless = meshes[i];
                if (meshless == null || meshless.nodes == null || meshless.nodes.Count <= 0)
                    continue;

                int count = meshless.nodes.Count;
                if (validIndex >= solveRanges.Count) {
                    changed = true;
                } else {
                    MeshRange existing = solveRanges[validIndex];
                    if (existing.meshless != meshless || existing.baseIndex != totalCount || existing.totalCount != count)
                        changed = true;
                }

                totalCount += count;
                maxSolveLayer = Mathf.Max(maxSolveLayer, meshless.maxLayer);
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
                    Meshless meshless = meshes[i];
                    if (meshless == null || meshless.nodes == null || meshless.nodes.Count <= 0)
                        continue;

                    int count = meshless.nodes.Count;
                    solveRanges.Add(new MeshRange {
                        meshless = meshless,
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
        /// Computes a stable signature for fixed-object ownership used by coloring cache invalidation.
        /// </summary>
        private int ComputeFixedObjectSignature() {
            int signature = 17;
            for (int i = 0; i < solveRanges.Count; i++) {
                Meshless meshless = solveRanges[i].meshless;
                signature = unchecked(signature * 31 + ((meshless != null && meshless.fixedObject) ? 1 : 0));
            }

            return signature;
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

            gameplayForce.Release();

            if (asyncCb != null) {
                asyncCb.Dispose();
                asyncCb = null;
            }

            coloring.Release();
            relaxDispatchMarkersByLayer.Clear();

            solveRanges.Clear();
            layoutInitialized = false;

            kernelsCached = false;
        }
    }
}
