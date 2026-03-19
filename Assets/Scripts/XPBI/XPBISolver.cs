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
        internal bool layoutInitialized;
        internal CommandBuffer asyncCb;

        internal struct MeshRange {
            public Meshless meshless;
            public int baseIndex;
            public int totalCount;
        }

        internal readonly List<MeshRange> solveRanges = new List<MeshRange>(64);
        private readonly Dictionary<int, string[]> relaxDispatchMarkersByLayer = new Dictionary<int, string[]>(8);

        internal readonly ComputeShader shader;
        internal readonly ComputeShader coloringShader;

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

        internal readonly GameplayForces gameplayForce;
        private readonly SolveLayout solveLayout;
        internal readonly LayerMappingCache layerMappingCache;
        private readonly HierarchySync hierarchySync;
        internal readonly CollisionEvents collisionEvent;
        private readonly LayerSolve layerSolve;
        internal readonly Coloring coloring;
        internal readonly SolverDebug solverDebug;

        internal readonly struct ProlongationConstraintProbe {
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

        /// <summary>
        /// Immutable input for one solve submission.
        /// </summary>
        internal readonly struct SolveRequest {
            public readonly IReadOnlyList<Meshless> Meshes;
            public readonly float DtPerTick;
            public readonly int TickCount;
            public readonly bool UseHierarchical;
            public readonly bool ConvergenceDebugEnabled;
            public readonly ComputeQueueType QueueType;
            public readonly int ReadSlot;
            public readonly int WriteSlot;
            public readonly GlobalDTHierarchy GlobalDTHierarchy;
            public readonly INeighborSearch Layer0NeighborSearch;
            public readonly float2 Layer0NeighborBoundsMin;
            public readonly float2 Layer0NeighborBoundsMax;

            public SolveRequest(
                IReadOnlyList<Meshless> meshes,
                float dtPerTick,
                int tickCount,
                bool useHierarchical,
                bool convergenceDebugEnabled,
                ComputeQueueType queueType,
                int readSlot,
                int writeSlot,
                GlobalDTHierarchy globalDTHierarchy,
                INeighborSearch layer0NeighborSearch,
                float2 layer0NeighborBoundsMin,
                float2 layer0NeighborBoundsMax
            ) {
                Meshes = meshes;
                DtPerTick = dtPerTick;
                TickCount = tickCount;
                UseHierarchical = useHierarchical;
                ConvergenceDebugEnabled = convergenceDebugEnabled;
                QueueType = queueType;
                ReadSlot = readSlot;
                WriteSlot = writeSlot;
                GlobalDTHierarchy = globalDTHierarchy;
                Layer0NeighborSearch = layer0NeighborSearch;
                Layer0NeighborBoundsMin = layer0NeighborBoundsMin;
                Layer0NeighborBoundsMax = layer0NeighborBoundsMax;
            }
        }

        /// <summary>
        /// Derived state and reusable objects for one solve submission.
        /// </summary>
        internal sealed class SolveSession {
            public SolveRequest Request;
            public int TotalCount;
            public int MaxSolveLayer;
            public bool UseHierarchical;
            public bool UseOverrideLayer0NeighborSearch;
            public bool EnableProlongationConstraintProbeDebug;
            public int FixedObjectSignature;
            public int ConvergenceDebugMaxLayer;
            public int ConvergenceDebugLayerCount;
            public int ConvergenceDebugMaxIterations;
            public int MaxProlongationProbeSamples;
            public bool[] ColoringUpdatedByLayer;
            public List<ProlongationConstraintProbe> ProlongationConstraintProbes;
            public int ProlongationProbeCursor;
        }

        /// <summary>
        /// Per-tick state used by phase recorders.
        /// </summary>
        internal readonly struct TickContext {
            public readonly int TickIndex;
            public readonly int ForceCount;

            public TickContext(int tickIndex, int forceCount) {
                TickIndex = tickIndex;
                ForceCount = forceCount;
            }
        }

        /// <summary>
        /// Layer-local execution data resolved from the global hierarchy.
        /// </summary>
        internal sealed class LayerContext {
            public int Layer;
            public INeighborSearch NeighborSearch;
            public int[] OwnerBodyByLocal;
            public int ActiveCount;
            public int FineCount;
            public float KernelH;
            public bool UseMappedIndices;
            public ComputeBuffer GlobalNodeMap;
            public ComputeBuffer GlobalToLocalMap;
            public ComputeBuffer OwnerByLocalBuffer;
        }

        /// <summary>
        /// Creates a new solver instance.
        /// </summary>
        /// <param name="solverShader">Compute shader implementing the XPBI pipeline.</param>
        /// <param name="coloringShader">Compute shader used to build/maintain graph coloring for colored relaxation.</param>
        public XPBISolver(ComputeShader solverShader, ComputeShader coloringShader) {
            this.shader = solverShader;
            this.coloringShader = coloringShader;

            gameplayForce = new GameplayForces(this);
            solveLayout = new SolveLayout(this);
            layerMappingCache = new LayerMappingCache(this);
            hierarchySync = new HierarchySync(this);
            collisionEvent = new CollisionEvents(this);
            coloring = new Coloring(this);
            layerSolve = new LayerSolve(this);
            solverDebug = new SolverDebug(this);
        }

        internal LayerSolve LayerSolve => layerSolve;

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
            SolveRequest request = new SolveRequest(
                meshes,
                dtPerTick,
                tickCount,
                useHierarchical,
                ConvergenceDebugEnabled,
                queueType,
                readSlot,
                writeSlot,
                globalDTHierarchy,
                layer0NeighborSearch,
                layer0NeighborBoundsMin,
                layer0NeighborBoundsMax);
            EnsureKernelsCached();

            if (!solveLayout.TryBuildSession(request, out SolveSession session))
                return default;

            EnsureAsyncCommandBufferForRecording();
            solverDebug.PrepareSession(session);

            for (int tick = 0; tick < session.Request.TickCount; tick++) {
                TickContext tickContext = new TickContext(tick, gameplayForce.EventCount);
                SetCommonShaderParams(session.Request.DtPerTick, Const.Gravity, Const.Compliance, session.TotalCount, 0);

                hierarchySync.RecordPreSolveParentRebuild(session);
                gameplayForce.RecordApplyForces(session, tickContext);
                collisionEvent.RecordLayer0Build(session, tickContext);

                for (int layer = session.MaxSolveLayer; layer >= 0; layer--) {
                    if (!layerMappingCache.TryBuildLayerContext(session, layer, out LayerContext layerContext))
                        continue;

                    layerSolve.Record(session, tickContext, layerContext);
                }

                hierarchySync.RecordIntegrate(session);
                hierarchySync.RecordPostIntegrateDtSync(session);
            }

            GraphicsFence fence = asyncCb.CreateGraphicsFence(
                GraphicsFenceType.AsyncQueueSynchronisation,
                SynchronisationStageFlags.ComputeProcessing);
            Graphics.ExecuteCommandBufferAsync(asyncCb, session.Request.QueueType);

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
