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

        internal readonly ComputeShader layerCacheShader;
        internal readonly ComputeShader layerSolveShader;
        internal readonly ComputeShader gameplayShader;
        internal readonly ComputeShader hierarchyShader;
        internal readonly ComputeShader collisionShader;
        internal readonly ComputeShader solverDebugShader;
        internal readonly ComputeShader coloringShader;
        private readonly ComputeShader[] commonParamShaders;

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
            public CommandBuffer AsyncCb;
            public int TotalCount;
            public ComputeBuffer Pos;
            public ComputeBuffer Vel;
            public ComputeBuffer InvMass;
            public IReadOnlyList<MeshRange> SolveRanges;
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
            public readonly bool HasForceEventsBuffer;

            public TickContext(int tickIndex, int forceCount, bool hasForceEventsBuffer) {
                TickIndex = tickIndex;
                ForceCount = forceCount;
                HasForceEventsBuffer = hasForceEventsBuffer;
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
        /// <param name="coloringShader">Compute shader used to build/maintain graph coloring for colored relaxation.</param>
        /// <param name="layerCacheSolveShader">Optional explicit reference to the cache/restriction layer compute shader.</param>
        /// <param name="layerSolveShader">Optional explicit reference to the actual solve layer compute shader.</param>
        /// <param name="gameplayForcesShader">Optional explicit reference to the gameplay-forces compute shader.</param>
        /// <param name="hierarchySyncShader">Optional explicit reference to the hierarchy-sync compute shader.</param>
        /// <param name="collisionEventsShader">Optional explicit reference to the collision-events compute shader.</param>
        public XPBISolver(
            ComputeShader coloringShader,
            ComputeShader layerCacheSolveShader = null,
            ComputeShader layerSolveShader = null,
            ComputeShader gameplayForcesShader = null,
            ComputeShader hierarchySyncShader = null,
            ComputeShader collisionEventsShader = null
        ) {
            this.layerCacheShader = layerCacheSolveShader ?? layerSolveShader;
            this.layerSolveShader = layerSolveShader ?? layerCacheSolveShader;
            gameplayShader = gameplayForcesShader;
            hierarchyShader = hierarchySyncShader;
            collisionShader = collisionEventsShader;
            solverDebugShader = this.layerSolveShader;
            this.coloringShader = coloringShader;
            commonParamShaders = BuildUniqueShaderList(this.layerCacheShader, this.layerSolveShader, gameplayShader, hierarchyShader, collisionShader, solverDebugShader);

            gameplayForce = new GameplayForces(this);
            layerMappingCache = new LayerMappingCache();
            hierarchySync = new HierarchySync(this);
            collisionEvent = new CollisionEvents(this);
            coloring = new Coloring(this);
            layerSolveRuntime = new LayerSolveRuntime(this);
            layerCacheRuntime = new LayerCacheRuntime(this);
            layerCachePass = new LayerCachePass(layerCacheRuntime);
            layerSolvePass = new LayerSolvePass(layerSolveRuntime);
            solverDebug = new SolverDebug(this);
        }

        private static ComputeShader[] BuildUniqueShaderList(params ComputeShader[] shaders) {
            List<ComputeShader> unique = new List<ComputeShader>(shaders.Length);
            for (int i = 0; i < shaders.Length; i++) {
                ComputeShader shader = shaders[i];
                if (shader == null || unique.Contains(shader))
                    continue;
                unique.Add(shader);
            }

            return unique.ToArray();
        }
    }
}
