using GPU.Neighbors;
using GPU.Delaunay;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using LayerContext = GPU.Solver.XPBISolver.LayerContext;
using SolveSession = GPU.Solver.XPBISolver.SolveSession;
using TickContext = GPU.Solver.XPBISolver.TickContext;

namespace GPU.Solver {
    internal sealed partial class CollisionEvents {
        private const int UseTransferredCollisionsDisabled = 0;

        private readonly XPBISolver solver;
        private readonly ComputeShader shader;

        public CollisionEvents(XPBISolver solver) {
            this.solver = solver;
            shader = solver.CollisionEventsShader;
        }

        private static void Dispatch(CommandBuffer cb, string marker, ComputeShader dispatchShader, int kernel, int groupsX, int groupsY, int groupsZ) {
            cb.BeginSample(marker);
            cb.DispatchCompute(dispatchShader, kernel, groupsX, groupsY, groupsZ);
            cb.EndSample(marker);
        }

        /// <summary>
        /// Records layer-0 collision-event generation.
        /// </summary>
        public void RecordLayer0Build(SolveSession session, TickContext tickContext) {
            if (!solver.layerMappingCache.TryBuildLayerContext(session, 0, out LayerContext layer0))
                return;

            DT layer0Dt = layer0.NeighborSearch as DT;
            if (layer0Dt == null)
                session.Request.GlobalDTHierarchy.TryGetLayerDt(0, out layer0Dt);
            if (layer0Dt == null)
                return;

            RecordBuildLayer0CollisionEventsPerTick(
                session,
                layer0.NeighborSearch,
                layer0Dt,
                layer0.ActiveCount,
                layer0.FineCount,
                layer0.KernelH,
                tickContext.TickIndex,
                layer0.UseMappedIndices,
                layer0.GlobalNodeMap,
                layer0.GlobalToLocalMap,
                layer0.CollisionOwnerByLocalBuffer);
        }

        /// <summary>
        /// Records layer-0 collision event build for the current tick.
        /// </summary>
        private void RecordBuildLayer0CollisionEventsPerTick(
            SolveSession session,
            INeighborSearch layer0NeighborSearch,
            DT layer0Dt,
            int layer0ActiveCount,
            int layer0FineCount,
            float layer0KernelH,
            int tickIndex,
            bool useDtGlobalNodeMap,
            ComputeBuffer dtGlobalNodeMap,
            ComputeBuffer dtGlobalToLayerLocalMap,
            ComputeBuffer dtOwnerByLocal
        ) {
            if (layer0NeighborSearch == null || layer0Dt == null || layer0ActiveCount < 3)
                return;

            float2 boundsMin = session.Request.Layer0NeighborBoundsMin;
            float2 boundsMax = session.Request.Layer0NeighborBoundsMax;
            if (math.any(boundsMax <= boundsMin)) {
                boundsMin = new float2(-1f, -1f);
                boundsMax = new float2(1f, 1f);
            }

            PrepareLayer0BuildBuffers(
                session.AsyncCb,
                layer0NeighborSearch,
                layer0Dt,
                session.Request.ReadSlot,
                layer0ActiveCount,
                layer0KernelH,
                boundsMin,
                boundsMax,
                useDtGlobalNodeMap,
                dtGlobalNodeMap,
                dtGlobalToLayerLocalMap,
                dtOwnerByLocal);
            session.AsyncCb.SetComputeIntParam(shader, "_UseTransferredCollisions", UseTransferredCollisionsDisabled);
            Dispatch(session.AsyncCb, "XPBI.ClearCollisionEventCount", shader, kClearCollisionEventCount, 1, 1, 1);
            Dispatch(session.AsyncCb, "XPBI.ClearBoundaryChunkCount", shader, kClearBoundaryChunkCount, 1, 1, 1);
            Dispatch(session.AsyncCb, "XPBI.BuildBoundaryChunksL0", shader, kBuildBoundaryChunksL0, XPBISolver.Groups256(layer0Dt.HalfEdgeCount), 1, 1);
            Dispatch(session.AsyncCb, "XPBI.InitChunkSortKeys", shader, kInitChunkSortKeys, XPBISolver.Groups256(BoundaryChunkSortCapacity), 1, 1);

            for (int k = 2; k <= BoundaryChunkSortCapacity; k <<= 1) {
                for (int j = k >> 1; j > 0; j >>= 1) {
                    session.AsyncCb.SetComputeIntParam(shader, "_BitonicK", k);
                    session.AsyncCb.SetComputeIntParam(shader, "_BitonicJ", j);
                    Dispatch(session.AsyncCb, "XPBI.BitonicSortChunkKeys", shader, kBitonicSortChunkKeys, XPBISolver.Groups256(BoundaryChunkSortCapacity), 1, 1);
                }
            }

            Dispatch(session.AsyncCb, "XPBI.BuildLbvhLeaves", shader, kBuildLbvhLeaves, XPBISolver.Groups256(BoundaryChunkSortCapacity), 1, 1);

            int levelCount = BoundaryChunkSortCapacity >> 1;
            int levelStart = LbvhLeafOffset - levelCount;
            while (levelCount > 0) {
                session.AsyncCb.SetComputeIntParam(shader, "_LbvhLevelStart", levelStart);
                session.AsyncCb.SetComputeIntParam(shader, "_LbvhLevelCount", levelCount);
                Dispatch(session.AsyncCb, "XPBI.BuildLbvhInternalLevel", shader, kBuildLbvhInternalLevel, XPBISolver.Groups256(levelCount), 1, 1);
                levelCount >>= 1;
                levelStart -= levelCount;
            }

            Dispatch(session.AsyncCb, "XPBI.TraverseLbvhEmitCollisionEvents", shader, kTraverseLbvhEmitCollisionEvents, XPBISolver.Groups256(BoundaryChunkCapacity), 1, 1);
            Dispatch(session.AsyncCb, "XPBI.BuildCollisionEventsL0", shader, kBuildCollisionEventsL0, 1, 1, 1);
        }

        internal void RecordTransferredRestriction(SolveSession session, LayerContext layerContext) {
            bool useTransferredCollisions = layerContext.Layer > 0;
            solver.LayerCacheRuntime.SetUseTransferredCollisionsParam(session.AsyncCb, useTransferredCollisions);
            if (!useTransferredCollisions)
                return;

            Dispatch(session.AsyncCb, "XPBI.ClearTransferredCollision", shader, kClearTransferredCollision, XPBISolver.Groups256(layerContext.ActiveCount), 1, 1);
            Dispatch(session.AsyncCb, "XPBI.RestrictCollisionEventsToActivePairs", shader, kRestrictCollisionEventsToActivePairs, XPBISolver.Groups256(collisionEvents != null ? collisionEvents.count : 0), 1, 1);
        }
    }
}
