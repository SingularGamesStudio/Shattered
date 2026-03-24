using System;
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

        private static int Groups64(int count) {
            if (count <= 0)
                return 0;
            return (count + 63) / 64;
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
                dtOwnerByLocal,
                true);

            int collisionOwnerCount = math.max(1, session.SolveRanges != null ? session.SolveRanges.Count : 0);
            int edgeDispatchCount = math.max(1, layer0Dt.HalfEdgeCount);
            int pairDispatchCount = math.max(1, QueryPairCount);

            int clearStateWork = math.max(
                math.max(collisionOwnerCount, layer0Dt.HalfEdgeCount),
                math.max(TotalBinCapacity, collisionEvents != null ? collisionEvents.count : 0));
            Dispatch(session.AsyncCb, "XPBI.ClearState", shader, kClearState, Groups64(clearStateWork), 1, 1);

            Dispatch(session.AsyncCb, "XPBI.BuildBoundaryFeatures", shader, kBuildBoundaryFeatures, Groups64(edgeDispatchCount), 1, 1);
            Dispatch(session.AsyncCb, "XPBI.BinBoundaryEdges", shader, kBinBoundaryEdges, Groups64(edgeDispatchCount), 1, 1);
            Dispatch(session.AsyncCb, "XPBI.BinBoundaryVertices", shader, kBinBoundaryVertices, Groups64(edgeDispatchCount), 1, 1);

            int sdfGroups = (SdfResolution + 7) / 8;
            Dispatch(session.AsyncCb, "XPBI.BuildOwnerFeatureField", shader, kBuildOwnerFeatureField, sdfGroups, sdfGroups, collisionOwnerCount);

            session.AsyncCb.SetComputeIntParam(shader, "_QuerySwap", 0);
            int queryWorkItems = math.max(1, MaxBoundaryEdgesPerOwner * pairDispatchCount);
            int queryGroupsX = Groups64(queryWorkItems);

            Dispatch(session.AsyncCb, "XPBI.QueryVertexContactsAB", shader, kQueryVertexContacts, queryGroupsX, 1, 1);
            Dispatch(session.AsyncCb, "XPBI.QueryEdgeEdgeContacts", shader, kQueryEdgeEdgeContacts, queryGroupsX, 1, 1);

            session.AsyncCb.SetComputeIntParam(shader, "_QuerySwap", 0);
        }

        internal void RecordTransferredRestriction(SolveSession session, LayerContext layerContext) {
            if (layerContext == null || layerContext.Layer <= 0) {
                solver.LayerCacheRuntime.SetUseTransferredCollisionsParam(session.AsyncCb, false);
                return;
            }

            if (kClearCoarseContacts < 0 || kPropagateVertexContacts < 0 || kPropagateEdgeContacts < 0) {
                solver.LayerCacheRuntime.SetUseTransferredCollisionsParam(session.AsyncCb, false);
                return;
            }

            if (collisionEvents == null || coarseContacts == null) {
                solver.LayerCacheRuntime.SetUseTransferredCollisionsParam(session.AsyncCb, false);
                return;
            }

            int coarseParentsPerNode = math.clamp(Const.ParentKNearest, 1, 4);
            int fineContactCapacity = collisionEvents.count;

            session.AsyncCb.SetComputeIntParam(shader, "_FineContactCount", fineContactCapacity);
            session.AsyncCb.SetComputeIntParam(shader, "_MaxCoarseContacts", coarseContacts.count);
            session.AsyncCb.SetComputeIntParam(shader, "_CoarseParentsPerNode", coarseParentsPerNode);

            Dispatch(session.AsyncCb, "XPBI.ClearCoarseContacts", shader, kClearCoarseContacts, Groups64(coarseContacts.count), 1, 1);

            int vertexWork = (int)math.min((long)fineContactCapacity * coarseParentsPerNode, int.MaxValue);
            if (vertexWork > 0)
                Dispatch(session.AsyncCb, "XPBI.PropagateVertexContacts", shader, kPropagateVertexContacts, Groups64(vertexWork), 1, 1);

            int edgeWork = (int)math.min((long)fineContactCapacity * 4L, int.MaxValue);
            if (edgeWork > 0)
                Dispatch(session.AsyncCb, "XPBI.PropagateEdgeContacts", shader, kPropagateEdgeContacts, Groups64(edgeWork), 1, 1);

            // Layer solve still consumes legacy _XferCol* manifold slots.
            // Keep this disabled until coarse-contact outputs are bridged into that layout.
            solver.LayerCacheRuntime.SetUseTransferredCollisionsParam(session.AsyncCb, false);
        }
    }
}
