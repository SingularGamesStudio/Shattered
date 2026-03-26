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

        // ------------------------------------------------------------------------
        // Layer 0: emit raw fine contacts only.
        // ------------------------------------------------------------------------
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

            Dispatch(session.AsyncCb, "XPBI.QueryNodeSurfaceContactsAB", shader, kQueryNodeSurfaceContacts, queryGroupsX, 1, 1);
            session.AsyncCb.SetComputeIntParam(shader, "_QuerySwap", 1);
            Dispatch(session.AsyncCb, "XPBI.QueryNodeSurfaceContactsBA", shader, kQueryNodeSurfaceContacts, queryGroupsX, 1, 1);
            Dispatch(session.AsyncCb, "XPBI.QueryEdgeEdgeNodeContacts", shader, kQueryEdgeEdgeNodeContacts, queryGroupsX, 1, 1);

            int scatterFineGroups = Groups64(math.max(1, collisionEvents != null ? collisionEvents.count : 0));
            Dispatch(session.AsyncCb, "XPBI.ClearFineNodeManifolds", shader, kClearFineNodeManifolds, Groups64(math.max(1, layer0ActiveCount * 2)), 1, 1);
            Dispatch(session.AsyncCb, "XPBI.ScatterFineNodeContactsToManifolds", shader, kScatterFineNodeContactsToManifolds, scatterFineGroups, 1, 1);

            session.AsyncCb.SetComputeIntParam(shader, "_QuerySwap", 0);
        }

        // ------------------------------------------------------------------------
        // Layer 0: build fine stencil + CSR immediately before solving layer 0.
        // ------------------------------------------------------------------------
        internal void RecordLayer0StencilBuild(SolveSession session, LayerContext layerContext) {
            _ = session;
            _ = layerContext;
        }

        // ------------------------------------------------------------------------
        // Upper layers: build coarse stencil + CSR for the current layer only.
        // ------------------------------------------------------------------------
        internal void RecordTransferredRestriction(SolveSession session, LayerContext layerContext) {
            if (layerContext == null || layerContext.Layer <= 0)
                return;

            if (kClearCoarseNodeContacts < 0 || kPropagateFineContactsToCoarse < 0)
                return;

            if (collisionEvents == null || coarseContacts == null)
                return;

            int coarseParentsPerNode = 2;

            // Capacity is still okay for dispatch sizing, but the shader-side propagation
            // kernels should clamp against the actual emitted fine count buffer, not treat
            // this as the real fine-contact count.
            session.AsyncCb.SetComputeIntParam(shader, "_FineContactCount", collisionEvents.count);
            session.AsyncCb.SetComputeIntParam(shader, "_MaxFineNodeContacts", collisionEvents.count);
            session.AsyncCb.SetComputeIntParam(shader, "_ActiveCount", layerContext.ActiveCount);
            session.AsyncCb.SetComputeIntParam(shader, "_CoarseNodeContactStride", 4);
            session.AsyncCb.SetComputeIntParam(shader, "_CoarseParentsPerNode", coarseParentsPerNode);

            solver.layerMappingCache.BindDtGlobalMappingParams(
                session.AsyncCb,
                shader,
                kClearCoarseNodeContacts,
                layerContext.UseMappedIndices,
                0,
                layerContext.GlobalNodeMap,
                layerContext.GlobalToLocalMap);

            solver.layerMappingCache.BindDtGlobalMappingParams(
                session.AsyncCb,
                shader,
                kPropagateFineContactsToCoarse,
                layerContext.UseMappedIndices,
                0,
                layerContext.GlobalNodeMap,
                layerContext.GlobalToLocalMap);

            int clearWork = math.max(layerContext.ActiveCount, layerContext.ActiveCount * 4);
            Dispatch(session.AsyncCb, "XPBI.ClearCoarseNodeContacts", shader, kClearCoarseNodeContacts, Groups64(math.max(1, clearWork)), 1, 1);

            int vertexWork = (int)math.min((long)collisionEvents.count * coarseParentsPerNode, int.MaxValue);
            if (vertexWork > 0)
                Dispatch(session.AsyncCb, "XPBI.PropagateFineContactsToCoarse", shader, kPropagateFineContactsToCoarse, Groups64(vertexWork), 1, 1);
        }
    }
}
