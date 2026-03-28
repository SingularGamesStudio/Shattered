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

            int queryWorkItems = math.max(1, MaxBoundaryEdgesPerOwner * pairDispatchCount);
            int queryGroupsX = Groups64(queryWorkItems);

            // Vertex-feature contacts: A -> B
            session.AsyncCb.SetComputeIntParam(shader, "_QuerySwap", 0);
            Dispatch(session.AsyncCb, "XPBI.QueryVertexContactsAB", shader, kQueryVertexContacts, queryGroupsX, 1, 1);

            // Vertex-feature contacts: B -> A
            session.AsyncCb.SetComputeIntParam(shader, "_QuerySwap", 1);
            Dispatch(session.AsyncCb, "XPBI.QueryVertexContactsBA", shader, kQueryVertexContacts, queryGroupsX, 1, 1);

            // Edge-edge contacts: only once, unswapped
            session.AsyncCb.SetComputeIntParam(shader, "_QuerySwap", 0);
            Dispatch(session.AsyncCb, "XPBI.QueryEdgeEdgeContacts", shader, kQueryEdgeEdgeContacts, queryGroupsX, 1, 1);

            // Leave default state predictable for later passes
            session.AsyncCb.SetComputeIntParam(shader, "_QuerySwap", 0);
        }

        // ------------------------------------------------------------------------
        // Layer 0: build fine stencil + CSR immediately before solving layer 0.
        // ------------------------------------------------------------------------
        internal void RecordLayer0StencilBuild(SolveSession session, LayerContext layerContext) {
            if (layerContext == null || layerContext.Layer != 0)
                return;

            int[] stencilKernels = {
                kClearNodeCollisionRefAux,
                kBuildFineContactStencils,
                kExclusiveScanNodeCollisionRefCount,
                kClearNodeCollisionRefWrite,
                kScatterFineContactRefs,
            };

            for (int i = 0; i < stencilKernels.Length; i++) {
                solver.layerMappingCache.BindDtGlobalMappingParams(
                    session.AsyncCb,
                    shader,
                    stencilKernels[i],
                    layerContext.UseMappedIndices,
                    0,
                    layerContext.GlobalNodeMap,
                    layerContext.GlobalToLocalMap);
            }

            session.AsyncCb.SetComputeIntParam(shader, "_ActiveCount", layerContext.ActiveCount);
            session.AsyncCb.SetComputeIntParam(shader, "_CollisionEventCapacity", collisionEvents != null ? collisionEvents.count : 0);
            session.AsyncCb.SetComputeIntParam(shader, "_CoarseContactCapacity", coarseContacts != null ? coarseContacts.count : 0);

            Dispatch(session.AsyncCb, "XPBI.ClearNodeCollisionRefAuxFine", shader, kClearNodeCollisionRefAux, Groups64(math.max(1, layerContext.ActiveCount)), 1, 1);
            Dispatch(session.AsyncCb, "XPBI.BuildFineContactStencils", shader, kBuildFineContactStencils, Groups64(math.max(1, collisionEvents != null ? collisionEvents.count : 0)), 1, 1);
            Dispatch(session.AsyncCb, "XPBI.ExclusiveScanNodeCollisionRefCountFine", shader, kExclusiveScanNodeCollisionRefCount, 1, 1, 1);
            Dispatch(session.AsyncCb, "XPBI.ClearNodeCollisionRefWriteFine", shader, kClearNodeCollisionRefWrite, Groups64(math.max(1, layerContext.ActiveCount)), 1, 1);
            Dispatch(session.AsyncCb, "XPBI.ScatterFineContactRefs", shader, kScatterFineContactRefs, Groups64(math.max(1, collisionEvents != null ? collisionEvents.count : 0)), 1, 1);
        }

        // ------------------------------------------------------------------------
        // Upper layers: build coarse stencil + CSR for the current layer only.
        // ------------------------------------------------------------------------
        internal void RecordTransferredRestriction(SolveSession session, LayerContext layerContext) {
            if (layerContext == null || layerContext.Layer <= 0)
                return;

            if (kClearCoarseContacts < 0 || kPropagateVertexContacts < 0 || kPropagateEdgeContacts < 0)
                return;

            if (collisionEvents == null || coarseContacts == null)
                return;

            int coarseParentsPerNode = 2;

            // Capacity is still okay for dispatch sizing, but the shader-side propagation
            // kernels should clamp against the actual emitted fine count buffer, not treat
            // this as the real fine-contact count.
            session.AsyncCb.SetComputeIntParam(shader, "_FineContactCapacity", collisionEvents.count);
            session.AsyncCb.SetComputeIntParam(shader, "_MaxCoarseContacts", coarseContacts.count);
            session.AsyncCb.SetComputeIntParam(shader, "_CoarseParentsPerNode", coarseParentsPerNode);

            Dispatch(session.AsyncCb, "XPBI.ClearCoarseContacts", shader, kClearCoarseContacts, Groups64(coarseContacts.count), 1, 1);

            int vertexWork = (int)math.min((long)collisionEvents.count * coarseParentsPerNode, int.MaxValue);
            if (vertexWork > 0)
                Dispatch(session.AsyncCb, "XPBI.PropagateVertexContacts", shader, kPropagateVertexContacts, Groups64(vertexWork), 1, 1);

            int edgeWork = (int)math.min((long)collisionEvents.count * 4L, int.MaxValue);
            if (edgeWork > 0)
                Dispatch(session.AsyncCb, "XPBI.PropagateEdgeContacts", shader, kPropagateEdgeContacts, Groups64(edgeWork), 1, 1);

            int[] stencilKernels = {
                kClearNodeCollisionRefAux,
                kBuildCoarseContactStencils,
                kExclusiveScanNodeCollisionRefCount,
                kClearNodeCollisionRefWrite,
                kScatterCoarseContactRefs,
            };

            for (int i = 0; i < stencilKernels.Length; i++) {
                solver.layerMappingCache.BindDtGlobalMappingParams(
                    session.AsyncCb,
                    shader,
                    stencilKernels[i],
                    layerContext.UseMappedIndices,
                    0,
                    layerContext.GlobalNodeMap,
                    layerContext.GlobalToLocalMap);
            }

            session.AsyncCb.SetComputeIntParam(shader, "_ActiveCount", layerContext.ActiveCount);
            session.AsyncCb.SetComputeIntParam(shader, "_CollisionEventCapacity", collisionEvents != null ? collisionEvents.count : 0);
            session.AsyncCb.SetComputeIntParam(shader, "_CoarseContactCapacity", coarseContacts != null ? coarseContacts.count : 0);

            Dispatch(session.AsyncCb, "XPBI.ClearNodeCollisionRefAuxCoarse", shader, kClearNodeCollisionRefAux, Groups64(math.max(1, layerContext.ActiveCount)), 1, 1);
            Dispatch(session.AsyncCb, "XPBI.BuildCoarseContactStencils", shader, kBuildCoarseContactStencils, Groups64(math.max(1, coarseContacts.count)), 1, 1);
            Dispatch(session.AsyncCb, "XPBI.ExclusiveScanNodeCollisionRefCountCoarse", shader, kExclusiveScanNodeCollisionRefCount, 1, 1, 1);
            Dispatch(session.AsyncCb, "XPBI.ClearNodeCollisionRefWriteCoarse", shader, kClearNodeCollisionRefWrite, Groups64(math.max(1, layerContext.ActiveCount)), 1, 1);
            Dispatch(session.AsyncCb, "XPBI.ScatterCoarseContactRefs", shader, kScatterCoarseContactRefs, Groups64(math.max(1, coarseContacts.count)), 1, 1);
        }
    }
}
