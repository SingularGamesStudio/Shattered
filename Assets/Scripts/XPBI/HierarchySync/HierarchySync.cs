using GPU.Delaunay;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using SolveSession = GPU.Solver.XPBISolver.SolveSession;

namespace GPU.Solver {
    internal sealed partial class HierarchySync {
        private readonly XPBISolver solver;
        private readonly ComputeShader shader;

        public HierarchySync(XPBISolver solver) {
            this.solver = solver;
            shader = solver.HierarchySyncShader;
        }

        private static void Dispatch(CommandBuffer cb, string marker, ComputeShader dispatchShader, int kernel, int groupsX, int groupsY, int groupsZ) {
            cb.BeginSample(marker);
            cb.DispatchCompute(dispatchShader, kernel, groupsX, groupsY, groupsZ);
            cb.EndSample(marker);
        }

        /// <summary>
        /// Records parent rebuild for hierarchical layers before per-layer relax passes.
        /// </summary>
        public void RecordPreSolveParentRebuild(SolveSession session) {
            if (!session.UseHierarchical)
                return;

            for (int layer = session.Request.GlobalDTHierarchy.MaxLayer; layer >= 1; layer--) {
                if (!session.Request.GlobalDTHierarchy.TryGetLayerDt(layer, out DT dtLayer) || dtLayer == null)
                    continue;

                if (!session.Request.GlobalDTHierarchy.TryGetLayerMappings(layer, out _, out _, out int[] globalFineNodeByLocal, out int activeCount, out int fineCount))
                    continue;
                if (fineCount <= activeCount)
                    continue;

                bool useMappedIndices = !XPBISolver.IsIdentityMapping(globalFineNodeByLocal, fineCount);
                var mapping = useMappedIndices
                    ? new LayerCacheRuntime.DtMappingContext(true, 0,
                        solver.layerMappingCache.EnsureGlobalLayerNodeMapBuffer(layer, globalFineNodeByLocal, fineCount),
                        solver.layerMappingCache.EnsureGlobalLayerGlobalToLocalBufferCached(layer, globalFineNodeByLocal, fineCount, session.TotalCount),
                        null,
                        null)
                    : new LayerCacheRuntime.DtMappingContext(false, 0, null, null, null, null);

                float parentMaxDistance = 0f;
                if (session.Request.GlobalDTHierarchy.TryGetLayerExecutionContext(layer, out _, out _, out float layerKernelH))
                    parentMaxDistance = math.max(0f, Const.ParentRelationMaxSupportScale * Const.WendlandSupport * layerKernelH);
                PrepareParentRebuildBuffers(session.AsyncCb, session.Pos, solver.parentIndex, solver.parentIndices, solver.parentWeights, new ParentRebuildContext(dtLayer, 0, activeCount, fineCount, parentMaxDistance, mapping));
                Dispatch(session.AsyncCb, "XPBI.RebuildParentsAtLayer", shader, kRebuildParentsAtLayer, XPBISolver.Groups256(fineCount - activeCount), 1, 1);
            }
        }

        /// <summary>
        /// Records per-layer DT position sync and DT maintenance passes after integration.
        /// </summary>
        public void RecordPostIntegrateDtSync(SolveSession session) {
            RecordPostIntegrateDtSync(session, session.Request.ReadSlot);
        }

        public void RecordPostIntegrateDtSync(SolveSession session, int dtReadSlot) {
            for (int layer = session.MaxSolveLayer; layer >= 0; layer--) {
                if (!session.Request.GlobalDTHierarchy.TryGetLayerDt(layer, out DT dtLayer) || dtLayer == null)
                    continue;

                if (!session.Request.GlobalDTHierarchy.TryGetLayerMappings(layer, out _, out _, out int[] globalFineNodeByLocal, out int activeCount, out int fineCount))
                    continue;
                if (activeCount < 3)
                    continue;

                bool useMappedIndices = !XPBISolver.IsIdentityMapping(globalFineNodeByLocal, fineCount);
                if (useMappedIndices) {
                    ComputeBuffer globalNodeMap = solver.layerMappingCache.EnsureGlobalLayerNodeMapBuffer(layer, globalFineNodeByLocal, fineCount);
                    ComputeBuffer globalToLocalMap = solver.layerMappingCache.EnsureGlobalLayerGlobalToLocalBufferCached(layer, globalFineNodeByLocal, fineCount, session.TotalCount);
                    PrepareUpdateDtPosParamsMapped(session.AsyncCb, session.Pos, dtLayer, globalNodeMap, globalToLocalMap, activeCount, session.Request.GlobalDTHierarchy.NormCenter, session.Request.GlobalDTHierarchy.NormInvHalfExtent, session.Request.WriteSlot);
                    Dispatch(session.AsyncCb, "XPBI.UpdateDtPositionsMapped", shader, kUpdateDtPositionsMapped, XPBISolver.Groups256(activeCount), 1, 1);
                } else {
                    PrepareUpdateDtPosParamsUnmapped(session.AsyncCb, session.Pos, dtLayer, 0, activeCount, session.Request.GlobalDTHierarchy.NormCenter, session.Request.GlobalDTHierarchy.NormInvHalfExtent, session.Request.WriteSlot);
                    Dispatch(session.AsyncCb, "XPBI.UpdateDtPositions", shader, kUpdateDtPositions, XPBISolver.Groups256(activeCount), 1, 1);
                }

                if (layer == 0 && session.UseOverrideLayer0NeighborSearch) {
                    if (session.Request.GlobalDTHierarchy.TryGetLayerExecutionContext(layer, out _, out _, out float layerKernelH)) {
                        float supportRadius = Mathf.Max(1e-5f, Const.WendlandSupport * layerKernelH);
                        float cellSize = supportRadius;
                        session.Request.Layer0NeighborSearch.EnqueueBuild(
                            session.AsyncCb,
                            session.Pos,
                            activeCount,
                            cellSize,
                            supportRadius,
                            session.Request.Layer0NeighborBoundsMin,
                            session.Request.Layer0NeighborBoundsMax,
                            dtReadSlot,
                            session.Request.WriteSlot,
                            Const.DTFixIterations,
                            Const.DTLegalizeIterations,
                            true);
                    }
                } else {
                    float layerSupportRadius = 1f;
                    if (session.Request.GlobalDTHierarchy.TryGetLayerExecutionContext(layer, out _, out _, out float dtLayerKernelH))
                        layerSupportRadius = Mathf.Max(1e-5f, Const.WendlandSupport * dtLayerKernelH);

                    float normalizedLayerSupportRadius = Mathf.Max(1e-8f, layerSupportRadius * session.Request.GlobalDTHierarchy.NormInvHalfExtent);
                    float normalizedLayerCellSize = normalizedLayerSupportRadius;
                    float normalizedLayerNeighborSupportRadius = normalizedLayerSupportRadius;

                    if (layer == 0 && dtLayer.SupportsFullRebuild) {
                        int kFrames = Mathf.Max(1, Const.DTLayer0FullRebuildEveryKFrames);
                        if ((Time.frameCount % kFrames) == 0) {//TODO: not framecount
                            dtLayer.EnqueueFullRebuild(
                                session.AsyncCb,
                                dtLayer.GetPositionsBuffer(session.Request.WriteSlot),
                                session.Request.WriteSlot,
                                rebuildAdjacencyAndTriMap: true);
                            continue;
                        }
                    }

                    dtLayer.EnqueueBuild(
                        session.AsyncCb,
                        dtLayer.GetPositionsBuffer(session.Request.WriteSlot),
                        activeCount,
                        normalizedLayerCellSize,
                        normalizedLayerNeighborSupportRadius,
                        float2.zero,
                        float2.zero,
                        dtReadSlot,
                        session.Request.WriteSlot,
                        Const.DTFixIterations,
                        Const.DTLegalizeIterations);
                }
            }
        }

        /// <summary>
        /// Records velocity clamping and position integration across all nodes.
        /// </summary>
        public void RecordIntegrate(SolveSession session) {
            session.AsyncCb.SetComputeIntParam(shader, "_Base", 0);
            session.AsyncCb.SetComputeIntParam(shader, "_TotalCount", session.TotalCount);
            PrepareIntegratePosParams(session.AsyncCb, session.Pos, session.Vel, session.InvMass);
            Dispatch(session.AsyncCb, "XPBI.ClampVelocities", shader, kClampVelocities, XPBISolver.Groups256(session.TotalCount), 1, 1);
            Dispatch(session.AsyncCb, "XPBI.IntegratePositions", shader, kIntegratePositions, XPBISolver.Groups256(session.TotalCount), 1, 1);
        }

    }
}
