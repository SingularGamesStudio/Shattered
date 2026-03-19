using GPU.Delaunay;
using Unity.Mathematics;
using UnityEngine;
using SolveSession = GPU.Solver.XPBISolver.SolveSession;

namespace GPU.Solver {
    internal sealed partial class HierarchySync {
        private readonly XPBISolver solver;

        public HierarchySync(XPBISolver solver) {
            this.solver = solver;
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
                    ? new LayerSolve.DtMappingContext(true, 0,
                        solver.layerMappingCache.EnsureGlobalLayerNodeMapBuffer(layer, globalFineNodeByLocal, fineCount),
                        solver.layerMappingCache.EnsureGlobalLayerGlobalToLocalBufferCached(layer, globalFineNodeByLocal, fineCount, session.TotalCount),
                        null)
                    : new LayerSolve.DtMappingContext(false, 0, null, null, null);

                PrepareParentRebuildBuffers(new ParentRebuildContext(dtLayer, 0, activeCount, fineCount, mapping));
                solver.Dispatch("XPBI.RebuildParentsAtLayer", solver.shader, kRebuildParentsAtLayer, XPBISolver.Groups256(fineCount - activeCount), 1, 1);
            }
        }

        /// <summary>
        /// Records per-layer DT position sync and DT maintenance passes after integration.
        /// </summary>
        public void RecordPostIntegrateDtSync(SolveSession session) {
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
                    PrepareUpdateDtPosParamsMapped(dtLayer, globalNodeMap, globalToLocalMap, activeCount, session.Request.GlobalDTHierarchy.NormCenter, session.Request.GlobalDTHierarchy.NormInvHalfExtent, session.Request.WriteSlot);
                    solver.Dispatch("XPBI.UpdateDtPositionsMapped", solver.shader, kUpdateDtPositionsMapped, XPBISolver.Groups256(activeCount), 1, 1);
                } else {
                    PrepareUpdateDtPosParamsUnmapped(dtLayer, 0, activeCount, session.Request.GlobalDTHierarchy.NormCenter, session.Request.GlobalDTHierarchy.NormInvHalfExtent, session.Request.WriteSlot);
                    solver.Dispatch("XPBI.UpdateDtPositions", solver.shader, kUpdateDtPositions, XPBISolver.Groups256(activeCount), 1, 1);
                }

                if (!(session.UseOverrideLayer0NeighborSearch && layer == 0)) {
                    float layerSupportRadius = 1f;
                    if (session.Request.GlobalDTHierarchy.TryGetLayerExecutionContext(layer, out _, out _, out float dtLayerKernelH))
                        layerSupportRadius = Mathf.Max(1e-5f, Const.WendlandSupport * dtLayerKernelH);

                    float normalizedLayerSupportRadius = Mathf.Max(1e-8f, layerSupportRadius * session.Request.GlobalDTHierarchy.NormInvHalfExtent);
                    float normalizedLayerCellSize = normalizedLayerSupportRadius;
                    float normalizedLayerNeighborSupportRadius = normalizedLayerSupportRadius;

                    dtLayer.EnqueueBuild(
                        solver.asyncCb,
                        dtLayer.GetPositionsBuffer(session.Request.WriteSlot),
                        activeCount,
                        normalizedLayerCellSize,
                        normalizedLayerNeighborSupportRadius,
                        float2.zero,
                        float2.zero,
                        session.Request.ReadSlot,
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
            solver.asyncCb.SetComputeIntParam(solver.shader, "_Base", 0);
            solver.asyncCb.SetComputeIntParam(solver.shader, "_TotalCount", session.TotalCount);
            PrepareIntegratePosParams();
            solver.Dispatch("XPBI.ClampVelocities", solver.shader, kClampVelocities, XPBISolver.Groups256(session.TotalCount), 1, 1);
            solver.Dispatch("XPBI.IntegratePositions", solver.shader, kIntegratePositions, XPBISolver.Groups256(session.TotalCount), 1, 1);
        }

    }
}
