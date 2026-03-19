using GPU.Delaunay;
using GPU.Neighbors;
using Unity.Mathematics;
using UnityEngine;

namespace GPU.Solver {
    internal sealed partial class HierarchySync {
        internal int kClampVelocities;
        internal int kIntegratePositions;
        internal int kUpdateDtPositions;
        internal int kUpdateDtPositionsMapped;
        internal int kRebuildParentsAtLayer;

        internal void CacheRuntimeKernels() {
            kClampVelocities = solver.shader.FindKernel("ClampVelocities");
            kIntegratePositions = solver.shader.FindKernel("IntegratePositions");
            kUpdateDtPositions = solver.shader.FindKernel("UpdateDtPositions");
            kUpdateDtPositionsMapped = solver.shader.FindKernel("UpdateDtPositionsMapped");
            kRebuildParentsAtLayer = solver.shader.FindKernel("RebuildParentsAtLayer");
        }

        public struct ParentRebuildContext {
            public readonly INeighborSearch NeighborSearch;
            public readonly int BaseIndex;
            public readonly int ActiveCount;
            public readonly int FineCount;
            public readonly LayerSolve.DtMappingContext Mapping;

            public ParentRebuildContext(
                INeighborSearch neighborSearch,
                int baseIndex,
                int activeCount,
                int fineCount,
                LayerSolve.DtMappingContext mapping
            ) {
                NeighborSearch = neighborSearch;
                BaseIndex = baseIndex;
                ActiveCount = activeCount;
                FineCount = fineCount;
                Mapping = mapping;
            }
        }

        public void PrepareParentRebuildBuffers(in ParentRebuildContext context) {
            INeighborSearch neighborSearch = context.NeighborSearch;
            int baseIndex = context.BaseIndex;
            int activeCount = context.ActiveCount;
            int fineCount = context.FineCount;
            solver.asyncCb.SetComputeIntParam(solver.shader, "_Base", baseIndex);
            solver.asyncCb.SetComputeIntParam(solver.shader, "_DtNeighborCount", neighborSearch.NeighborCount);
            solver.asyncCb.SetComputeIntParam(solver.shader, "_ParentRangeStart", baseIndex + activeCount);
            solver.asyncCb.SetComputeIntParam(solver.shader, "_ParentRangeEnd", baseIndex + fineCount);
            solver.asyncCb.SetComputeIntParam(solver.shader, "_ParentCoarseCount", activeCount);

            solver.asyncCb.SetComputeBufferParam(solver.shader, kRebuildParentsAtLayer, "_Pos", solver.pos);
            solver.asyncCb.SetComputeBufferParam(solver.shader, kRebuildParentsAtLayer, "_ParentIndex", solver.parentIndex);
            solver.asyncCb.SetComputeBufferParam(solver.shader, kRebuildParentsAtLayer, "_ParentIndices", solver.parentIndices);
            solver.asyncCb.SetComputeBufferParam(solver.shader, kRebuildParentsAtLayer, "_ParentWeights", solver.parentWeights);
            solver.asyncCb.SetComputeBufferParam(solver.shader, kRebuildParentsAtLayer, "_DtNeighbors", neighborSearch.NeighborsBuffer);
            solver.asyncCb.SetComputeBufferParam(solver.shader, kRebuildParentsAtLayer, "_DtNeighborCounts", neighborSearch.NeighborCountsBuffer);
            solver.layerMappingCache.BindDtGlobalMappingParams(
                kRebuildParentsAtLayer,
                context.Mapping.UseDtGlobalNodeMap,
                context.Mapping.DtLocalBase,
                context.Mapping.DtGlobalNodeMap,
                context.Mapping.DtGlobalToLayerLocalMap);
        }

        public void PrepareIntegratePosParams() {
            solver.asyncCb.SetComputeBufferParam(solver.shader, kClampVelocities, "_Vel", solver.vel);
            solver.asyncCb.SetComputeBufferParam(solver.shader, kClampVelocities, "_InvMass", solver.invMass);
            solver.asyncCb.SetComputeBufferParam(solver.shader, kIntegratePositions, "_Pos", solver.pos);
            solver.asyncCb.SetComputeBufferParam(solver.shader, kIntegratePositions, "_Vel", solver.vel);
            solver.asyncCb.SetComputeBufferParam(solver.shader, kIntegratePositions, "_InvMass", solver.invMass);
        }

        internal void PrepareUpdateDtPosParamsMapped(DT dtLayer, ComputeBuffer dtGlobalNodeMap, ComputeBuffer dtGlobalToLayerLocalMap, int activeCount, float2 dtNormCenter, float dtNormInvHalfExtent, int pingWrite) {
            solver.asyncCb.SetComputeIntParam(solver.shader, "_ActiveCount", activeCount);
            solver.asyncCb.SetComputeIntParam(solver.shader, "_DtNeighborCount", dtLayer.NeighborCount);
            solver.asyncCb.SetComputeBufferParam(solver.shader, kUpdateDtPositionsMapped, "_Pos", solver.pos);
            solver.asyncCb.SetComputeBufferParam(solver.shader, kUpdateDtPositionsMapped, "_DtPositions", dtLayer.GetPositionsBuffer(pingWrite));
            solver.asyncCb.SetComputeBufferParam(solver.shader, kUpdateDtPositionsMapped, "_DtGlobalNodeMap", dtGlobalNodeMap);
            solver.layerMappingCache.BindDtGlobalMappingParams(kUpdateDtPositionsMapped, true, 0, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            solver.asyncCb.SetComputeVectorParam(solver.shader, "_DtNormCenter", new Vector4(dtNormCenter.x, dtNormCenter.y, 0f, 0f));
            solver.asyncCb.SetComputeFloatParam(solver.shader, "_DtNormInvHalfExtent", dtNormInvHalfExtent);
        }

        internal void PrepareUpdateDtPosParamsUnmapped(DT dtLayer, int baseIndex, int activeCount, float2 dtNormCenter, float dtNormInvHalfExtent, int pingWrite) {
            solver.asyncCb.SetComputeIntParam(solver.shader, "_Base", baseIndex);
            solver.asyncCb.SetComputeIntParam(solver.shader, "_ActiveCount", activeCount);
            solver.asyncCb.SetComputeIntParam(solver.shader, "_DtNeighborCount", dtLayer.NeighborCount);
            solver.asyncCb.SetComputeBufferParam(solver.shader, kUpdateDtPositions, "_Pos", solver.pos);
            solver.asyncCb.SetComputeBufferParam(solver.shader, kUpdateDtPositions, "_DtPositions", dtLayer.GetPositionsBuffer(pingWrite));
            solver.layerMappingCache.BindDtGlobalMappingParams(kUpdateDtPositions, false, 0, null, null);
            solver.asyncCb.SetComputeVectorParam(solver.shader, "_DtNormCenter", new Vector4(dtNormCenter.x, dtNormCenter.y, 0f, 0f));
            solver.asyncCb.SetComputeFloatParam(solver.shader, "_DtNormInvHalfExtent", dtNormInvHalfExtent);
        }
    }
}
