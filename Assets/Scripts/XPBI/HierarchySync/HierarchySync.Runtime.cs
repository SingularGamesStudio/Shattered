using GPU.Delaunay;
using GPU.Neighbors;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace GPU.Solver {
    internal sealed partial class HierarchySync {
        internal int kClampVelocities;
        internal int kIntegratePositions;
        internal int kUpdateDtPositions;
        internal int kUpdateDtPositionsMapped;
        internal int kRebuildParentsAtLayer;

        internal void CacheRuntimeKernels() {
            kClampVelocities = shader.FindKernel("ClampVelocities");
            kIntegratePositions = shader.FindKernel("IntegratePositions");
            kUpdateDtPositions = shader.FindKernel("UpdateDtPositions");
            kUpdateDtPositionsMapped = shader.FindKernel("UpdateDtPositionsMapped");
            kRebuildParentsAtLayer = shader.FindKernel("RebuildParentsAtLayer");
        }

        public struct ParentRebuildContext {
            public readonly INeighborSearch NeighborSearch;
            public readonly int BaseIndex;
            public readonly int ActiveCount;
            public readonly int FineCount;
            public readonly LayerCacheRuntime.DtMappingContext Mapping;

            public ParentRebuildContext(
                INeighborSearch neighborSearch,
                int baseIndex,
                int activeCount,
                int fineCount,
                LayerCacheRuntime.DtMappingContext mapping
            ) {
                NeighborSearch = neighborSearch;
                BaseIndex = baseIndex;
                ActiveCount = activeCount;
                FineCount = fineCount;
                Mapping = mapping;
            }
        }

        public void PrepareParentRebuildBuffers(CommandBuffer cb, ComputeBuffer pos, ComputeBuffer parentIndex, ComputeBuffer parentIndices, ComputeBuffer parentWeights, in ParentRebuildContext context) {
            INeighborSearch neighborSearch = context.NeighborSearch;
            int baseIndex = context.BaseIndex;
            int activeCount = context.ActiveCount;
            int fineCount = context.FineCount;
            cb.SetComputeIntParam(shader, "_Base", baseIndex);
            cb.SetComputeIntParam(shader, "_DtNeighborCount", neighborSearch.NeighborCount);
            cb.SetComputeIntParam(shader, "_ParentRangeStart", baseIndex + activeCount);
            cb.SetComputeIntParam(shader, "_ParentRangeEnd", baseIndex + fineCount);
            cb.SetComputeIntParam(shader, "_ParentCoarseCount", activeCount);

            cb.SetComputeBufferParam(shader, kRebuildParentsAtLayer, "_Pos", pos);
            cb.SetComputeBufferParam(shader, kRebuildParentsAtLayer, "_ParentIndex", parentIndex);
            cb.SetComputeBufferParam(shader, kRebuildParentsAtLayer, "_ParentIndices", parentIndices);
            cb.SetComputeBufferParam(shader, kRebuildParentsAtLayer, "_ParentWeights", parentWeights);
            cb.SetComputeBufferParam(shader, kRebuildParentsAtLayer, "_DtNeighbors", neighborSearch.NeighborsBuffer);
            cb.SetComputeBufferParam(shader, kRebuildParentsAtLayer, "_DtNeighborCounts", neighborSearch.NeighborCountsBuffer);
            solver.layerMappingCache.BindDtGlobalMappingParams(
                cb,
                shader,
                kRebuildParentsAtLayer,
                context.Mapping.UseDtGlobalNodeMap,
                context.Mapping.DtLocalBase,
                context.Mapping.DtGlobalNodeMap,
                context.Mapping.DtGlobalToLayerLocalMap);
        }

        public void PrepareIntegratePosParams(CommandBuffer cb, ComputeBuffer pos, ComputeBuffer vel, ComputeBuffer invMass) {
            cb.SetComputeBufferParam(shader, kClampVelocities, "_Vel", vel);
            cb.SetComputeBufferParam(shader, kClampVelocities, "_InvMass", invMass);
            cb.SetComputeBufferParam(shader, kIntegratePositions, "_Pos", pos);
            cb.SetComputeBufferParam(shader, kIntegratePositions, "_Vel", vel);
            cb.SetComputeBufferParam(shader, kIntegratePositions, "_InvMass", invMass);
        }

        internal void PrepareUpdateDtPosParamsMapped(CommandBuffer cb, ComputeBuffer pos, DT dtLayer, ComputeBuffer dtGlobalNodeMap, ComputeBuffer dtGlobalToLayerLocalMap, int activeCount, float2 dtNormCenter, float dtNormInvHalfExtent, int pingWrite) {
            cb.SetComputeIntParam(shader, "_ActiveCount", activeCount);
            cb.SetComputeIntParam(shader, "_DtNeighborCount", dtLayer.NeighborCount);
            cb.SetComputeBufferParam(shader, kUpdateDtPositionsMapped, "_Pos", pos);
            cb.SetComputeBufferParam(shader, kUpdateDtPositionsMapped, "_DtPositions", dtLayer.GetPositionsBuffer(pingWrite));
            cb.SetComputeBufferParam(shader, kUpdateDtPositionsMapped, "_DtGlobalNodeMap", dtGlobalNodeMap);
            solver.layerMappingCache.BindDtGlobalMappingParams(cb, shader, kUpdateDtPositionsMapped, true, 0, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            cb.SetComputeVectorParam(shader, "_DtNormCenter", new Vector4(dtNormCenter.x, dtNormCenter.y, 0f, 0f));
            cb.SetComputeFloatParam(shader, "_DtNormInvHalfExtent", dtNormInvHalfExtent);
        }

        internal void PrepareUpdateDtPosParamsUnmapped(CommandBuffer cb, ComputeBuffer pos, DT dtLayer, int baseIndex, int activeCount, float2 dtNormCenter, float dtNormInvHalfExtent, int pingWrite) {
            cb.SetComputeIntParam(shader, "_Base", baseIndex);
            cb.SetComputeIntParam(shader, "_ActiveCount", activeCount);
            cb.SetComputeIntParam(shader, "_DtNeighborCount", dtLayer.NeighborCount);
            cb.SetComputeBufferParam(shader, kUpdateDtPositions, "_Pos", pos);
            cb.SetComputeBufferParam(shader, kUpdateDtPositions, "_DtPositions", dtLayer.GetPositionsBuffer(pingWrite));
            solver.layerMappingCache.BindDtGlobalMappingParams(cb, shader, kUpdateDtPositions, false, 0, null, null);
            cb.SetComputeVectorParam(shader, "_DtNormCenter", new Vector4(dtNormCenter.x, dtNormCenter.y, 0f, 0f));
            cb.SetComputeFloatParam(shader, "_DtNormInvHalfExtent", dtNormInvHalfExtent);
        }
    }
}
