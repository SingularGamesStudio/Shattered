using Unity.Mathematics;
using UnityEngine;
using GPU.Neighbors;
using UnityEngine.Rendering;

namespace GPU.Solver {
	internal sealed partial class CollisionEvents {
		internal ComputeBuffer collisionEvents;
		private ComputeBuffer collisionEventCount;
		private ComputeBuffer xferColCount;
		private ComputeBuffer xferColNXBits;
		private ComputeBuffer xferColNYBits;
		private ComputeBuffer xferColPenBits;

		internal int kClearCollisionEventCount;
		internal int kBuildCollisionEventsL0;
		internal int kClearTransferredCollision;
		internal int kRestrictCollisionEventsToActivePairs;

		internal ComputeBuffer CollisionEventsBuffer => collisionEvents;
		internal ComputeBuffer CollisionEventCountBuffer => collisionEventCount;
		internal ComputeBuffer XferColCountBuffer => xferColCount;
		internal ComputeBuffer XferColNXBitsBuffer => xferColNXBits;
		internal ComputeBuffer XferColNYBitsBuffer => xferColNYBits;
		internal ComputeBuffer XferColPenBitsBuffer => xferColPenBits;
		internal int ClearCollisionEventCountKernel => kClearCollisionEventCount;
		internal int BuildCollisionEventsL0Kernel => kBuildCollisionEventsL0;
		internal int ClearTransferredCollisionKernel => kClearTransferredCollision;
		internal int RestrictCollisionEventsToActivePairsKernel => kRestrictCollisionEventsToActivePairs;

		internal void AllocateRuntimeBuffers(int newCapacity) {
			collisionEvents = new ComputeBuffer(math.max(1024, newCapacity * Const.NeighborCount), sizeof(uint) * 2 + sizeof(float) * 4, ComputeBufferType.Structured);
			collisionEventCount = new ComputeBuffer(1, sizeof(uint), ComputeBufferType.Structured);
			xferColCount = new ComputeBuffer(newCapacity * Const.NeighborCount, sizeof(uint), ComputeBufferType.Structured);
			xferColNXBits = new ComputeBuffer(newCapacity * Const.NeighborCount, sizeof(uint), ComputeBufferType.Structured);
			xferColNYBits = new ComputeBuffer(newCapacity * Const.NeighborCount, sizeof(uint), ComputeBufferType.Structured);
			xferColPenBits = new ComputeBuffer(newCapacity * Const.NeighborCount, sizeof(uint), ComputeBufferType.Structured);
		}

		internal void ReleaseRuntimeBuffers() {
			collisionEvents?.Dispose(); collisionEvents = null;
			collisionEventCount?.Dispose(); collisionEventCount = null;
			xferColCount?.Dispose(); xferColCount = null;
			xferColNXBits?.Dispose(); xferColNXBits = null;
			xferColNYBits?.Dispose(); xferColNYBits = null;
			xferColPenBits?.Dispose(); xferColPenBits = null;
		}

		internal void CacheRuntimeKernels() {
			kClearCollisionEventCount = shader.FindKernel("ClearCollisionEventCount");
			kBuildCollisionEventsL0 = shader.FindKernel("BuildCollisionEventsL0");
			kClearTransferredCollision = shader.FindKernel("ClearTransferredCollision");
			kRestrictCollisionEventsToActivePairs = shader.FindKernel("RestrictCollisionEventsToActivePairs");
		}

		private void PrepareLayer0BuildBuffers(
            CommandBuffer cb, 
            INeighborSearch layer0NeighborSearch,
            int layer0ActiveCount,
            float layer0KernelH,
            bool useDtGlobalNodeMap,
            ComputeBuffer dtGlobalNodeMap,
            ComputeBuffer dtGlobalToLayerLocalMap,
            ComputeBuffer dtOwnerByLocal
        ) {
            cb.SetComputeIntParam(shader, "_Base", 0);
            cb.SetComputeIntParam(shader, "_ActiveCount", layer0ActiveCount);
            cb.SetComputeIntParam(shader, "_DtNeighborCount", layer0NeighborSearch.NeighborCount);
            cb.SetComputeFloatParam(shader, "_LayerKernelH", layer0KernelH);
            cb.SetComputeIntParam(shader, "_UseDtOwnerFilter", dtOwnerByLocal != null ? 1 : 0);
            cb.SetComputeIntParam(shader, "_CollisionEventCapacity", collisionEvents != null ? collisionEvents.count : 0);

            cb.SetComputeBufferParam(shader, kClearCollisionEventCount, "_CollisionEventCount", collisionEventCount);

            cb.SetComputeBufferParam(shader, kBuildCollisionEventsL0, "_Pos", solver.pos);
			cb.SetComputeBufferParam(shader, kBuildCollisionEventsL0, "_Vel", solver.vel);
            cb.SetComputeBufferParam(shader, kBuildCollisionEventsL0, "_DtNeighbors", layer0NeighborSearch.NeighborsBuffer);
            cb.SetComputeBufferParam(shader, kBuildCollisionEventsL0, "_DtNeighborCounts", layer0NeighborSearch.NeighborCountsBuffer);
            cb.SetComputeBufferParam(shader, kBuildCollisionEventsL0, "_DtOwnerByLocal", dtOwnerByLocal ?? solver.layerMappingCache.DefaultDtOwnerByLocal);
			cb.SetComputeBufferParam(shader, kBuildCollisionEventsL0, "_DtCollisionOwnerByLocal", dtOwnerByLocal ?? solver.layerMappingCache.DefaultDtCollisionOwnerByLocal);
            cb.SetComputeBufferParam(shader, kBuildCollisionEventsL0, "_CollisionEvents", collisionEvents);
            cb.SetComputeBufferParam(shader, kBuildCollisionEventsL0, "_CollisionEventCount", collisionEventCount);

            cb.SetComputeBufferParam(shader, kClearTransferredCollision, "_XferColCount", xferColCount);
            cb.SetComputeBufferParam(shader, kClearTransferredCollision, "_XferColNXBits", xferColNXBits);
            cb.SetComputeBufferParam(shader, kClearTransferredCollision, "_XferColNYBits", xferColNYBits);
            cb.SetComputeBufferParam(shader, kClearTransferredCollision, "_XferColPenBits", xferColPenBits);

            cb.SetComputeBufferParam(shader, kRestrictCollisionEventsToActivePairs, "_ParentIndex", solver.parentIndex);
            cb.SetComputeBufferParam(shader, kRestrictCollisionEventsToActivePairs, "_ParentIndices", solver.parentIndices);
            cb.SetComputeBufferParam(shader, kRestrictCollisionEventsToActivePairs, "_ParentWeights", solver.parentWeights);
            cb.SetComputeBufferParam(shader, kRestrictCollisionEventsToActivePairs, "_DtNeighbors", layer0NeighborSearch.NeighborsBuffer);
            cb.SetComputeBufferParam(shader, kRestrictCollisionEventsToActivePairs, "_DtNeighborCounts", layer0NeighborSearch.NeighborCountsBuffer);
            cb.SetComputeBufferParam(shader, kRestrictCollisionEventsToActivePairs, "_CollisionEvents", collisionEvents);
            cb.SetComputeBufferParam(shader, kRestrictCollisionEventsToActivePairs, "_CollisionEventCount", collisionEventCount);
            cb.SetComputeBufferParam(shader, kRestrictCollisionEventsToActivePairs, "_XferColCount", xferColCount);
            cb.SetComputeBufferParam(shader, kRestrictCollisionEventsToActivePairs, "_XferColNXBits", xferColNXBits);
            cb.SetComputeBufferParam(shader, kRestrictCollisionEventsToActivePairs, "_XferColNYBits", xferColNYBits);
            cb.SetComputeBufferParam(shader, kRestrictCollisionEventsToActivePairs, "_XferColPenBits", xferColPenBits);

            solver.layerMappingCache.BindDtGlobalMappingParams(cb, shader, kBuildCollisionEventsL0, useDtGlobalNodeMap, 0, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            solver.layerMappingCache.BindDtGlobalMappingParams(cb, shader, kClearTransferredCollision, useDtGlobalNodeMap, 0, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            solver.layerMappingCache.BindDtGlobalMappingParams(cb, shader, kRestrictCollisionEventsToActivePairs, useDtGlobalNodeMap, 0, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
        }
	}
}
