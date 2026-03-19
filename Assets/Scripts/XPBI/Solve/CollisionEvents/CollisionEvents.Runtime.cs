using Unity.Mathematics;
using UnityEngine;

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
			kClearCollisionEventCount = solver.shader.FindKernel("ClearCollisionEventCount");
			kBuildCollisionEventsL0 = solver.shader.FindKernel("BuildCollisionEventsL0");
			kClearTransferredCollision = solver.shader.FindKernel("ClearTransferredCollision");
			kRestrictCollisionEventsToActivePairs = solver.shader.FindKernel("RestrictCollisionEventsToActivePairs");
		}
	}
}
