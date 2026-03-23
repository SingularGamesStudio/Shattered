using Unity.Mathematics;
using UnityEngine;
using GPU.Neighbors;
using GPU.Delaunay;
using UnityEngine.Rendering;

namespace GPU.Solver {
	internal sealed partial class CollisionEvents {
		internal ComputeBuffer collisionEvents;
		private ComputeBuffer collisionEventCount;
		private ComputeBuffer xferColCount;
		private ComputeBuffer xferColNXBits;
		private ComputeBuffer xferColNYBits;
		private ComputeBuffer xferColPenBits;
		private ComputeBuffer boundaryChunkCount;
		private ComputeBuffer boundaryChunkEdges;
		private ComputeBuffer boundaryChunkAabbs;
		private ComputeBuffer boundaryChunkMortonKeys;
		private ComputeBuffer boundaryChunkSortKeys;
		private ComputeBuffer boundaryChunkSortIndices;
		private ComputeBuffer lbvhNodeAabbs;
		private ComputeBuffer lbvhNodeChunk;
		private ComputeBuffer collisionDebugStats;

		private int boundaryChunkCapacity;
		private int boundaryChunkSortCapacity;
		private int lbvhLeafOffset;
		private int lbvhNodeCapacity;

		internal int kClearCollisionEventCount;
		internal int kClearBoundaryChunkCount;
		internal int kBuildBoundaryChunksL0;
		internal int kInitChunkSortKeys;
		internal int kBitonicSortChunkKeys;
		internal int kBuildLbvhLeaves;
		internal int kBuildLbvhInternalLevel;
		internal int kTraverseLbvhEmitCollisionEvents;
		internal int kClearCollisionDebugStats;
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
		internal int BoundaryChunkSortCapacity => boundaryChunkSortCapacity;
		internal int BoundaryChunkCapacity => boundaryChunkCapacity;
		internal int LbvhLeafOffset => lbvhLeafOffset;
		internal int LbvhNodeCapacity => lbvhNodeCapacity;
		internal ComputeBuffer CollisionDebugStatsBuffer => collisionDebugStats;

		private static int NextPow2(int v) {
			if (v <= 1)
				return 1;

			v--;
			v |= v >> 1;
			v |= v >> 2;
			v |= v >> 4;
			v |= v >> 8;
			v |= v >> 16;
			return v + 1;
		}

		internal void AllocateRuntimeBuffers(int newCapacity) {
			int collisionCapacity = math.max(4096, newCapacity * 32);
			boundaryChunkCapacity = math.max(2048, newCapacity * 12);
			boundaryChunkSortCapacity = NextPow2(boundaryChunkCapacity);
			lbvhLeafOffset = boundaryChunkSortCapacity - 1;
			lbvhNodeCapacity = boundaryChunkSortCapacity * 2 - 1;

			collisionEvents = new ComputeBuffer(collisionCapacity, sizeof(uint) * 2 + sizeof(float) * 4, ComputeBufferType.Structured);
			collisionEventCount = new ComputeBuffer(1, sizeof(uint), ComputeBufferType.Structured);
			xferColCount = new ComputeBuffer(newCapacity * Const.NeighborCount, sizeof(uint), ComputeBufferType.Structured);
			xferColNXBits = new ComputeBuffer(newCapacity * Const.NeighborCount, sizeof(uint), ComputeBufferType.Structured);
			xferColNYBits = new ComputeBuffer(newCapacity * Const.NeighborCount, sizeof(uint), ComputeBufferType.Structured);
			xferColPenBits = new ComputeBuffer(newCapacity * Const.NeighborCount, sizeof(uint), ComputeBufferType.Structured);
			boundaryChunkCount = new ComputeBuffer(1, sizeof(uint), ComputeBufferType.Structured);
			boundaryChunkEdges = new ComputeBuffer(boundaryChunkCapacity, sizeof(uint) * 3 + sizeof(int), ComputeBufferType.Structured);
			boundaryChunkAabbs = new ComputeBuffer(boundaryChunkCapacity, sizeof(float) * 4, ComputeBufferType.Structured);
			boundaryChunkMortonKeys = new ComputeBuffer(boundaryChunkCapacity, sizeof(uint), ComputeBufferType.Structured);
			boundaryChunkSortKeys = new ComputeBuffer(boundaryChunkSortCapacity, sizeof(uint), ComputeBufferType.Structured);
			boundaryChunkSortIndices = new ComputeBuffer(boundaryChunkSortCapacity, sizeof(uint), ComputeBufferType.Structured);
			lbvhNodeAabbs = new ComputeBuffer(lbvhNodeCapacity, sizeof(float) * 4, ComputeBufferType.Structured);
			lbvhNodeChunk = new ComputeBuffer(lbvhNodeCapacity, sizeof(int), ComputeBufferType.Structured);
			collisionDebugStats = new ComputeBuffer(16, sizeof(uint), ComputeBufferType.Structured);
		}

		internal void ReleaseRuntimeBuffers() {
			collisionEvents?.Dispose(); collisionEvents = null;
			collisionEventCount?.Dispose(); collisionEventCount = null;
			xferColCount?.Dispose(); xferColCount = null;
			xferColNXBits?.Dispose(); xferColNXBits = null;
			xferColNYBits?.Dispose(); xferColNYBits = null;
			xferColPenBits?.Dispose(); xferColPenBits = null;
			boundaryChunkCount?.Dispose(); boundaryChunkCount = null;
			boundaryChunkEdges?.Dispose(); boundaryChunkEdges = null;
			boundaryChunkAabbs?.Dispose(); boundaryChunkAabbs = null;
			boundaryChunkMortonKeys?.Dispose(); boundaryChunkMortonKeys = null;
			boundaryChunkSortKeys?.Dispose(); boundaryChunkSortKeys = null;
			boundaryChunkSortIndices?.Dispose(); boundaryChunkSortIndices = null;
			lbvhNodeAabbs?.Dispose(); lbvhNodeAabbs = null;
			lbvhNodeChunk?.Dispose(); lbvhNodeChunk = null;
			collisionDebugStats?.Dispose(); collisionDebugStats = null;
			boundaryChunkCapacity = 0;
			boundaryChunkSortCapacity = 0;
			lbvhLeafOffset = 0;
			lbvhNodeCapacity = 0;
		}

		internal void CacheRuntimeKernels() {
			kClearCollisionEventCount = shader.FindKernel("ClearCollisionEventCount");
			kClearBoundaryChunkCount = shader.FindKernel("ClearBoundaryChunkCount");
			kBuildBoundaryChunksL0 = shader.FindKernel("BuildBoundaryChunksL0");
			kInitChunkSortKeys = shader.FindKernel("InitChunkSortKeys");
			kBitonicSortChunkKeys = shader.FindKernel("BitonicSortChunkKeys");
			kBuildLbvhLeaves = shader.FindKernel("BuildLbvhLeaves");
			kBuildLbvhInternalLevel = shader.FindKernel("BuildLbvhInternalLevel");
			kTraverseLbvhEmitCollisionEvents = shader.FindKernel("TraverseLbvhEmitCollisionEvents");
			kClearCollisionDebugStats = shader.FindKernel("ClearCollisionDebugStats");
			kBuildCollisionEventsL0 = shader.FindKernel("BuildCollisionEventsL0");
			kClearTransferredCollision = shader.FindKernel("ClearTransferredCollision");
			kRestrictCollisionEventsToActivePairs = shader.FindKernel("RestrictCollisionEventsToActivePairs");
		}

		private void PrepareLayer0BuildBuffers(
            CommandBuffer cb, 
            INeighborSearch layer0NeighborSearch,
            DT layer0Dt,
			int dtReadSlot,
            int layer0ActiveCount,
            float layer0KernelH,
            float2 layer0BoundsMin,
            float2 layer0BoundsMax,
            bool useDtGlobalNodeMap,
            ComputeBuffer dtGlobalNodeMap,
            ComputeBuffer dtGlobalToLayerLocalMap,
            ComputeBuffer dtOwnerByLocal
        ) {
			if (layer0Dt == null)
				return;

            cb.SetComputeIntParam(shader, "_Base", 0);
            cb.SetComputeIntParam(shader, "_ActiveCount", layer0ActiveCount);
            cb.SetComputeIntParam(shader, "_DtNeighborCount", layer0NeighborSearch.NeighborCount);
            cb.SetComputeFloatParam(shader, "_LayerKernelH", layer0KernelH);
            cb.SetComputeIntParam(shader, "_UseDtOwnerFilter", dtOwnerByLocal != null ? 1 : 0);
            cb.SetComputeIntParam(shader, "_CollisionEventCapacity", collisionEvents != null ? collisionEvents.count : 0);
            cb.SetComputeIntParam(shader, "_DtHalfEdgeCount", layer0Dt.HalfEdgeCount);
            cb.SetComputeIntParam(shader, "_BoundaryChunkCapacity", boundaryChunkCapacity);
            cb.SetComputeIntParam(shader, "_BoundaryChunkSortCapacity", boundaryChunkSortCapacity);
            cb.SetComputeIntParam(shader, "_LbvhLeafOffset", lbvhLeafOffset);
            cb.SetComputeIntParam(shader, "_LbvhNodeCount", lbvhNodeCapacity);
            cb.SetComputeVectorParam(shader, "_LbvhBoundsMin", (Vector2)layer0BoundsMin);
            cb.SetComputeVectorParam(shader, "_LbvhBoundsMax", (Vector2)layer0BoundsMax);

            cb.SetComputeBufferParam(shader, kClearCollisionEventCount, "_CollisionEventCount", collisionEventCount);
            cb.SetComputeBufferParam(shader, kClearBoundaryChunkCount, "_BoundaryChunkCount", boundaryChunkCount);
			cb.SetComputeBufferParam(shader, kClearCollisionDebugStats, "_CollisionDebugStats", collisionDebugStats);

            cb.SetComputeBufferParam(shader, kBuildCollisionEventsL0, "_Pos", solver.pos);
			cb.SetComputeBufferParam(shader, kBuildCollisionEventsL0, "_Vel", solver.vel);
            cb.SetComputeBufferParam(shader, kBuildCollisionEventsL0, "_DtNeighbors", layer0NeighborSearch.NeighborsBuffer);
            cb.SetComputeBufferParam(shader, kBuildCollisionEventsL0, "_DtNeighborCounts", layer0NeighborSearch.NeighborCountsBuffer);
            cb.SetComputeBufferParam(shader, kBuildCollisionEventsL0, "_DtOwnerByLocal", dtOwnerByLocal ?? solver.layerMappingCache.DefaultDtOwnerByLocal);
			cb.SetComputeBufferParam(shader, kBuildCollisionEventsL0, "_DtCollisionOwnerByLocal", dtOwnerByLocal ?? solver.layerMappingCache.DefaultDtCollisionOwnerByLocal);
            cb.SetComputeBufferParam(shader, kBuildCollisionEventsL0, "_CollisionEvents", collisionEvents);
            cb.SetComputeBufferParam(shader, kBuildCollisionEventsL0, "_CollisionEventCount", collisionEventCount);
            cb.SetComputeBufferParam(shader, kBuildCollisionEventsL0, "_BoundaryChunkCount", boundaryChunkCount);
            cb.SetComputeBufferParam(shader, kBuildCollisionEventsL0, "_BoundaryChunks", boundaryChunkEdges);
            cb.SetComputeBufferParam(shader, kBuildCollisionEventsL0, "_BoundaryChunkAabbs", boundaryChunkAabbs);
            cb.SetComputeBufferParam(shader, kBuildCollisionEventsL0, "_BoundaryChunkSortIndices", boundaryChunkSortIndices);
            cb.SetComputeBufferParam(shader, kBuildCollisionEventsL0, "_LbvhNodeAabbs", lbvhNodeAabbs);
            cb.SetComputeBufferParam(shader, kBuildCollisionEventsL0, "_LbvhNodeChunk", lbvhNodeChunk);

            cb.SetComputeBufferParam(shader, kBuildBoundaryChunksL0, "_Pos", solver.pos);
			cb.SetComputeBufferParam(shader, kBuildBoundaryChunksL0, "_Vel", solver.vel);
            cb.SetComputeBufferParam(shader, kBuildBoundaryChunksL0, "_DtOwnerByLocal", dtOwnerByLocal ?? solver.layerMappingCache.DefaultDtOwnerByLocal);
			cb.SetComputeBufferParam(shader, kBuildBoundaryChunksL0, "_DtCollisionOwnerByLocal", dtOwnerByLocal ?? solver.layerMappingCache.DefaultDtCollisionOwnerByLocal);
			cb.SetComputeBufferParam(shader, kBuildBoundaryChunksL0, "_DtHalfEdges", layer0Dt.GetHalfEdgesBuffer(dtReadSlot));
            cb.SetComputeBufferParam(shader, kBuildBoundaryChunksL0, "_DtBoundaryEdgeFlags", layer0Dt.BoundaryEdgeFlagsBuffer);
            cb.SetComputeBufferParam(shader, kBuildBoundaryChunksL0, "_BoundaryChunkCount", boundaryChunkCount);
            cb.SetComputeBufferParam(shader, kBuildBoundaryChunksL0, "_BoundaryChunks", boundaryChunkEdges);
            cb.SetComputeBufferParam(shader, kBuildBoundaryChunksL0, "_BoundaryChunkAabbs", boundaryChunkAabbs);
            cb.SetComputeBufferParam(shader, kBuildBoundaryChunksL0, "_BoundaryChunkMortonKeys", boundaryChunkMortonKeys);
			cb.SetComputeBufferParam(shader, kBuildBoundaryChunksL0, "_CollisionDebugStats", collisionDebugStats);

            cb.SetComputeBufferParam(shader, kInitChunkSortKeys, "_BoundaryChunkCount", boundaryChunkCount);
            cb.SetComputeBufferParam(shader, kInitChunkSortKeys, "_BoundaryChunkMortonKeys", boundaryChunkMortonKeys);
            cb.SetComputeBufferParam(shader, kInitChunkSortKeys, "_BoundaryChunkSortKeys", boundaryChunkSortKeys);
            cb.SetComputeBufferParam(shader, kInitChunkSortKeys, "_BoundaryChunkSortIndices", boundaryChunkSortIndices);

            cb.SetComputeBufferParam(shader, kBitonicSortChunkKeys, "_BoundaryChunkSortKeys", boundaryChunkSortKeys);
            cb.SetComputeBufferParam(shader, kBitonicSortChunkKeys, "_BoundaryChunkSortIndices", boundaryChunkSortIndices);

            cb.SetComputeBufferParam(shader, kBuildLbvhLeaves, "_BoundaryChunkCount", boundaryChunkCount);
            cb.SetComputeBufferParam(shader, kBuildLbvhLeaves, "_BoundaryChunkSortIndices", boundaryChunkSortIndices);
            cb.SetComputeBufferParam(shader, kBuildLbvhLeaves, "_BoundaryChunkAabbs", boundaryChunkAabbs);
            cb.SetComputeBufferParam(shader, kBuildLbvhLeaves, "_LbvhNodeAabbs", lbvhNodeAabbs);
            cb.SetComputeBufferParam(shader, kBuildLbvhLeaves, "_LbvhNodeChunk", lbvhNodeChunk);

            cb.SetComputeBufferParam(shader, kBuildLbvhInternalLevel, "_LbvhNodeAabbs", lbvhNodeAabbs);
            cb.SetComputeBufferParam(shader, kBuildLbvhInternalLevel, "_LbvhNodeChunk", lbvhNodeChunk);

            cb.SetComputeBufferParam(shader, kTraverseLbvhEmitCollisionEvents, "_Pos", solver.pos);
			cb.SetComputeBufferParam(shader, kTraverseLbvhEmitCollisionEvents, "_Vel", solver.vel);
            cb.SetComputeBufferParam(shader, kTraverseLbvhEmitCollisionEvents, "_DtCollisionOwnerByLocal", dtOwnerByLocal ?? solver.layerMappingCache.DefaultDtCollisionOwnerByLocal);
            cb.SetComputeBufferParam(shader, kTraverseLbvhEmitCollisionEvents, "_BoundaryChunkCount", boundaryChunkCount);
            cb.SetComputeBufferParam(shader, kTraverseLbvhEmitCollisionEvents, "_BoundaryChunks", boundaryChunkEdges);
            cb.SetComputeBufferParam(shader, kTraverseLbvhEmitCollisionEvents, "_BoundaryChunkAabbs", boundaryChunkAabbs);
            cb.SetComputeBufferParam(shader, kTraverseLbvhEmitCollisionEvents, "_BoundaryChunkSortIndices", boundaryChunkSortIndices);
            cb.SetComputeBufferParam(shader, kTraverseLbvhEmitCollisionEvents, "_LbvhNodeAabbs", lbvhNodeAabbs);
            cb.SetComputeBufferParam(shader, kTraverseLbvhEmitCollisionEvents, "_LbvhNodeChunk", lbvhNodeChunk);
            cb.SetComputeBufferParam(shader, kTraverseLbvhEmitCollisionEvents, "_CollisionEvents", collisionEvents);
            cb.SetComputeBufferParam(shader, kTraverseLbvhEmitCollisionEvents, "_CollisionEventCount", collisionEventCount);
			cb.SetComputeBufferParam(shader, kTraverseLbvhEmitCollisionEvents, "_CollisionDebugStats", collisionDebugStats);

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

			solver.layerMappingCache.BindDtGlobalMappingParams(cb, shader, kBuildBoundaryChunksL0, useDtGlobalNodeMap, 0, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
			solver.layerMappingCache.BindDtGlobalMappingParams(cb, shader, kInitChunkSortKeys, useDtGlobalNodeMap, 0, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
			solver.layerMappingCache.BindDtGlobalMappingParams(cb, shader, kBitonicSortChunkKeys, useDtGlobalNodeMap, 0, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
			solver.layerMappingCache.BindDtGlobalMappingParams(cb, shader, kBuildLbvhLeaves, useDtGlobalNodeMap, 0, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
			solver.layerMappingCache.BindDtGlobalMappingParams(cb, shader, kBuildLbvhInternalLevel, useDtGlobalNodeMap, 0, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
			solver.layerMappingCache.BindDtGlobalMappingParams(cb, shader, kTraverseLbvhEmitCollisionEvents, useDtGlobalNodeMap, 0, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            solver.layerMappingCache.BindDtGlobalMappingParams(cb, shader, kBuildCollisionEventsL0, useDtGlobalNodeMap, 0, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            solver.layerMappingCache.BindDtGlobalMappingParams(cb, shader, kClearTransferredCollision, useDtGlobalNodeMap, 0, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            solver.layerMappingCache.BindDtGlobalMappingParams(cb, shader, kRestrictCollisionEventsToActivePairs, useDtGlobalNodeMap, 0, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
        }
	}
}
