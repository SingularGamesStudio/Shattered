using System;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace GPU.Neighbors {
    public interface INeighborSearch : IDisposable {
        int NeighborCount { get; }
        ComputeBuffer NeighborsBuffer { get; }
        ComputeBuffer NeighborCountsBuffer { get; }
        ComputeBuffer DirtyVertexFlagsBuffer { get; }

        void MarkAllDirty(CommandBuffer cb);

        // Rebuild/refresh neighbor buffers for the current positions.
        void EnqueueBuild(
            CommandBuffer cb,
            ComputeBuffer positions,  // float2 positions in world space
            int realVertexCount,
            float cellSize,
            float supportRadius,
            float2 boundsMin,
            float2 boundsMax,
            int readSlot,
            int writeSlot,
            int fixIterations,
            int legalizeIterations,
            bool rebuildAdjacencyAndTriMap = true);
    }
}
