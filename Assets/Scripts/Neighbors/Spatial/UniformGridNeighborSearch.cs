using System;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace GPU.Neighbors {
    // Paper-faithful: uniform grid (Δx = k = 2r), rebuild each timestep, query adjacent cells. [web:9]
    public sealed class UniformGridNeighborSearch : INeighborSearch {
        private const string MarkerPrefix = "XPBI.UniformGrid.";

        private readonly ComputeShader _shader;
        private readonly bool _ownsShaderInstance;

        private readonly int _kClearHeads;
        private readonly int _kBuildLists;
        private readonly int _kClearCounts;
        private readonly int _kBuildNeighbors;

        private ComputeBuffer _cellHeads;   // int[cellCount], -1 empty
        private ComputeBuffer _next;        // int[N], linked-list next
        private ComputeBuffer _neighbors;   // int[N * maxNeighbors]
        private ComputeBuffer _counts;      // int[N]

        private int _maxNeighbors;
        private int _capacityN;
        private int _cellCount;

        public int NeighborCount => _maxNeighbors;
        public ComputeBuffer NeighborsBuffer => _neighbors;
        public ComputeBuffer NeighborCountsBuffer => _counts;
        public ComputeBuffer DirtyVertexFlagsBuffer => null;

        public UniformGridNeighborSearch(ComputeShader shader, int maxNeighbors) {
            if (!shader) throw new ArgumentNullException(nameof(shader));
            if (maxNeighbors <= 0) throw new ArgumentOutOfRangeException(nameof(maxNeighbors));

            _shader = UnityEngine.Object.Instantiate(shader);
            _ownsShaderInstance = true;

            _maxNeighbors = maxNeighbors;

            _kClearHeads = _shader.FindKernel("ClearHeads");
            _kBuildLists = _shader.FindKernel("BuildLists");
            _kClearCounts = _shader.FindKernel("ClearCounts");
            _kBuildNeighbors = _shader.FindKernel("BuildNeighbors");
        }

        public void EnsureCapacity(int n, int gridW, int gridH) {
            if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
            if (gridW <= 0 || gridH <= 0) throw new ArgumentOutOfRangeException("grid dims");
            int cellCount = gridW * gridH;

            bool needRealloc = _neighbors == null || n > _capacityN || cellCount != _cellCount;
            if (!needRealloc) return;

            DisposeBuffers();

            _capacityN = n;
            _cellCount = cellCount;

            _cellHeads = new ComputeBuffer(_cellCount, sizeof(int), ComputeBufferType.Structured);
            _next = new ComputeBuffer(_capacityN, sizeof(int), ComputeBufferType.Structured);
            _neighbors = new ComputeBuffer(_capacityN * _maxNeighbors, sizeof(int), ComputeBufferType.Structured);
            _counts = new ComputeBuffer(_capacityN, sizeof(int), ComputeBufferType.Structured);
        }

        public void MarkAllDirty(CommandBuffer cb) {
        }

        private static void Dispatch(CommandBuffer cb, ComputeShader shader, int kernel, int x, int y, int z, string marker) {
            cb.BeginSample(marker);
            cb.DispatchCompute(shader, kernel, x, y, z);
            cb.EndSample(marker);
        }

        public void EnqueueBuild(
            CommandBuffer cb,
            ComputeBuffer positions,
            int realVertexCount,
            float cellSize,
            float supportRadius,
            float2 boundsMin,
            float2 boundsMax,
            int readSlot,
            int writeSlot,
            int fixIterations,
            int legalizeIterations,
            bool rebuildAdjacencyAndTriMap = true) {

            if (cb == null) throw new ArgumentNullException(nameof(cb));
            if (positions == null) throw new ArgumentNullException(nameof(positions));
            if (realVertexCount <= 0) throw new ArgumentOutOfRangeException(nameof(realVertexCount));
            if (cellSize <= 0f) throw new ArgumentOutOfRangeException(nameof(cellSize));
            if (supportRadius <= 0) throw new ArgumentOutOfRangeException(nameof(supportRadius));

            float2 bmin = boundsMin;
            float2 bmax = boundsMax;
            if (bmax.x <= bmin.x || bmax.y <= bmin.y)
                throw new ArgumentException("Invalid bounds for uniform grid neighbor search.");

            int gridW = math.max(1, (int)math.ceil((bmax.x - bmin.x) / cellSize));
            int gridH = math.max(1, (int)math.ceil((bmax.y - bmin.y) / cellSize));

            EnsureCapacity(realVertexCount, gridW, gridH);

            cb.SetComputeIntParam(_shader, "_N", realVertexCount);
            cb.SetComputeIntParam(_shader, "_MaxNeighbors", _maxNeighbors);
            cb.SetComputeIntParam(_shader, "_GridW", gridW);
            cb.SetComputeIntParam(_shader, "_GridH", gridH);
            cb.SetComputeVectorParam(_shader, "_BoundsMin", new Vector4(bmin.x, bmin.y, 0, 0));
            cb.SetComputeFloatParam(_shader, "_InvCellSize", 1.0f / cellSize);
            cb.SetComputeFloatParam(_shader, "_SupportRadius2", supportRadius * supportRadius);

            cb.SetComputeBufferParam(_shader, _kClearHeads, "_CellHeads", _cellHeads);
            cb.SetComputeBufferParam(_shader, _kBuildLists, "_CellHeads", _cellHeads);
            cb.SetComputeBufferParam(_shader, _kBuildLists, "_Next", _next);
            cb.SetComputeBufferParam(_shader, _kBuildLists, "_Positions", positions);

            cb.SetComputeBufferParam(_shader, _kClearCounts, "_Counts", _counts);

            cb.SetComputeBufferParam(_shader, _kBuildNeighbors, "_CellHeads", _cellHeads);
            cb.SetComputeBufferParam(_shader, _kBuildNeighbors, "_Next", _next);
            cb.SetComputeBufferParam(_shader, _kBuildNeighbors, "_Positions", positions);
            cb.SetComputeBufferParam(_shader, _kBuildNeighbors, "_Neighbors", _neighbors);
            cb.SetComputeBufferParam(_shader, _kBuildNeighbors, "_Counts", _counts);

            Dispatch(cb, _shader, _kClearHeads, (_cellCount + 255) / 256, 1, 1, MarkerPrefix + "ClearHeads");
            Dispatch(cb, _shader, _kClearCounts, (realVertexCount + 255) / 256, 1, 1, MarkerPrefix + "ClearCounts");
            Dispatch(cb, _shader, _kBuildLists, (realVertexCount + 255) / 256, 1, 1, MarkerPrefix + "BuildLists");
            Dispatch(cb, _shader, _kBuildNeighbors, (realVertexCount + 255) / 256, 1, 1, MarkerPrefix + "BuildNeighbors");
        }

        public void Dispose() {
            DisposeBuffers();
            if (_ownsShaderInstance && _shader) UnityEngine.Object.Destroy(_shader);
        }

        private void DisposeBuffers() {
            _cellHeads?.Dispose(); _cellHeads = null;
            _next?.Dispose(); _next = null;
            _neighbors?.Dispose(); _neighbors = null;
            _counts?.Dispose(); _counts = null;
            _capacityN = 0;
            _cellCount = 0;
        }
    }
}
