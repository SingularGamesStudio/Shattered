using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace GPU.Delaunay {
    /// <summary>
    /// Manages the GPU‑based Delaunay triangulation.
    /// Provides triple‑buffered geometry data for rendering and a set of shared working buffers.
    /// </summary>
    public sealed class DT : IDisposable {
        //---------------------------------------------------------------------------
        // Shader and kernel IDs
        //---------------------------------------------------------------------------
        private readonly ComputeShader _shader;
        private readonly bool _ownsShaderInstance;

        private readonly int _kernelClearTriLocks;
        private readonly int _kernelClearVertexToEdge;
        private readonly int _kernelBuildVertexToEdge;
        private readonly int _kernelBuildNeighbors;
        private readonly int _kernelFixHalfEdges;
        private readonly int _kernelLegalizeHalfEdges;
        private readonly int _kernelClearTriToHE;
        private readonly int _kernelBuildRenderableTriToHE;
        private readonly int _kernelClearDirtyVertexFlags;
        private readonly int _kernelMarkAllDirty;
        private readonly int _kernelCopyHalfEdges;

        //---------------------------------------------------------------------------
        // Buffers – triple buffered for rendering, single working buffers for topology ops
        //---------------------------------------------------------------------------
        private readonly ComputeBuffer[] _positions = new ComputeBuffer[3];      // vertex positions (read‑only in kernels)
        private readonly ComputeBuffer[] _halfEdges = new ComputeBuffer[3];      // half‑edge arrays (modified during flips)
        private readonly ComputeBuffer[] _triToHE = new ComputeBuffer[3];        // triangle → representative half‑edge (for rendering)

        private ComputeBuffer _triLocks;           // per‑triangle lock (0 = free)
        private ComputeBuffer _vToE;                // vertex → outgoing half‑edge
        private ComputeBuffer _neighbors;           // flat neighbour list per real vertex
        private ComputeBuffer _neighborCounts;      // number of neighbours per real vertex
        private ComputeBuffer _flipCount;           // global flip counter (single uint)
        private ComputeBuffer _dirtyVertexFlags;    // per‑real‑vertex dirty flag (1 if neighbour list needs rebuild)

        //---------------------------------------------------------------------------
        // Scratch data
        //---------------------------------------------------------------------------
        private float2[] _positionScratch;
        private readonly uint[] _flipScratch = { 0u };

        //---------------------------------------------------------------------------
        // Sizes
        //---------------------------------------------------------------------------
        private int _vertexCount;           // total vertices (real + ghost)
        private int _realVertexCount;       // number of real vertices (0 … _realVertexCount-1)
        private int _halfEdgeCount;          // number of allocated half‑edges
        private int _triCount;               // number of triangles
        private int _neighborCount;           // max neighbours per vertex (size of neighbour arrays)

        private int _renderSlot;              // current slot used for rendering (0,1,2)

        //---------------------------------------------------------------------------
        // Public properties
        //---------------------------------------------------------------------------
        public int HalfEdgeCount => _halfEdgeCount;
        public int TriCount => _triCount;
        public int NeighborCount => _neighborCount;

        /// <summary>Buffer of neighbour indices (flat, length = _realVertexCount * _neighborCount).</summary>
        public ComputeBuffer NeighborsBuffer => _neighbors;

        /// <summary>Buffer of neighbour counts per vertex.</summary>
        public ComputeBuffer NeighborCountsBuffer => _neighborCounts;

        /// <summary>Dirty flags per real vertex (1 = neighbour list outdated).</summary>
        public ComputeBuffer DirtyVertexFlagsBuffer => _dirtyVertexFlags;

        /// <summary>Positions buffer for the current render slot.</summary>
        public ComputeBuffer PositionsBuffer => _positions[_renderSlot];

        /// <summary>Half‑edges buffer for the current render slot.</summary>
        public ComputeBuffer HalfEdgesBuffer => _halfEdges[_renderSlot];

        /// <summary>Triangle‑to‑halfedge map for the current render slot.</summary>
        public ComputeBuffer TriToHEBuffer => _triToHE[_renderSlot];

        public ComputeBuffer GetPositionsBuffer(int slot) => _positions[slot];
        public ComputeBuffer GetHalfEdgesBuffer(int slot) => _halfEdges[slot];
        public ComputeBuffer GetTriToHEBuffer(int slot) => _triToHE[slot];

        //---------------------------------------------------------------------------
        // Construction
        //---------------------------------------------------------------------------
        /// <summary>
        /// Creates a new DT instance using a copy of the given compute shader.
        /// </summary>
        /// <param name="shader">Compute shader containing all required kernels.</param>
        public DT(ComputeShader shader) {
            if (!shader) throw new ArgumentNullException(nameof(shader));

            // Instantiate a copy so that multiple DT instances can run independently.
            _shader = UnityEngine.Object.Instantiate(shader);
            _ownsShaderInstance = true;

            // Cache kernel indices for faster dispatch.
            _kernelClearTriLocks = _shader.FindKernel("ClearTriLocks");
            _kernelClearVertexToEdge = _shader.FindKernel("ClearVertexToEdge");
            _kernelBuildVertexToEdge = _shader.FindKernel("BuildVertexToEdge");
            _kernelBuildNeighbors = _shader.FindKernel("BuildNeighbors");
            _kernelFixHalfEdges = _shader.FindKernel("FixHalfEdges");
            _kernelLegalizeHalfEdges = _shader.FindKernel("LegalizeHalfEdges");
            _kernelClearTriToHE = _shader.FindKernel("ClearTriToHE");
            _kernelBuildRenderableTriToHE = _shader.FindKernel("BuildRenderableTriToHE");
            _kernelClearDirtyVertexFlags = _shader.FindKernel("ClearDirtyVertexFlags");
            _kernelMarkAllDirty = _shader.FindKernel("MarkAllDirty");
            _kernelCopyHalfEdges = _shader.FindKernel("CopyHalfEdges");
        }

        //---------------------------------------------------------------------------
        // Initialisation
        //---------------------------------------------------------------------------
        /// <summary>
        /// Initialises the triangulation with a fixed set of points and an initial half‑edge mesh.
        /// </summary>
        /// <param name="allPoints">All vertex positions (real points followed by one or more ghost/super points).</param>
        /// <param name="realPointCount">Number of real points (must be less than allPoints.Count).</param>
        /// <param name="initialHalfEdges">Initial half‑edge array describing the triangulation.</param>
        /// <param name="triangleCount">Number of triangles.</param>
        /// <param name="neighborCount">Maximum number of neighbours per vertex (used for neighbour lists).</param>
        public void Init(
            IReadOnlyList<float2> allPoints,
            int realPointCount,
            DTBuilder.HalfEdge[] initialHalfEdges,
            int triangleCount,
            int neighborCount) {
            DisposeBuffers();

            if (allPoints == null) throw new ArgumentNullException(nameof(allPoints));
            if (initialHalfEdges == null) throw new ArgumentNullException(nameof(initialHalfEdges));
            if (realPointCount <= 0) throw new ArgumentOutOfRangeException(nameof(realPointCount));
            if (triangleCount < 0) throw new ArgumentOutOfRangeException(nameof(triangleCount));
            if (neighborCount <= 0) throw new ArgumentOutOfRangeException(nameof(neighborCount));

            _vertexCount = allPoints.Count;
            _realVertexCount = realPointCount;

            // Require at least one super-triangle worth of ghost points.
            if (_vertexCount < _realVertexCount + 3)
                throw new ArgumentException("Expected at least real points + 3 super points.", nameof(allPoints));

            _halfEdgeCount = initialHalfEdges.Length;
            if (_halfEdgeCount == 0) throw new ArgumentException("Half-edge buffer is empty.", nameof(initialHalfEdges));
            if ((_halfEdgeCount % 3) != 0) throw new ArgumentException("Half-edge count must be a multiple of 3.", nameof(initialHalfEdges));

            _triCount = triangleCount;
            _neighborCount = neighborCount;

            // Scratch copy of positions for CPU upload.
            _positionScratch = new float2[_vertexCount];
            for (int i = 0; i < _vertexCount; i++)
                _positionScratch[i] = allPoints[i];

            //---------------------------------------------------------------------------
            // Create triple‑buffers for renderable data.
            //---------------------------------------------------------------------------
            for (int i = 0; i < 3; i++) {
                _positions[i] = new ComputeBuffer(_vertexCount, sizeof(float) * 2, ComputeBufferType.Structured);
                _halfEdges[i] = new ComputeBuffer(_halfEdgeCount, sizeof(int) * 4, ComputeBufferType.Structured);
                _triToHE[i] = new ComputeBuffer(_triCount, sizeof(int), ComputeBufferType.Structured);

                // Upload identical initial data to all slots.
                _positions[i].SetData(_positionScratch);
                _halfEdges[i].SetData(initialHalfEdges);
            }

            //---------------------------------------------------------------------------
            // Create shared working buffers.
            //---------------------------------------------------------------------------
            _triLocks = new ComputeBuffer(_triCount, sizeof(int), ComputeBufferType.Structured);
            _vToE = new ComputeBuffer(_vertexCount, sizeof(int), ComputeBufferType.Structured);
            _neighbors = new ComputeBuffer(_realVertexCount * _neighborCount, sizeof(int), ComputeBufferType.Structured);
            _neighborCounts = new ComputeBuffer(_realVertexCount, sizeof(int), ComputeBufferType.Structured);
            _flipCount = new ComputeBuffer(1, sizeof(uint), ComputeBufferType.Structured);
            _dirtyVertexFlags = new ComputeBuffer(_realVertexCount, sizeof(uint), ComputeBufferType.Structured);

            // Initialise dirty flags to 0 (all clean).
            _dirtyVertexFlags.SetData(new uint[_realVertexCount]);

            // Build initial adjacency and triangle maps synchronously for all three slots.
            for (int i = 0; i < 3; i++)
                RebuildVertexAdjacencyAndTriMapSync(i);

            _renderSlot = 0;
        }

        //---------------------------------------------------------------------------
        // CommandBuffer helpers – set common parameters
        //---------------------------------------------------------------------------
        private void SetCommonParams(CommandBuffer cb) {
            cb.SetComputeIntParam(_shader, "_VertexCount", _vertexCount);
            cb.SetComputeIntParam(_shader, "_RealVertexCount", _realVertexCount);
            cb.SetComputeIntParam(_shader, "_HalfEdgeCount", _halfEdgeCount);
            cb.SetComputeIntParam(_shader, "_TriCount", _triCount);
            cb.SetComputeIntParam(_shader, "_NeighborCount", _neighborCount);
        }

        private void DispatchClearTriLocks(CommandBuffer cb) {
            cb.SetComputeBufferParam(_shader, _kernelClearTriLocks, "_TriLocks", _triLocks);
            cb.DispatchCompute(_shader, _kernelClearTriLocks, (_triCount + 255) / 256, 1, 1);
        }

        private void DispatchClearDirtyVertexFlags(CommandBuffer cb) {
            cb.SetComputeBufferParam(_shader, _kernelClearDirtyVertexFlags, "_DirtyVertexFlags", _dirtyVertexFlags);
            cb.DispatchCompute(_shader, _kernelClearDirtyVertexFlags, (_realVertexCount + 255) / 256, 1, 1);
        }

        //---------------------------------------------------------------------------
        // Public maintenance API
        //---------------------------------------------------------------------------
        /// <summary>
        /// Enqueues a series of edge‑fixing and Delaunay‑legalisation passes on a specific slot.
        /// </summary>
        /// <param name="cb">Command buffer to record into.</param>
        /// <param name="positionsForMaintain">Position buffer to use (usually the slot's own positions, but can be a separate velocity‑updated buffer).</param>
        /// <param name="readSlot">Source slot (0‑2) used as topology input for this step.</param>
        /// <param name="writeSlot">Target slot (0‑2) whose half‑edges will be modified.</param>
        /// <param name="fixIterations">Number of “FixHalfEdges” passes.</param>
        /// <param name="legalizeIterations">Number of “LegalizeHalfEdges” passes.</param>
        /// <param name="rebuildAdjacencyAndTriMap">If true, rebuilds neighbour lists and triangle maps after all flips.</param>
        public void EnqueueMaintain(
            CommandBuffer cb,
            ComputeBuffer positionsForMaintain,
            int readSlot,
            int writeSlot,
            int fixIterations,
            int legalizeIterations,
            bool rebuildAdjacencyAndTriMap = true) {
            if (cb == null) throw new ArgumentNullException(nameof(cb));
            if (positionsForMaintain == null) throw new ArgumentNullException(nameof(positionsForMaintain));
            if (readSlot < 0 || readSlot > 2) throw new ArgumentOutOfRangeException(nameof(readSlot));
            if (writeSlot < 0 || writeSlot > 2) throw new ArgumentOutOfRangeException(nameof(writeSlot));

            SetCommonParams(cb);

            if (readSlot != writeSlot) {
                cb.SetComputeBufferParam(_shader, _kernelCopyHalfEdges, "_HalfEdgesSrc", _halfEdges[readSlot]);
                cb.SetComputeBufferParam(_shader, _kernelCopyHalfEdges, "_HalfEdgesDst", _halfEdges[writeSlot]);
                cb.DispatchCompute(_shader, _kernelCopyHalfEdges, (_halfEdgeCount + 255) / 256, 1, 1);
            }

            // Clear dirty flags once for this maintenance batch.
            DispatchClearDirtyVertexFlags(cb);

            // Bind shared working buffers (these are the same for all kernels).
            cb.SetComputeBufferParam(_shader, _kernelClearTriLocks, "_TriLocks", _triLocks);
            cb.SetComputeBufferParam(_shader, _kernelFixHalfEdges, "_TriLocks", _triLocks);
            cb.SetComputeBufferParam(_shader, _kernelFixHalfEdges, "_FlipCount", _flipCount);
            cb.SetComputeBufferParam(_shader, _kernelLegalizeHalfEdges, "_TriLocks", _triLocks);
            cb.SetComputeBufferParam(_shader, _kernelLegalizeHalfEdges, "_FlipCount", _flipCount);

            cb.SetComputeBufferParam(_shader, _kernelClearVertexToEdge, "_VToE", _vToE);
            cb.SetComputeBufferParam(_shader, _kernelBuildVertexToEdge, "_VToE", _vToE);
            cb.SetComputeBufferParam(_shader, _kernelBuildNeighbors, "_VToE", _vToE);
            cb.SetComputeBufferParam(_shader, _kernelBuildNeighbors, "_Neighbors", _neighbors);
            cb.SetComputeBufferParam(_shader, _kernelBuildNeighbors, "_NeighborCounts", _neighborCounts);

            // Dirty vertex flags are used by the flip kernels.
            cb.SetComputeBufferParam(_shader, _kernelFixHalfEdges, "_DirtyVertexFlags", _dirtyVertexFlags);
            cb.SetComputeBufferParam(_shader, _kernelLegalizeHalfEdges, "_DirtyVertexFlags", _dirtyVertexFlags);

            // Bind the slot‑specific half‑edge buffer and triangle map.
            cb.SetComputeBufferParam(_shader, _kernelFixHalfEdges, "_HalfEdges", _halfEdges[writeSlot]);
            cb.SetComputeBufferParam(_shader, _kernelLegalizeHalfEdges, "_HalfEdges", _halfEdges[writeSlot]);
            cb.SetComputeBufferParam(_shader, _kernelBuildVertexToEdge, "_HalfEdges", _halfEdges[writeSlot]);
            cb.SetComputeBufferParam(_shader, _kernelBuildNeighbors, "_HalfEdges", _halfEdges[writeSlot]);
            cb.SetComputeBufferParam(_shader, _kernelClearTriToHE, "_TriToHE", _triToHE[writeSlot]);
            cb.SetComputeBufferParam(_shader, _kernelBuildRenderableTriToHE, "_HalfEdges", _halfEdges[writeSlot]);
            cb.SetComputeBufferParam(_shader, _kernelBuildRenderableTriToHE, "_TriToHE", _triToHE[writeSlot]);

            // Bind positions (may be same as slot's positions or a separate updated buffer).
            cb.SetComputeBufferParam(_shader, _kernelFixHalfEdges, "_Positions", positionsForMaintain);
            cb.SetComputeBufferParam(_shader, _kernelLegalizeHalfEdges, "_Positions", positionsForMaintain);

            int groups = (_halfEdgeCount + 255) / 256;

            // Fix passes (remove inversions / crossed edges).
            for (int i = 0; i < fixIterations; i++) {
                DispatchClearTriLocks(cb);
                cb.SetBufferData(_flipCount, _flipScratch);          // reset flip counter before each pass
                cb.DispatchCompute(_shader, _kernelFixHalfEdges, groups, 1, 1);
            }

            // Legalisation passes (Delaunay).
            for (int i = 0; i < legalizeIterations; i++) {
                DispatchClearTriLocks(cb);
                cb.SetBufferData(_flipCount, _flipScratch);
                cb.DispatchCompute(_shader, _kernelLegalizeHalfEdges, groups, 1, 1);
            }

            if (rebuildAdjacencyAndTriMap) {
                EnqueueRebuildVertexAdjacencyAndTriMap(cb, writeSlot);
            }
        }

        /// <summary>
        /// Enqueues a full rebuild of vertex neighbour lists and the triangle‑to‑halfedge map for a specific slot.
        /// </summary>
        /// <param name="cb">Command buffer to record into.</param>
        /// <param name="slot">Target slot (0‑2).</param>
        public void EnqueueRebuildVertexAdjacencyAndTriMap(CommandBuffer cb, int slot) {
            if (cb == null) throw new ArgumentNullException(nameof(cb));
            if (slot < 0 || slot > 2) throw new ArgumentOutOfRangeException(nameof(slot));

            SetCommonParams(cb);

            // Shared working buffers.
            cb.SetComputeBufferParam(_shader, _kernelClearVertexToEdge, "_VToE", _vToE);
            cb.SetComputeBufferParam(_shader, _kernelBuildVertexToEdge, "_VToE", _vToE);
            cb.SetComputeBufferParam(_shader, _kernelBuildNeighbors, "_VToE", _vToE);
            cb.SetComputeBufferParam(_shader, _kernelBuildNeighbors, "_Neighbors", _neighbors);
            cb.SetComputeBufferParam(_shader, _kernelBuildNeighbors, "_NeighborCounts", _neighborCounts);

            // Slot‑specific half‑edge and triangle map.
            cb.SetComputeBufferParam(_shader, _kernelBuildVertexToEdge, "_HalfEdges", _halfEdges[slot]);
            cb.SetComputeBufferParam(_shader, _kernelBuildNeighbors, "_HalfEdges", _halfEdges[slot]);
            cb.SetComputeBufferParam(_shader, _kernelClearTriToHE, "_TriToHE", _triToHE[slot]);
            cb.SetComputeBufferParam(_shader, _kernelBuildRenderableTriToHE, "_HalfEdges", _halfEdges[slot]);
            cb.SetComputeBufferParam(_shader, _kernelBuildRenderableTriToHE, "_TriToHE", _triToHE[slot]);

            cb.DispatchCompute(_shader, _kernelClearVertexToEdge, (_vertexCount + 255) / 256, 1, 1);
            cb.DispatchCompute(_shader, _kernelBuildVertexToEdge, (_halfEdgeCount + 255) / 256, 1, 1);
            cb.DispatchCompute(_shader, _kernelBuildNeighbors, (_realVertexCount + 255) / 256, 1, 1);
            cb.DispatchCompute(_shader, _kernelClearTriToHE, (_triCount + 255) / 256, 1, 1);
            cb.DispatchCompute(_shader, _kernelBuildRenderableTriToHE, (_halfEdgeCount + 255) / 256, 1, 1);
        }

        /// <summary>
        /// Synchronous version of <see cref="EnqueueRebuildVertexAdjacencyAndTriMap"/> (used during init).
        /// </summary>
        private void RebuildVertexAdjacencyAndTriMapSync(int slot) {
            // Set common parameters directly on the shader.
            _shader.SetInt("_VertexCount", _vertexCount);
            _shader.SetInt("_RealVertexCount", _realVertexCount);
            _shader.SetInt("_HalfEdgeCount", _halfEdgeCount);
            _shader.SetInt("_TriCount", _triCount);
            _shader.SetInt("_NeighborCount", _neighborCount);

            // Bind buffers.
            _shader.SetBuffer(_kernelClearVertexToEdge, "_VToE", _vToE);
            _shader.SetBuffer(_kernelBuildVertexToEdge, "_VToE", _vToE);
            _shader.SetBuffer(_kernelBuildNeighbors, "_VToE", _vToE);
            _shader.SetBuffer(_kernelBuildNeighbors, "_Neighbors", _neighbors);
            _shader.SetBuffer(_kernelBuildNeighbors, "_NeighborCounts", _neighborCounts);

            _shader.SetBuffer(_kernelBuildVertexToEdge, "_HalfEdges", _halfEdges[slot]);
            _shader.SetBuffer(_kernelBuildNeighbors, "_HalfEdges", _halfEdges[slot]);
            _shader.SetBuffer(_kernelClearTriToHE, "_TriToHE", _triToHE[slot]);
            _shader.SetBuffer(_kernelBuildRenderableTriToHE, "_HalfEdges", _halfEdges[slot]);
            _shader.SetBuffer(_kernelBuildRenderableTriToHE, "_TriToHE", _triToHE[slot]);

            _shader.Dispatch(_kernelClearVertexToEdge, (_vertexCount + 255) / 256, 1, 1);
            _shader.Dispatch(_kernelBuildVertexToEdge, (_halfEdgeCount + 255) / 256, 1, 1);
            _shader.Dispatch(_kernelBuildNeighbors, (_realVertexCount + 255) / 256, 1, 1);
            _shader.Dispatch(_kernelClearTriToHE, (_triCount + 255) / 256, 1, 1);
            _shader.Dispatch(_kernelBuildRenderableTriToHE, (_halfEdgeCount + 255) / 256, 1, 1);
        }

        //---------------------------------------------------------------------------
        // Utility methods
        //---------------------------------------------------------------------------
        /// <summary>
        /// Returns the total number of flips performed during the last maintenance pass.
        /// Note: This performs a synchronous readback from the GPU and may stall the pipeline.
        /// </summary>
        public uint GetLastFlipCount() {
            if (_flipCount == null) return 0;
            uint[] result = new uint[1];
            _flipCount.GetData(result);
            return result[0];
        }

        /// <summary>
        /// Marks all real vertices as dirty (forces neighbour list rebuild on next adjacency build).
        /// </summary>
        public void MarkAllDirty(CommandBuffer cb) {
            if (cb == null) throw new ArgumentNullException(nameof(cb));
            SetCommonParams(cb);
            cb.SetComputeBufferParam(_shader, _kernelMarkAllDirty, "_DirtyVertexFlags", _dirtyVertexFlags);
            cb.DispatchCompute(_shader, _kernelMarkAllDirty, (_realVertexCount + 255) / 256, 1, 1);
        }

        //---------------------------------------------------------------------------
        // IDisposable implementation
        //---------------------------------------------------------------------------
        public void Dispose() {
            DisposeBuffers();
            if (_ownsShaderInstance && _shader)
                UnityEngine.Object.Destroy(_shader);
        }

        private void DisposeBuffers() {
            for (int i = 0; i < 3; i++) {
                _positions[i]?.Dispose();
                _positions[i] = null;
                _halfEdges[i]?.Dispose();
                _halfEdges[i] = null;
                _triToHE[i]?.Dispose();
                _triToHE[i] = null;
            }

            _triLocks?.Dispose();
            _triLocks = null;
            _vToE?.Dispose();
            _vToE = null;
            _neighbors?.Dispose();
            _neighbors = null;
            _neighborCounts?.Dispose();
            _neighborCounts = null;
            _flipCount?.Dispose();
            _flipCount = null;
            _dirtyVertexFlags?.Dispose();
            _dirtyVertexFlags = null;

            _positionScratch = null;
            _vertexCount = _realVertexCount = _halfEdgeCount = _triCount = _neighborCount = 0;
            _renderSlot = 0;
        }
    }
}