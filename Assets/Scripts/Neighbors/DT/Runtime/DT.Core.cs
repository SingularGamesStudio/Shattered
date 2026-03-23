using System;
using System.Collections.Generic;
using GPU.Neighbors;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEngine.Rendering;

namespace GPU.Delaunay {
    /// <summary>
    /// Manages the GPU‑based Delaunay triangulation.
    /// Provides triple‑buffered geometry data for rendering and a set of shared working buffers.
    /// </summary>
    public sealed partial class DT : INeighborSearch {
        private const string MarkerPrefix = "XPBI.DT.";

        //---------------------------------------------------------------------------
        // Shader and kernel IDs
        //---------------------------------------------------------------------------
        private readonly ComputeShader _shader;
        private readonly bool _ownsShaderInstance;

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
        private bool _useSupportRadiusFilter;
        private float _supportRadius2;

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

        /// <summary>Per-real-vertex outward boundary normal (zero for non-boundary vertices).</summary>
        public ComputeBuffer BoundaryNormalsBuffer => _boundaryNormals;

        /// <summary>Per-half-edge boundary flag (1 = boundary edge candidate).</summary>
        public ComputeBuffer BoundaryEdgeFlagsBuffer => _boundaryEdgeFlags;

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
            int neighborCount,
            IReadOnlyList<int> ownerByLocal = null) {
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
            _useSupportRadiusFilter = false;
            _supportRadius2 = 0f;

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
            _triToHEAll = new ComputeBuffer(_triCount, sizeof(int), ComputeBufferType.Structured);
            _triInternal = new ComputeBuffer(_triCount, sizeof(uint), ComputeBufferType.Structured);
            _boundaryEdgeFlags = new ComputeBuffer(_halfEdgeCount, sizeof(uint), ComputeBufferType.Structured);
            _boundaryNormals = new ComputeBuffer(_realVertexCount, sizeof(float) * 2, ComputeBufferType.Structured);
            _ownerByVertex = new ComputeBuffer(_realVertexCount, sizeof(int), ComputeBufferType.Structured);

            // Initialise dirty flags to 0 (all clean).
            _dirtyVertexFlags.SetData(new uint[_realVertexCount]);
            _boundaryNormals.SetData(new float2[_realVertexCount]);
            _triInternal.SetData(new uint[_triCount]);
            _boundaryEdgeFlags.SetData(new uint[_halfEdgeCount]);

            int[] owners = new int[_realVertexCount];
            for (int i = 0; i < _realVertexCount; i++) owners[i] = -1;
            if (ownerByLocal != null) {
                int copy = math.min(_realVertexCount, ownerByLocal.Count);
                for (int i = 0; i < copy; i++) owners[i] = ownerByLocal[i];
            }
            _ownerByVertex.SetData(owners);

            // Build initial adjacency and triangle maps synchronously for all three slots.
            for (int i = 0; i < 3; i++)
                RebuildVertexAdjacencyAndTriMapSync(i);

            _renderSlot = 0;
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
        public void EnqueueBuild(
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
                Dispatch(cb, _shader, _kernelCopyHalfEdges, (_halfEdgeCount + 255) / 256, 1, 1, MarkerPrefix + "CopyHalfEdges");
            }

            // Clear dirty flags once for this maintenance batch.
            DispatchClearDirtyVertexFlags(cb);

            // Bind shared working buffers (these are the same for all kernels).
            PrepareBuildBuffers(cb, positionsForMaintain, writeSlot);

            int groups = (_halfEdgeCount + 255) / 256;

            // Fix passes (remove inversions / crossed edges).
            for (int i = 0; i < fixIterations; i++) {
                DispatchClearTriLocks(cb);
                cb.SetBufferData(_flipCount, _flipScratch);          // reset flip counter before each pass
                Dispatch(cb, _shader, _kernelFixHalfEdges, groups, 1, 1, MarkerPrefix + "FixHalfEdges");
            }

            // Legalisation passes (Delaunay).
            for (int i = 0; i < legalizeIterations; i++) {
                DispatchClearTriLocks(cb);
                cb.SetBufferData(_flipCount, _flipScratch);
                Dispatch(cb, _shader, _kernelLegalizeHalfEdges, groups, 1, 1, MarkerPrefix + "LegalizeHalfEdges");
            }

            if (rebuildAdjacencyAndTriMap) {
                EnqueueRebuildVertexAdjacencyAndTriMap(cb, writeSlot);
            }

            int boundaryGroups = math.max((_triCount + 255) / 256, math.max((_halfEdgeCount + 255) / 256, (_realVertexCount + 255) / 256));
            Dispatch(cb, _shader, _kernelClearBoundaryData, boundaryGroups, 1, 1, MarkerPrefix + "ClearBoundaryData");
            Dispatch(cb, _shader, _kernelBuildTriToHEAll, (_halfEdgeCount + 255) / 256, 1, 1, MarkerPrefix + "BuildTriToHEAll");
            Dispatch(cb, _shader, _kernelClassifyInternalTriangles, (_triCount + 255) / 256, 1, 1, MarkerPrefix + "ClassifyInternalTriangles");
            Dispatch(cb, _shader, _kernelMarkBoundaryEdges, (_halfEdgeCount + 255) / 256, 1, 1, MarkerPrefix + "MarkBoundaryEdges");
            Dispatch(cb, _shader, _kernelClearVertexToEdge, (_vertexCount + 255) / 256, 1, 1, MarkerPrefix + "BoundaryClearVertexToEdge");
            Dispatch(cb, _shader, _kernelBuildVertexToEdge, (_halfEdgeCount + 255) / 256, 1, 1, MarkerPrefix + "BoundaryBuildVertexToEdge");
            Dispatch(cb, _shader, _kernelBuildBoundaryNormals, (_realVertexCount + 255) / 256, 1, 1, MarkerPrefix + "BuildBoundaryNormals");
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
            _useSupportRadiusFilter = supportRadius > 0f;
            _supportRadius2 = _useSupportRadiusFilter ? supportRadius * supportRadius : 0f;
            EnqueueBuild(cb, positions, readSlot, writeSlot, fixIterations, legalizeIterations, rebuildAdjacencyAndTriMap);
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
            Dispatch(cb, _shader, _kernelMarkAllDirty, (_realVertexCount + 255) / 256, 1, 1, MarkerPrefix + "MarkAllDirty");
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
            _triToHEAll?.Dispose();
            _triToHEAll = null;
            _triInternal?.Dispose();
            _triInternal = null;
            _boundaryEdgeFlags?.Dispose();
            _boundaryEdgeFlags = null;
            _boundaryNormals?.Dispose();
            _boundaryNormals = null;
            _ownerByVertex?.Dispose();
            _ownerByVertex = null;

            _positionScratch = null;
            _vertexCount = _realVertexCount = _halfEdgeCount = _triCount = _neighborCount = 0;
            _renderSlot = 0;
        }
    }
}