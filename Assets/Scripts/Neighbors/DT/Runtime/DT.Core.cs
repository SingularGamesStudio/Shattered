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
        private const int IndirectArgStrideBytes = 12;
        private const int FrArgClearTriangleRejectFlags = 0 * IndirectArgStrideBytes;
        private const int FrArgEmitTriangleEdgeRecords = 1 * IndirectArgStrideBytes;
        private const int FrArgRejectSortedEdgeRecords = 2 * IndirectArgStrideBytes;
        private const int FrArgCompactValidTrianglesToTemp = 3 * IndirectArgStrideBytes;
        private const int FrArgCopyFilteredTrianglesBack = 4 * IndirectArgStrideBytes;
        private const int FrArgBuildHalfEdges = 5 * IndirectArgStrideBytes;
        private const int FrArgBuildDirectedEdgeHash = 6 * IndirectArgStrideBytes;
        private const int FrArgResolveTwins = 7 * IndirectArgStrideBytes;
        private const int FrArgBuildVertexToEdgeBoundary = 8 * IndirectArgStrideBytes;
        private const int FrArgBuildTriToHEAll = 9 * IndirectArgStrideBytes;
        private const int FrArgClassifyInternalTriangles = 10 * IndirectArgStrideBytes;
        private const int FrArgMarkBoundaryEdges = 11 * IndirectArgStrideBytes;
        private const int FrArgBoundaryBuildVertexToEdge = 12 * IndirectArgStrideBytes;
        private const int FrIndirectArgRecordCount = 13;

        private const int CTR_TRI_COUNT = 0;
        private const int CTR_HE_USED = 1;
        private const int CTR_TRI_FILTERED = 5;

        public int _maxHalfEdges;
        public int _maxTriangles;

        private static int GroupsFor(int count, int threads) {
            return math.max(1, (count + threads - 1) / threads);
        }

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

        /// <summary>Per-triangle internal classification (1 = internal triangle) in current render slot.</summary>
        public ComputeBuffer TriInternalBuffer => _triInternal[_renderSlot];

        /// <summary>Positions buffer for the current render slot.</summary>
        public ComputeBuffer PositionsBuffer => _positions[_renderSlot];

        /// <summary>Half‑edges buffer for the current render slot.</summary>
        public ComputeBuffer HalfEdgesBuffer => _halfEdges[_renderSlot];

        /// <summary>Triangle‑to‑halfedge map for the current render slot.</summary>
        public ComputeBuffer TriToHEBuffer => _triToHE[_renderSlot];

        public ComputeBuffer GetPositionsBuffer(int slot) => _positions[slot];
        public ComputeBuffer GetHalfEdgesBuffer(int slot) => _halfEdges[slot];
        public ComputeBuffer GetTriToHEBuffer(int slot) => _triToHE[slot];
        public ComputeBuffer GetTriInternalBuffer(int slot) => _triInternal[slot];

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
            _maxHalfEdges = 2 * _halfEdgeCount;
            if (_halfEdgeCount == 0) throw new ArgumentException("Half-edge buffer is empty.", nameof(initialHalfEdges));
            if ((_halfEdgeCount % 3) != 0) throw new ArgumentException("Half-edge count must be a multiple of 3.", nameof(initialHalfEdges));

            _triCount = triangleCount;
            _maxTriangles = triangleCount * 2;
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
                _triInternal[i] = new ComputeBuffer(_triCount, sizeof(uint), ComputeBufferType.Structured);

                // Upload identical initial data to all slots.
                _positions[i].SetData(_positionScratch);
                _halfEdges[i].SetData(initialHalfEdges);
                _triInternal[i].SetData(new uint[_triCount]);
            }

            //---------------------------------------------------------------------------
            // Create shared working buffers.
            //---------------------------------------------------------------------------
            _triLocks = new ComputeBuffer(_triCount, sizeof(int), ComputeBufferType.Structured);
            _vToE = new ComputeBuffer(_vertexCount, sizeof(int), ComputeBufferType.Structured);
            _debug = new ComputeBuffer(1, sizeof(int), ComputeBufferType.Structured);
            _neighbors = new ComputeBuffer(_realVertexCount * _neighborCount, sizeof(int), ComputeBufferType.Structured);
            _neighborCounts = new ComputeBuffer(_realVertexCount, sizeof(int), ComputeBufferType.Structured);
            _flipCount = new ComputeBuffer(1, sizeof(uint), ComputeBufferType.Structured);
            _dirtyVertexFlags = new ComputeBuffer(_realVertexCount, sizeof(uint), ComputeBufferType.Structured);
            _triToHEAll = new ComputeBuffer(_triCount, sizeof(int), ComputeBufferType.Structured);
            _boundaryEdgeFlags = new ComputeBuffer(_halfEdgeCount, sizeof(uint), ComputeBufferType.Structured);
            _boundaryNormals = new ComputeBuffer(_realVertexCount, sizeof(float) * 2, ComputeBufferType.Structured);
            _ownerByVertex = new ComputeBuffer(_vertexCount, sizeof(int), ComputeBufferType.Structured);

            // Initialise dirty flags to 0 (all clean).
            _dirtyVertexFlags.SetData(new uint[_realVertexCount]);
            _boundaryNormals.SetData(new float2[_realVertexCount]);
            _boundaryEdgeFlags.SetData(new uint[_halfEdgeCount]);

            int[] owners = new int[_vertexCount];
            for (int i = 0; i < _vertexCount; i++) owners[i] = -1;
            if (ownerByLocal != null) {
                int copy = math.min(_realVertexCount, ownerByLocal.Count);
                for (int i = 0; i < copy; i++) owners[i] = ownerByLocal[i];
            }
            _ownerByVertex.SetData(owners);
            _ownerByVertexInit = owners;

            EnsureFullRebuildBuffers();

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

        public bool SupportsFullRebuild => _fullRebuildShader != null;

        public void EnqueueFullRebuild(
            CommandBuffer cb,
            ComputeBuffer positionsForMaintain,
            float supportRadius,
            int writeSlot,
            bool rebuildAdjacencyAndTriMap = true) {
            _useSupportRadiusFilter = supportRadius > 0f;
            _supportRadius2 = _useSupportRadiusFilter ? supportRadius * supportRadius : 0f;
            EnqueueFullRebuildIndirect(cb, positionsForMaintain, writeSlot, rebuildAdjacencyAndTriMap);
        }

        public void EnqueueFullRebuildIndirect(
            CommandBuffer cb,
            ComputeBuffer positionsForMaintain,
            int writeSlot,
            bool rebuildAdjacencyAndTriMap = true) {
            if (cb == null) throw new ArgumentNullException(nameof(cb));
            if (positionsForMaintain == null) throw new ArgumentNullException(nameof(positionsForMaintain));
            if (writeSlot < 0 || writeSlot > 2) throw new ArgumentOutOfRangeException(nameof(writeSlot));
            if (!SupportsFullRebuild)
                throw new InvalidOperationException("DT full rebuild shader is not configured.");

            EnsureFullRebuildBuffers();
            EnsureFullRebuildIndirectArgs();
            SetFullRebuildCommonParams(cb, positionsForMaintain, writeSlot);
            BindFullRebuildIndirectArgs(cb);

            int partialGroups = math.max(1, (_realVertexCount + 255) / 256);
            int gridCells = math.max(1, _rebuildGridW * _rebuildGridH);
            int gridGroups1D = math.max(1, (gridCells + 255) / 256);
            int gridGroupsX = math.max(1, (_rebuildGridW + 7) / 8);
            int gridGroupsY = math.max(1, (_rebuildGridH + 7) / 8);

            Dispatch(cb, _fullRebuildShader, _frKernelBoundsReducePartials, partialGroups, 1, 1, MarkerPrefix + "FR.BoundsReducePartials");
            Dispatch(cb, _fullRebuildShader, _frKernelBoundsFinalize, 1, 1, 1, MarkerPrefix + "FR.BoundsFinalize");

            Dispatch(cb, _fullRebuildShader, _frKernelClearRebuildGrid, gridGroups1D, 1, 1, MarkerPrefix + "FR.ClearRebuildGrid");
            Dispatch(cb, _fullRebuildShader, _frKernelClearTriangleHash, GroupsFor(_rebuildTriHashSize, 256), 1, 1, MarkerPrefix + "FR.ClearTriangleHash");
            Dispatch(cb, _fullRebuildShader, _frKernelClearEdgeHash, GroupsFor(_rebuildEdgeHashSize, 256), 1, 1, MarkerPrefix + "FR.ClearEdgeHash");

            int clearMeshCount = math.max(_vertexCount,math.max(_maxHalfEdges, _maxTriangles));
            Dispatch(cb, _fullRebuildShader, _frKernelClearMeshState, GroupsFor(clearMeshCount, 256), 1, 1, MarkerPrefix + "FR.ClearMeshState");

            // Keep static per-vertex object ownership for owner-isolated rebuild passes.
            if (_ownerByVertexInit != null)
                cb.SetBufferData(_ownerByVertex, _ownerByVertexInit);

            var ownerSet = new HashSet<int>();
            var activeOwners = new List<int>(8);
            if (_ownerByVertexInit != null) {
                int scan = math.min(_realVertexCount, _ownerByVertexInit.Length);
                for (int i = 0; i < scan; i++) {
                    int owner = _ownerByVertexInit[i];
                    if (owner >= 0 && ownerSet.Add(owner))
                        activeOwners.Add(owner);
                }
            }
            if (activeOwners.Count == 0)
                activeOwners.Add(-1);

            int jfaStep = 1;
            int maxDim = math.max(_rebuildGridW, _rebuildGridH);
            while (jfaStep < maxDim)
                jfaStep <<= 1;

            for (int ownerIdx = 0; ownerIdx < activeOwners.Count; ownerIdx++) {
                int owner = activeOwners[ownerIdx];
                cb.SetComputeIntParam(_fullRebuildShader, "_ActiveOwner", owner);

                Dispatch(cb, _fullRebuildShader, _frKernelClearRebuildGrid, gridGroups1D, 1, 1, MarkerPrefix + "FR.ClearRebuildGrid");
                Dispatch(cb, _fullRebuildShader, _frKernelSeedSitesToGrid, GroupsFor(_vertexCount, 256), 1, 1, MarkerPrefix + "FR.SeedSitesToGrid");
                Dispatch(cb, _fullRebuildShader, _frKernelInitVoronoiFromSeeds, gridGroups1D, 1, 1, MarkerPrefix + "FR.InitVoronoiFromSeeds");

                int ownerJfaStep = jfaStep;
                bool writeToB = true;
                for (int step = ownerJfaStep; step >= 1; step >>= 1) {
                    cb.SetComputeIntParam(_fullRebuildShader, "_NeighborCount", step);
                    if (writeToB)
                        Dispatch(cb, _fullRebuildShader, _frKernelJumpFloodAtoB, gridGroupsX, gridGroupsY, 1, MarkerPrefix + "FR.JumpFloodAtoB");
                    else
                        Dispatch(cb, _fullRebuildShader, _frKernelJumpFloodBtoA, gridGroupsX, gridGroupsY, 1, MarkerPrefix + "FR.JumpFloodBtoA");
                    writeToB = !writeToB;
                }

                if (!writeToB)
                    Dispatch(cb, _fullRebuildShader, _frKernelJumpFloodBtoA, gridGroupsX, gridGroupsY, 1, MarkerPrefix + "FR.JumpFloodBtoA.Finalize");

                Dispatch(cb, _fullRebuildShader, _frKernelRemoveIslandsAtoB, gridGroupsX, gridGroupsY, 1, MarkerPrefix + "FR.RemoveIslandsAtoB");
                Dispatch(cb, _fullRebuildShader, _frKernelRemoveIslandsBtoA, gridGroupsX, gridGroupsY, 1, MarkerPrefix + "FR.RemoveIslandsBtoA");
                Dispatch(cb, _fullRebuildShader, _frKernelExtractTrianglesFromVoronoi, gridGroupsX, gridGroupsY, 1, MarkerPrefix + "FR.ExtractTrianglesFromVoronoi");
            }
            Dispatch(cb, _fullRebuildShader, _frKernelCompactTrianglesFromHash, GroupsFor(_rebuildTriHashSize, 256), 1, 1, MarkerPrefix + "FR.CompactTrianglesFromHash");

            // Full rebuild may emit cross-object candidates; restore static owners before
            // filtering so mixed-owner triangles can be rejected deterministically.
            if (_ownerByVertexInit != null)
                cb.SetBufferData(_ownerByVertex, _ownerByVertexInit);

            WriteIndirectArgs(cb, FrArgClearTriangleRejectFlags, CTR_TRI_COUNT, 256, 1);
            WriteIndirectArgs(cb, FrArgEmitTriangleEdgeRecords, CTR_TRI_COUNT, 256, 1);
            WriteIndirectArgs(cb, FrArgRejectSortedEdgeRecords, CTR_TRI_COUNT, 256, 3);
            WriteIndirectArgs(cb, FrArgCompactValidTrianglesToTemp, CTR_TRI_COUNT, 256, 1);

            DispatchIndirect(cb, _fullRebuildShader, _frKernelClearTriangleRejectFlags, FrArgClearTriangleRejectFlags);
            Dispatch(cb, _fullRebuildShader, _frKernelClearEdgeRecords, GroupsFor(_rebuildMaxEdgeRecords, 256), 1, 1, MarkerPrefix + "FR.ClearEdgeRecords");
            DispatchIndirect(cb, _fullRebuildShader, _frKernelEmitTriangleEdgeRecords, FrArgEmitTriangleEdgeRecords);

            int sortCount = 1;
            while (sortCount < _rebuildMaxEdgeRecords)
                sortCount <<= 1;

            int sortGroups = (sortCount + 255) / 256;
            for (int k = 2; k <= sortCount; k <<= 1) {
                cb.SetComputeIntParam(_fullRebuildShader, "_SortK", k);
                for (int j = k >> 1; j > 0; j >>= 1) {
                    cb.SetComputeIntParam(_fullRebuildShader, "_SortJ", j);
                    Dispatch(cb, _fullRebuildShader, _frKernelSortEdgeRecordsBitonic, sortGroups, 1, 1, MarkerPrefix + "FR.SortEdgeRecordsBitonic");
                }
            }

            DispatchIndirect(cb, _fullRebuildShader, _frKernelRejectTrianglesFromSortedEdgeRecords, FrArgRejectSortedEdgeRecords);
            Dispatch(cb, _fullRebuildShader, _frKernelResetFilteredTriCounter, 1, 1, 1, MarkerPrefix + "FR.ResetFilteredTriCounter");
            DispatchIndirect(cb, _fullRebuildShader, _frKernelCompactValidTrianglesToTemp, FrArgCompactValidTrianglesToTemp);

            WriteIndirectArgs(cb, FrArgCopyFilteredTrianglesBack, CTR_TRI_FILTERED, 256, 1);
            DispatchIndirect(cb, _fullRebuildShader, _frKernelCopyFilteredTrianglesBack, FrArgCopyFilteredTrianglesBack);
            Dispatch(cb, _fullRebuildShader, _frKernelFinalizeFilteredTriCount, 1, 1, 1, MarkerPrefix + "FR.FinalizeFilteredTriCount");

            Dispatch(cb, _fullRebuildShader, _frKernelClearEdgeHash, GroupsFor(_rebuildEdgeHashSize, 256), 1, 1, MarkerPrefix + "FR.ClearEdgeHash.2");

            WriteIndirectArgs(cb, FrArgBuildHalfEdges, CTR_TRI_COUNT, 256, 1);
            WriteIndirectArgs(cb, FrArgBuildDirectedEdgeHash, CTR_HE_USED, 256, 1);
            WriteIndirectArgs(cb, FrArgResolveTwins, CTR_HE_USED, 256, 1);
            WriteIndirectArgs(cb, FrArgBuildVertexToEdgeBoundary, CTR_HE_USED, 256, 1);

            DispatchIndirect(cb, _fullRebuildShader, _frKernelBuildHalfEdgesFromTriangles, FrArgBuildHalfEdges);
            DispatchIndirect(cb, _fullRebuildShader, _frKernelBuildDirectedEdgeHash, FrArgBuildDirectedEdgeHash);
            DispatchIndirect(cb, _fullRebuildShader, _frKernelResolveTwinsFromEdgeHash, FrArgResolveTwins);
            DispatchIndirect(cb, _fullRebuildShader, _frKernelBuildVertexToEdgeAndBoundary, FrArgBuildVertexToEdgeBoundary);

            if (_ownerByVertexInit != null)
                cb.SetBufferData(_ownerByVertex, _ownerByVertexInit);

            if (rebuildAdjacencyAndTriMap)
                EnqueueRebuildVertexAdjacencyAndTriMap(cb, writeSlot);

            // Clear dirty flags once for this maintenance batch.
            DispatchClearDirtyVertexFlags(cb);

            SetCommonParams(cb);
            PrepareBuildBuffers(cb, positionsForMaintain, writeSlot);

            int boundaryClearGroups = math.max(GroupsFor(_realVertexCount, 256), math.max(GroupsFor(_maxHalfEdges, 256), GroupsFor(_maxTriangles, 256)));
            Dispatch(cb, _shader, _kernelClearBoundaryData, boundaryClearGroups, 1, 1, MarkerPrefix + "ClearBoundaryData");

            WriteIndirectArgs(cb, FrArgBuildTriToHEAll, CTR_HE_USED, 256, 1);
            WriteIndirectArgs(cb, FrArgClassifyInternalTriangles, CTR_TRI_COUNT, 256, 1);
            WriteIndirectArgs(cb, FrArgMarkBoundaryEdges, CTR_HE_USED, 256, 1);
            WriteIndirectArgs(cb, FrArgBoundaryBuildVertexToEdge, CTR_HE_USED, 256, 1);

            DispatchIndirect(cb, _shader, _kernelBuildTriToHEAll, FrArgBuildTriToHEAll);
            DispatchIndirect(cb, _shader, _kernelClassifyInternalTriangles, FrArgClassifyInternalTriangles);
            DispatchIndirect(cb, _shader, _kernelMarkBoundaryEdges, FrArgMarkBoundaryEdges);

            Dispatch(cb, _shader, _kernelClearVertexToEdge, GroupsFor(_vertexCount, 256), 1, 1, MarkerPrefix + "BoundaryClearVertexToEdge");
            DispatchIndirect(cb, _shader, _kernelBuildVertexToEdge, FrArgBoundaryBuildVertexToEdge);
            Dispatch(cb, _shader, _kernelBuildBoundaryNormals, GroupsFor(_realVertexCount, 256), 1, 1, MarkerPrefix + "BuildBoundaryNormals");
        }

        private void EnsureFullRebuildIndirectArgs() {
            int uintCount = FrIndirectArgRecordCount * 3;
            if (_frIndirectArgs == null || !_frIndirectArgs.IsValid() || _frIndirectArgs.count != uintCount) {
                _frIndirectArgs?.Dispose();
                _frIndirectArgs = new ComputeBuffer(uintCount, sizeof(uint), ComputeBufferType.IndirectArguments);
                _frIndirectArgs.SetData(new uint[uintCount]);
            }
        }

        private void BindFullRebuildIndirectArgs(CommandBuffer cb) {
            cb.SetComputeBufferParam(_fullRebuildShader, _frKernelWriteIndirectArgsFromCounter, "_IndirectDispatchArgs", _frIndirectArgs);
        }

        private void WriteIndirectArgs(
            CommandBuffer cb,
            int argsOffsetBytes,
            int counterIndex,
            int threadsPerGroup,
            int multiplier = 1) {
            cb.SetComputeIntParam(_fullRebuildShader, "_IndirectArgsOffsetWords", argsOffsetBytes / 4);
            cb.SetComputeIntParam(_fullRebuildShader, "_IndirectCounterIndex", counterIndex);
            cb.SetComputeIntParam(_fullRebuildShader, "_IndirectThreadsPerGroup", threadsPerGroup);
            cb.SetComputeIntParam(_fullRebuildShader, "_IndirectCountMultiplier", multiplier);
            Dispatch(cb, _fullRebuildShader, _frKernelWriteIndirectArgsFromCounter, 1, 1, 1, MarkerPrefix + "FR.WriteIndirectArgs");
        }

        private void DispatchIndirect(CommandBuffer cb, ComputeShader shader, int kernel, int argsOffsetBytes) {
            cb.DispatchCompute(shader, kernel, _frIndirectArgs, (uint)argsOffsetBytes);
        }

        private void EnsureFullRebuildBuffers() {
            if (!SupportsFullRebuild || _vertexCount <= 0 || _realVertexCount <= 0 || _triCount <= 0 || _halfEdgeCount <= 0)
                return;

            int side = (int)math.ceil(math.sqrt(math.max(1, _realVertexCount)) * 2f);
            _rebuildGridW = math.max(8, side);
            _rebuildGridH = _rebuildGridW;
            _rebuildTriHashSize = math.max(64, _triCount * 4);
            _rebuildEdgeHashSize = math.max(64, _halfEdgeCount * 4);
            int edgeRecordMin = math.max(3, _triCount * 3);
            int edgeRecordCapacity = 1;
            while (edgeRecordCapacity < edgeRecordMin)
                edgeRecordCapacity <<= 1;
            _rebuildMaxEdgeRecords = edgeRecordCapacity;

            int gridCells = _rebuildGridW * _rebuildGridH;

            if (_gridSeedOwner == null || _gridSeedOwner.count != gridCells) {
                _gridSeedOwner?.Dispose();
                _gridSeedOwner = new ComputeBuffer(gridCells, sizeof(int), ComputeBufferType.Structured);
            }
            if (_voronoiA == null || _voronoiA.count != gridCells) {
                _voronoiA?.Dispose();
                _voronoiA = new ComputeBuffer(gridCells, sizeof(int), ComputeBufferType.Structured);
            }
            if (_voronoiB == null || _voronoiB.count != gridCells) {
                _voronoiB?.Dispose();
                _voronoiB = new ComputeBuffer(gridCells, sizeof(int), ComputeBufferType.Structured);
            }

            if (_triHashA == null || _triHashA.count != _rebuildTriHashSize) {
                _triHashA?.Dispose();
                _triHashA = new ComputeBuffer(_rebuildTriHashSize, sizeof(uint), ComputeBufferType.Structured);
            }
            if (_triHashB == null || _triHashB.count != _rebuildTriHashSize) {
                _triHashB?.Dispose();
                _triHashB = new ComputeBuffer(_rebuildTriHashSize, sizeof(uint), ComputeBufferType.Structured);
            }
            if (_triHashC == null || _triHashC.count != _rebuildTriHashSize) {
                _triHashC?.Dispose();
                _triHashC = new ComputeBuffer(_rebuildTriHashSize, sizeof(uint), ComputeBufferType.Structured);
            }
            if (_triHashState == null || _triHashState.count != _rebuildTriHashSize) {
                _triHashState?.Dispose();
                _triHashState = new ComputeBuffer(_rebuildTriHashSize, sizeof(uint), ComputeBufferType.Structured);
            }

            if (_triRaw == null || _triRaw.count != _triCount) {
                _triRaw?.Dispose();
                _triRaw = new ComputeBuffer(_triCount, sizeof(uint) * 3, ComputeBufferType.Structured);
            }
            if (_triReject == null || _triReject.count != _triCount) {
                _triReject?.Dispose();
                _triReject = new ComputeBuffer(_triCount, sizeof(uint), ComputeBufferType.Structured);
            }
            if (_triTemp == null || _triTemp.count != _triCount) {
                _triTemp?.Dispose();
                _triTemp = new ComputeBuffer(_triCount, sizeof(uint) * 3, ComputeBufferType.Structured);
            }
            if (_siteSeenInTri == null || _siteSeenInTri.count != _vertexCount) {
                _siteSeenInTri?.Dispose();
                _siteSeenInTri = new ComputeBuffer(_vertexCount, sizeof(uint), ComputeBufferType.Structured);
            }
            if (_missingSites == null || _missingSites.count != _realVertexCount) {
                _missingSites?.Dispose();
                _missingSites = new ComputeBuffer(_realVertexCount, sizeof(int), ComputeBufferType.Structured);
            }

            if (_edgeHashSrc == null || _edgeHashSrc.count != _rebuildEdgeHashSize) {
                _edgeHashSrc?.Dispose();
                _edgeHashSrc = new ComputeBuffer(_rebuildEdgeHashSize, sizeof(uint), ComputeBufferType.Structured);
            }
            if (_edgeHashDst == null || _edgeHashDst.count != _rebuildEdgeHashSize) {
                _edgeHashDst?.Dispose();
                _edgeHashDst = new ComputeBuffer(_rebuildEdgeHashSize, sizeof(uint), ComputeBufferType.Structured);
            }
            if (_edgeHashHE == null || _edgeHashHE.count != _rebuildEdgeHashSize) {
                _edgeHashHE?.Dispose();
                _edgeHashHE = new ComputeBuffer(_rebuildEdgeHashSize, sizeof(int), ComputeBufferType.Structured);
            }
            if (_edgeHashState == null || _edgeHashState.count != _rebuildEdgeHashSize) {
                _edgeHashState?.Dispose();
                _edgeHashState = new ComputeBuffer(_rebuildEdgeHashSize, sizeof(uint), ComputeBufferType.Structured);
            }

            if (_edgeRecHash == null || _edgeRecHash.count != _rebuildMaxEdgeRecords) {
                _edgeRecHash?.Dispose();
                _edgeRecHash = new ComputeBuffer(_rebuildMaxEdgeRecords, sizeof(uint), ComputeBufferType.Structured);
            }
            if (_edgeRecA == null || _edgeRecA.count != _rebuildMaxEdgeRecords) {
                _edgeRecA?.Dispose();
                _edgeRecA = new ComputeBuffer(_rebuildMaxEdgeRecords, sizeof(uint), ComputeBufferType.Structured);
            }
            if (_edgeRecB == null || _edgeRecB.count != _rebuildMaxEdgeRecords) {
                _edgeRecB?.Dispose();
                _edgeRecB = new ComputeBuffer(_rebuildMaxEdgeRecords, sizeof(uint), ComputeBufferType.Structured);
            }
            if (_edgeRecTri == null || _edgeRecTri.count != _rebuildMaxEdgeRecords) {
                _edgeRecTri?.Dispose();
                _edgeRecTri = new ComputeBuffer(_rebuildMaxEdgeRecords, sizeof(int), ComputeBufferType.Structured);
            }
            if (_edgeRecOpp == null || _edgeRecOpp.count != _rebuildMaxEdgeRecords) {
                _edgeRecOpp?.Dispose();
                _edgeRecOpp = new ComputeBuffer(_rebuildMaxEdgeRecords, sizeof(int), ComputeBufferType.Structured);
            }

            if (_rebuildCounters == null || _rebuildCounters.count != 6) {
                _rebuildCounters?.Dispose();
                _rebuildCounters = new ComputeBuffer(6, sizeof(uint), ComputeBufferType.Structured);
            }
        }

        private void SetFullRebuildCommonParams(CommandBuffer cb, ComputeBuffer positionsForMaintain, int writeSlot) {
            cb.SetComputeIntParam(_fullRebuildShader, "_VertexCount", _vertexCount);
            cb.SetComputeIntParam(_fullRebuildShader, "_RealVertexCount", _realVertexCount);
            cb.SetComputeIntParam(_fullRebuildShader, "_HalfEdgeCount", _halfEdgeCount);
            cb.SetComputeIntParam(_fullRebuildShader, "_TriCount", _triCount);
            cb.SetComputeIntParam(_fullRebuildShader, "_NeighborCount", _neighborCount);
            cb.SetComputeIntParam(_fullRebuildShader, "_UseSupportRadiusFilter", _useSupportRadiusFilter ? 1 : 0);
            cb.SetComputeFloatParam(_fullRebuildShader, "_SupportRadius2", _supportRadius2);
            cb.SetComputeIntParam(_fullRebuildShader, "_BoundsPartialCount", math.max(1, (_realVertexCount + 255) / 256));

            cb.SetComputeIntParam(_fullRebuildShader, "_GridW", _rebuildGridW);
            cb.SetComputeIntParam(_fullRebuildShader, "_GridH", _rebuildGridH);
            cb.SetComputeIntParam(_fullRebuildShader, "_TriHashSize", _rebuildTriHashSize);
            cb.SetComputeIntParam(_fullRebuildShader, "_EdgeHashSize", _rebuildEdgeHashSize);
            cb.SetComputeIntParam(_fullRebuildShader, "_MaxEdgeRecords", _rebuildMaxEdgeRecords);
            cb.SetComputeIntParam(_fullRebuildShader, "_MaxTriangles", _triCount);
            cb.SetComputeIntParam(_fullRebuildShader, "_MaxHalfEdges", _halfEdgeCount);
            cb.SetComputeIntParam(_fullRebuildShader, "_MaxMissingVertices", _realVertexCount);
            cb.SetComputeIntParam(_fullRebuildShader, "_InsertionWalkLimit", 32);
            cb.SetComputeIntParam(_fullRebuildShader, "_SortK", 2);
            cb.SetComputeIntParam(_fullRebuildShader, "_SortJ", 1);
            cb.SetComputeFloatParam(_fullRebuildShader, "_RebuildPadding", 0.05f);
            cb.SetComputeFloatParam(_fullRebuildShader, "_InsideEps", 1e-7f);

            int partialGroups = math.max(1, (_realVertexCount + 255) / 256);
            EnsureBoundsBuffers(partialGroups);

            int[] kernels = {
                _frKernelBoundsReducePartials,
                _frKernelBoundsFinalize,
                _frKernelClearRebuildGrid,
                _frKernelClearTriangleHash,
                _frKernelClearEdgeHash,
                _frKernelClearMeshState,
                _frKernelSeedSitesToGrid,
                _frKernelAssignOwnersByCell,
                _frKernelInitVoronoiFromSeeds,
                _frKernelJumpFloodAtoB,
                _frKernelJumpFloodBtoA,
                _frKernelRemoveIslandsAtoB,
                _frKernelRemoveIslandsBtoA,
                _frKernelExtractTrianglesFromVoronoi,
                _frKernelCompactTrianglesFromHash,
                _frKernelClearTriangleRejectFlags,
                _frKernelClearEdgeRecords,
                _frKernelEmitTriangleEdgeRecords,
                _frKernelSortEdgeRecordsBitonic,
                _frKernelRejectTrianglesFromSortedEdgeRecords,
                _frKernelResetFilteredTriCounter,
                _frKernelCompactValidTrianglesToTemp,
                _frKernelCopyFilteredTrianglesBack,
                _frKernelFinalizeFilteredTriCount,
                _frKernelInitAllocatorsFromTriCount,
                _frKernelWriteIndirectArgsFromCounter,
                _frKernelBuildHalfEdgesFromTriangles,
                _frKernelBuildDirectedEdgeHash,
                _frKernelResolveTwinsFromEdgeHash,
                _frKernelBuildVertexToEdgeAndBoundary,
            };

            for (int i = 0; i < kernels.Length; i++) {
                int k = kernels[i];
                cb.SetComputeBufferParam(_fullRebuildShader, k, "_Positions", positionsForMaintain);
                cb.SetComputeBufferParam(_fullRebuildShader, k, "_HalfEdges", _halfEdges[writeSlot]);
                cb.SetComputeBufferParam(_fullRebuildShader, k, "_TriLocks", _triLocks);
                cb.SetComputeBufferParam(_fullRebuildShader, k, "_VToE", _vToE);
                cb.SetComputeBufferParam(_fullRebuildShader, k, "_Debug", _debug);
                cb.SetComputeBufferParam(_fullRebuildShader, k, "_Neighbors", _neighbors);
                cb.SetComputeBufferParam(_fullRebuildShader, k, "_NeighborCounts", _neighborCounts);
                cb.SetComputeBufferParam(_fullRebuildShader, k, "_FlipCount", _flipCount);
                cb.SetComputeBufferParam(_fullRebuildShader, k, "_TriToHE", _triToHE[writeSlot]);
                cb.SetComputeBufferParam(_fullRebuildShader, k, "_TriToHEAll", _triToHEAll);
                cb.SetComputeBufferParam(_fullRebuildShader, k, "_DirtyVertexFlags", _dirtyVertexFlags);
                cb.SetComputeBufferParam(_fullRebuildShader, k, "_TriInternal", _triInternal[writeSlot]);
                cb.SetComputeBufferParam(_fullRebuildShader, k, "_BoundaryEdgeFlags", _boundaryEdgeFlags);
                cb.SetComputeBufferParam(_fullRebuildShader, k, "_BoundaryNormals", _boundaryNormals);
                cb.SetComputeBufferParam(_fullRebuildShader, k, "_OwnerByVertex", _ownerByVertex);

                cb.SetComputeBufferParam(_fullRebuildShader, k, "_BoundsPartials", _boundsPartials);
                cb.SetComputeBufferParam(_fullRebuildShader, k, "_BoundsResult", _boundsResult);

                cb.SetComputeBufferParam(_fullRebuildShader, k, "_GridSeedOwner", _gridSeedOwner);
                cb.SetComputeBufferParam(_fullRebuildShader, k, "_VoronoiA", _voronoiA);
                cb.SetComputeBufferParam(_fullRebuildShader, k, "_VoronoiB", _voronoiB);

                cb.SetComputeBufferParam(_fullRebuildShader, k, "_TriHashA", _triHashA);
                cb.SetComputeBufferParam(_fullRebuildShader, k, "_TriHashB", _triHashB);
                cb.SetComputeBufferParam(_fullRebuildShader, k, "_TriHashC", _triHashC);
                cb.SetComputeBufferParam(_fullRebuildShader, k, "_TriHashState", _triHashState);
                cb.SetComputeBufferParam(_fullRebuildShader, k, "_TriRaw", _triRaw);
                cb.SetComputeBufferParam(_fullRebuildShader, k, "_TriReject", _triReject);
                cb.SetComputeBufferParam(_fullRebuildShader, k, "_TriTemp", _triTemp);

                cb.SetComputeBufferParam(_fullRebuildShader, k, "_SiteSeenInTri", _siteSeenInTri);
                cb.SetComputeBufferParam(_fullRebuildShader, k, "_MissingSites", _missingSites);

                cb.SetComputeBufferParam(_fullRebuildShader, k, "_EdgeHashSrc", _edgeHashSrc);
                cb.SetComputeBufferParam(_fullRebuildShader, k, "_EdgeHashDst", _edgeHashDst);
                cb.SetComputeBufferParam(_fullRebuildShader, k, "_EdgeHashHE", _edgeHashHE);
                cb.SetComputeBufferParam(_fullRebuildShader, k, "_EdgeHashState", _edgeHashState);

                cb.SetComputeBufferParam(_fullRebuildShader, k, "_EdgeRecHash", _edgeRecHash);
                cb.SetComputeBufferParam(_fullRebuildShader, k, "_EdgeRecA", _edgeRecA);
                cb.SetComputeBufferParam(_fullRebuildShader, k, "_EdgeRecB", _edgeRecB);
                cb.SetComputeBufferParam(_fullRebuildShader, k, "_EdgeRecTri", _edgeRecTri);
                cb.SetComputeBufferParam(_fullRebuildShader, k, "_EdgeRecOpp", _edgeRecOpp);

                cb.SetComputeBufferParam(_fullRebuildShader, k, "_RebuildCounters", _rebuildCounters);
            }
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
            if (_ownsFullRebuildShaderInstance && _fullRebuildShader)
                UnityEngine.Object.Destroy(_fullRebuildShader);
        }

        private void DisposeBuffers() {
            for (int i = 0; i < 3; i++) {
                _positions[i]?.Dispose();
                _positions[i] = null;
                _halfEdges[i]?.Dispose();
                _halfEdges[i] = null;
                _triToHE[i]?.Dispose();
                _triToHE[i] = null;
                _triInternal[i]?.Dispose();
                _triInternal[i] = null;
            }

            _triLocks?.Dispose();
            _triLocks = null;
            _vToE?.Dispose();
            _vToE = null;
            _debug?.Dispose();
            _debug = null;
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
            _boundaryEdgeFlags?.Dispose();
            _boundaryEdgeFlags = null;
            _boundaryNormals?.Dispose();
            _boundaryNormals = null;
            _ownerByVertex?.Dispose();
            _ownerByVertex = null;
            _gridSeedOwner?.Dispose();
            _gridSeedOwner = null;
            _voronoiA?.Dispose();
            _voronoiA = null;
            _voronoiB?.Dispose();
            _voronoiB = null;
            _triHashA?.Dispose();
            _triHashA = null;
            _triHashB?.Dispose();
            _triHashB = null;
            _triHashC?.Dispose();
            _triHashC = null;
            _triHashState?.Dispose();
            _triHashState = null;
            _triRaw?.Dispose();
            _triRaw = null;
            _triReject?.Dispose();
            _triReject = null;
            _triTemp?.Dispose();
            _triTemp = null;
            _siteSeenInTri?.Dispose();
            _siteSeenInTri = null;
            _missingSites?.Dispose();
            _missingSites = null;
            _edgeHashSrc?.Dispose();
            _edgeHashSrc = null;
            _edgeHashDst?.Dispose();
            _edgeHashDst = null;
            _edgeHashHE?.Dispose();
            _edgeHashHE = null;
            _edgeHashState?.Dispose();
            _edgeHashState = null;
            _edgeRecHash?.Dispose();
            _edgeRecHash = null;
            _edgeRecA?.Dispose();
            _edgeRecA = null;
            _edgeRecB?.Dispose();
            _edgeRecB = null;
            _edgeRecTri?.Dispose();
            _edgeRecTri = null;
            _edgeRecOpp?.Dispose();
            _edgeRecOpp = null;
            _rebuildCounters?.Dispose();
            _rebuildCounters = null;
            _frIndirectArgs?.Dispose();
            _frIndirectArgs = null;
            _boundsPartials?.Dispose();
            _boundsPartials = null;
            _boundsResult?.Dispose();
            _boundsResult = null;

            _positionScratch = null;
            _ownerByVertexInit = null;
            _vertexCount = _realVertexCount = _halfEdgeCount = _triCount = _neighborCount = 0;
            _renderSlot = 0;
            _rebuildGridW = _rebuildGridH = _rebuildTriHashSize = _rebuildEdgeHashSize = _rebuildMaxEdgeRecords = 0;
            _boundsReadbackPending = false;
            _hasLatestWorldBounds = false;
            _latestWorldBoundsMin = 0f;
            _latestWorldBoundsMax = 0f;
        }
    }
}