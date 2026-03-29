using System;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEngine.Rendering;

namespace GPU.Delaunay {
    public sealed partial class DT {
        
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
        private readonly int _kernelClearBoundaryData;
        private readonly int _kernelBuildTriToHEAll;
        private readonly int _kernelClassifyInternalTriangles;
        private readonly int _kernelMarkBoundaryEdges;
        private readonly int _kernelBuildBoundaryNormals;
        private readonly int _kernelReduceBounds;
        private readonly int _kernelFinalizeBounds;

        private readonly ComputeShader _fullRebuildShader;
        private readonly bool _ownsFullRebuildShaderInstance;
        private readonly int _frKernelBoundsReducePartials;
        private readonly int _frKernelBoundsFinalize;
        private readonly int _frKernelClearRebuildGrid;
        private readonly int _frKernelClearTriangleHash;
        private readonly int _frKernelClearEdgeHash;
        private readonly int _frKernelClearMeshState;
        private readonly int _frKernelSeedSitesToGrid;
        private readonly int _frKernelAssignOwnersByCell;
        private readonly int _frKernelInitVoronoiFromSeeds;
        private readonly int _frKernelJumpFloodAtoB;
        private readonly int _frKernelJumpFloodBtoA;
        private readonly int _frKernelRemoveIslandsAtoB;
        private readonly int _frKernelRemoveIslandsBtoA;
        private readonly int _frKernelExtractTrianglesFromVoronoi;
        private readonly int _frKernelCompactTrianglesFromHash;
        private readonly int _frKernelClearTriangleRejectFlags;
        private readonly int _frKernelClearEdgeRecords;
        private readonly int _frKernelEmitTriangleEdgeRecords;
        private readonly int _frKernelSortEdgeRecordsBitonic;
        private readonly int _frKernelRejectTrianglesFromSortedEdgeRecords;
        private readonly int _frKernelResetFilteredTriCounter;
        private readonly int _frKernelCompactValidTrianglesToTemp;
        private readonly int _frKernelCopyFilteredTrianglesBack;
        private readonly int _frKernelFinalizeFilteredTriCount;
        private readonly int _frKernelInitAllocatorsFromTriCount;
        private readonly int _frKernelBuildHalfEdgesFromTriangles;
        private readonly int _frKernelBuildDirectedEdgeHash;
        private readonly int _frKernelResolveTwinsFromEdgeHash;
        private readonly int _frKernelBuildVertexToEdgeAndBoundary;

        private ComputeBuffer _gridSeedOwner;
        private ComputeBuffer _voronoiA;
        private ComputeBuffer _voronoiB;
        private ComputeBuffer _triHashA;
        private ComputeBuffer _triHashB;
        private ComputeBuffer _triHashC;
        private ComputeBuffer _triHashState;
        private ComputeBuffer _triRaw;
        private ComputeBuffer _siteSeenInTri;
        private ComputeBuffer _missingSites;
        private ComputeBuffer _edgeHashSrc;
        private ComputeBuffer _edgeHashDst;
        private ComputeBuffer _edgeHashHE;
        private ComputeBuffer _edgeHashState;
        private ComputeBuffer _triReject;
        private ComputeBuffer _triTemp;
        private ComputeBuffer _edgeRecHash;
        private ComputeBuffer _edgeRecA;
        private ComputeBuffer _edgeRecB;
        private ComputeBuffer _edgeRecTri;
        private ComputeBuffer _edgeRecOpp;
        private ComputeBuffer _rebuildCounters;

        private int _rebuildGridW;
        private int _rebuildGridH;
        private int _rebuildTriHashSize;
        private int _rebuildEdgeHashSize;
        private int _rebuildMaxEdgeRecords;

        private int[] _ownerByVertexInit;

        //---------------------------------------------------------------------------
        // Buffers – triple buffered for rendering, single working buffers for topology ops
        //---------------------------------------------------------------------------
        private readonly ComputeBuffer[] _positions = new ComputeBuffer[3];      // vertex positions (read‑only in kernels)
        private readonly ComputeBuffer[] _halfEdges = new ComputeBuffer[3];      // half‑edge arrays (modified during flips)
        private readonly ComputeBuffer[] _triToHE = new ComputeBuffer[3];        // triangle → representative half‑edge (for rendering)

        private ComputeBuffer _triLocks;           // per‑triangle lock (0 = free)
        private ComputeBuffer _vToE;                // vertex → outgoing half‑edge
        private ComputeBuffer _debug;               // debug buffer
        private ComputeBuffer _neighbors;           // flat neighbour list per real vertex
        private ComputeBuffer _neighborCounts;      // number of neighbours per real vertex
        private ComputeBuffer _flipCount;           // global flip counter (single uint)
        private ComputeBuffer _dirtyVertexFlags;    // per‑real‑vertex dirty flag (1 if neighbour list needs rebuild)
        private ComputeBuffer _triToHEAll;          // triangle -> representative half-edge (all triangles)
        private readonly ComputeBuffer[] _triInternal = new ComputeBuffer[3];      // 1 if triangle classified internal
        private ComputeBuffer _boundaryEdgeFlags;   // 1 if half-edge classified boundary
        private ComputeBuffer _boundaryNormals;     // per-real-vertex outward normal
        private ComputeBuffer _ownerByVertex;       // owner id per real vertex
        private ComputeBuffer _boundsPartials;      // per-group bounds partials
        private ComputeBuffer _boundsResult;        // single float4: min.xy, max.xy

        private bool _boundsReadbackPending;
        private bool _hasLatestWorldBounds;
        private float2 _latestWorldBoundsMin;
        private float2 _latestWorldBoundsMax;

        //---------------------------------------------------------------------------
        // Construction
        //---------------------------------------------------------------------------
        /// <summary>
        /// Creates a new DT instance using a copy of the given compute shader.
        /// </summary>
        /// <param name="shader">Compute shader containing all required kernels.</param>
        public DT(ComputeShader shader, ComputeShader fullRebuildShader = null) {
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
            _kernelClearBoundaryData = _shader.FindKernel("ClearBoundaryData");
            _kernelBuildTriToHEAll = _shader.FindKernel("BuildTriToHEAll");
            _kernelClassifyInternalTriangles = _shader.FindKernel("ClassifyInternalTriangles");
            _kernelMarkBoundaryEdges = _shader.FindKernel("MarkBoundaryEdges");
            _kernelBuildBoundaryNormals = _shader.FindKernel("BuildBoundaryNormals");
            _kernelReduceBounds = _shader.FindKernel("ReduceBounds");
            _kernelFinalizeBounds = _shader.FindKernel("FinalizeBounds");

            if (fullRebuildShader != null) {
                _fullRebuildShader = UnityEngine.Object.Instantiate(fullRebuildShader);
                _ownsFullRebuildShaderInstance = true;

                _frKernelBoundsReducePartials = _fullRebuildShader.FindKernel("BoundsReducePartials");
                _frKernelBoundsFinalize = _fullRebuildShader.FindKernel("BoundsFinalize");
                _frKernelClearRebuildGrid = _fullRebuildShader.FindKernel("ClearRebuildGrid");
                _frKernelClearTriangleHash = _fullRebuildShader.FindKernel("ClearTriangleHash");
                _frKernelClearEdgeHash = _fullRebuildShader.FindKernel("ClearEdgeHash");
                _frKernelClearMeshState = _fullRebuildShader.FindKernel("ClearMeshState");
                _frKernelSeedSitesToGrid = _fullRebuildShader.FindKernel("SeedSitesToGrid");
                _frKernelAssignOwnersByCell = _fullRebuildShader.FindKernel("AssignOwnersByCell");
                _frKernelInitVoronoiFromSeeds = _fullRebuildShader.FindKernel("InitVoronoiFromSeeds");
                _frKernelJumpFloodAtoB = _fullRebuildShader.FindKernel("JumpFloodAtoB");
                _frKernelJumpFloodBtoA = _fullRebuildShader.FindKernel("JumpFloodBtoA");
                _frKernelRemoveIslandsAtoB = _fullRebuildShader.FindKernel("RemoveIslandsAtoB");
                _frKernelRemoveIslandsBtoA = _fullRebuildShader.FindKernel("RemoveIslandsBtoA");
                _frKernelExtractTrianglesFromVoronoi = _fullRebuildShader.FindKernel("ExtractTrianglesFromVoronoi");
                _frKernelCompactTrianglesFromHash = _fullRebuildShader.FindKernel("CompactTrianglesFromHash");
                _frKernelClearTriangleRejectFlags = _fullRebuildShader.FindKernel("ClearTriangleRejectFlags");
                _frKernelClearEdgeRecords = _fullRebuildShader.FindKernel("ClearEdgeRecords");
                _frKernelEmitTriangleEdgeRecords = _fullRebuildShader.FindKernel("EmitTriangleEdgeRecords");
                _frKernelSortEdgeRecordsBitonic = _fullRebuildShader.FindKernel("SortEdgeRecordsBitonic");
                _frKernelRejectTrianglesFromSortedEdgeRecords = _fullRebuildShader.FindKernel("RejectTrianglesFromSortedEdgeRecords");
                _frKernelResetFilteredTriCounter = _fullRebuildShader.FindKernel("ResetFilteredTriCounter");
                _frKernelCompactValidTrianglesToTemp = _fullRebuildShader.FindKernel("CompactValidTrianglesToTemp");
                _frKernelCopyFilteredTrianglesBack = _fullRebuildShader.FindKernel("CopyFilteredTrianglesBack");
                _frKernelFinalizeFilteredTriCount = _fullRebuildShader.FindKernel("FinalizeFilteredTriCount");
                _frKernelInitAllocatorsFromTriCount = _fullRebuildShader.FindKernel("InitAllocatorsFromTriCount");
                _frKernelBuildHalfEdgesFromTriangles = _fullRebuildShader.FindKernel("BuildHalfEdgesFromTriangles");
                _frKernelBuildDirectedEdgeHash = _fullRebuildShader.FindKernel("BuildDirectedEdgeHash");
                _frKernelResolveTwinsFromEdgeHash = _fullRebuildShader.FindKernel("ResolveTwinsFromEdgeHash");
                _frKernelBuildVertexToEdgeAndBoundary = _fullRebuildShader.FindKernel("BuildVertexToEdgeAndBoundary");
            } else {
                _frKernelBoundsReducePartials = -1;
                _frKernelBoundsFinalize = -1;
                _frKernelClearRebuildGrid = -1;
                _frKernelClearTriangleHash = -1;
                _frKernelClearEdgeHash = -1;
                _frKernelClearMeshState = -1;
                _frKernelSeedSitesToGrid = -1;
                _frKernelAssignOwnersByCell = -1;
                _frKernelInitVoronoiFromSeeds = -1;
                _frKernelJumpFloodAtoB = -1;
                _frKernelJumpFloodBtoA = -1;
                _frKernelRemoveIslandsAtoB = -1;
                _frKernelRemoveIslandsBtoA = -1;
                _frKernelExtractTrianglesFromVoronoi = -1;
                _frKernelCompactTrianglesFromHash = -1;
                _frKernelClearTriangleRejectFlags = -1;
                _frKernelClearEdgeRecords = -1;
                _frKernelEmitTriangleEdgeRecords = -1;
                _frKernelSortEdgeRecordsBitonic = -1;
                _frKernelRejectTrianglesFromSortedEdgeRecords = -1;
                _frKernelResetFilteredTriCounter = -1;
                _frKernelCompactValidTrianglesToTemp = -1;
                _frKernelCopyFilteredTrianglesBack = -1;
                _frKernelFinalizeFilteredTriCount = -1;
                _frKernelInitAllocatorsFromTriCount = -1;
                _frKernelBuildHalfEdgesFromTriangles = -1;
                _frKernelBuildDirectedEdgeHash = -1;
                _frKernelResolveTwinsFromEdgeHash = -1;
                _frKernelBuildVertexToEdgeAndBoundary = -1;
            }
        }

        public bool TryGetLatestWorldBounds(out float2 min, out float2 max) {
            min = default;
            max = default;
            if (!_hasLatestWorldBounds)
                return false;

            min = _latestWorldBoundsMin;
            max = _latestWorldBoundsMax;
            return true;
        }

        public void RequestWorldBoundsAsync(int slot, float2 normCenter, float normInvHalfExtent) {
            if (_boundsReadbackPending)
                return;

            if (slot < 0 || slot > 2)
                return;

            if (_positions[slot] == null || !_positions[slot].IsValid())
                return;

            if (_realVertexCount <= 0)
                return;

            int groups = (_realVertexCount + 255) / 256;
            EnsureBoundsBuffers(groups);
            if (_boundsPartials == null || _boundsResult == null)
                return;

            _shader.SetInt("_VertexCount", _vertexCount);
            _shader.SetInt("_RealVertexCount", _realVertexCount);
            _shader.SetInt("_BoundsPartialCount", groups);

            _shader.SetBuffer(_kernelReduceBounds, "_Positions", _positions[slot]);
            _shader.SetBuffer(_kernelReduceBounds, "_BoundsPartials", _boundsPartials);
            _shader.SetBuffer(_kernelFinalizeBounds, "_BoundsPartials", _boundsPartials);
            _shader.SetBuffer(_kernelFinalizeBounds, "_BoundsResult", _boundsResult);

            _shader.Dispatch(_kernelReduceBounds, groups, 1, 1);
            _shader.Dispatch(_kernelFinalizeBounds, 1, 1, 1);

            _boundsReadbackPending = true;
            AsyncGPUReadback.Request(_boundsResult, req => {
                _boundsReadbackPending = false;
                if (req.hasError || _boundsResult == null || !_boundsResult.IsValid())
                    return;

                var data = req.GetData<float4>();
                if (data.Length < 1)
                    return;

                float4 r = data[0];
                float inv = normInvHalfExtent > 1e-8f ? (1f / normInvHalfExtent) : 1f;
                _latestWorldBoundsMin = r.xy * inv + normCenter;
                _latestWorldBoundsMax = r.zw * inv + normCenter;
                _hasLatestWorldBounds = true;
            });
        }

        private void EnsureBoundsBuffers(int groupCount) {
            int needed = math.max(1, groupCount);
            if (_boundsPartials == null || !_boundsPartials.IsValid() || _boundsPartials.count != needed) {
                _boundsPartials?.Dispose();
                _boundsPartials = new ComputeBuffer(needed, sizeof(float) * 4, ComputeBufferType.Structured);
            }

            if (_boundsResult == null || !_boundsResult.IsValid() || _boundsResult.count != 1) {
                _boundsResult?.Dispose();
                _boundsResult = new ComputeBuffer(1, sizeof(float) * 4, ComputeBufferType.Structured);
            }
        }

        private void PrepareBuildBuffers(
            CommandBuffer cb,
            ComputeBuffer positionsForMaintain,
            int writeSlot) {
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
            cb.SetComputeBufferParam(_shader, _kernelBuildNeighbors, "_Positions", positionsForMaintain);
            cb.SetComputeBufferParam(_shader, _kernelClearTriToHE, "_TriToHE", _triToHE[writeSlot]);
            cb.SetComputeBufferParam(_shader, _kernelBuildRenderableTriToHE, "_HalfEdges", _halfEdges[writeSlot]);
            cb.SetComputeBufferParam(_shader, _kernelBuildRenderableTriToHE, "_TriToHE", _triToHE[writeSlot]);
            cb.SetComputeBufferParam(_shader, _kernelBuildTriToHEAll, "_HalfEdges", _halfEdges[writeSlot]);
            cb.SetComputeBufferParam(_shader, _kernelBuildTriToHEAll, "_TriToHEAll", _triToHEAll);
            cb.SetComputeBufferParam(_shader, _kernelClassifyInternalTriangles, "_HalfEdges", _halfEdges[writeSlot]);
            cb.SetComputeBufferParam(_shader, _kernelClassifyInternalTriangles, "_TriToHEAll", _triToHEAll);
            cb.SetComputeBufferParam(_shader, _kernelClassifyInternalTriangles, "_TriInternal", _triInternal[writeSlot]);
            cb.SetComputeBufferParam(_shader, _kernelClassifyInternalTriangles, "_OwnerByVertex", _ownerByVertex);
            cb.SetComputeBufferParam(_shader, _kernelClassifyInternalTriangles, "_Positions", positionsForMaintain);
            cb.SetComputeBufferParam(_shader, _kernelMarkBoundaryEdges, "_HalfEdges", _halfEdges[writeSlot]);
            cb.SetComputeBufferParam(_shader, _kernelMarkBoundaryEdges, "_TriInternal", _triInternal[writeSlot]);
            cb.SetComputeBufferParam(_shader, _kernelMarkBoundaryEdges, "_BoundaryEdgeFlags", _boundaryEdgeFlags);
            cb.SetComputeBufferParam(_shader, _kernelBuildBoundaryNormals, "_HalfEdges", _halfEdges[writeSlot]);
            cb.SetComputeBufferParam(_shader, _kernelBuildBoundaryNormals, "_VToE", _vToE);
            cb.SetComputeBufferParam(_shader, _kernelBuildBoundaryNormals, "_TriInternal", _triInternal[writeSlot]);
            cb.SetComputeBufferParam(_shader, _kernelBuildBoundaryNormals, "_BoundaryEdgeFlags", _boundaryEdgeFlags);
            cb.SetComputeBufferParam(_shader, _kernelBuildBoundaryNormals, "_BoundaryNormals", _boundaryNormals);
            cb.SetComputeBufferParam(_shader, _kernelBuildBoundaryNormals, "_Positions", positionsForMaintain);
            cb.SetComputeBufferParam(_shader, _kernelClearBoundaryData, "_TriToHEAll", _triToHEAll);
            cb.SetComputeBufferParam(_shader, _kernelClearBoundaryData, "_TriInternal", _triInternal[writeSlot]);
            cb.SetComputeBufferParam(_shader, _kernelClearBoundaryData, "_BoundaryEdgeFlags", _boundaryEdgeFlags);
            cb.SetComputeBufferParam(_shader, _kernelClearBoundaryData, "_BoundaryNormals", _boundaryNormals);

            // Bind positions (may be same as slot's positions or a separate updated buffer).
            cb.SetComputeBufferParam(_shader, _kernelFixHalfEdges, "_Positions", positionsForMaintain);
            cb.SetComputeBufferParam(_shader, _kernelLegalizeHalfEdges, "_Positions", positionsForMaintain);
        }

        private void SetCommonParams(CommandBuffer cb) {
            cb.SetComputeIntParam(_shader, "_VertexCount", _vertexCount);
            cb.SetComputeIntParam(_shader, "_RealVertexCount", _realVertexCount);
            cb.SetComputeIntParam(_shader, "_HalfEdgeCount", _halfEdgeCount);
            cb.SetComputeIntParam(_shader, "_TriCount", _triCount);
            cb.SetComputeIntParam(_shader, "_NeighborCount", _neighborCount);
            cb.SetComputeIntParam(_shader, "_UseSupportRadiusFilter", _useSupportRadiusFilter ? 1 : 0);
            cb.SetComputeFloatParam(_shader, "_SupportRadius2", _supportRadius2);
        }

        private void DispatchClearTriLocks(CommandBuffer cb) {
            cb.SetComputeBufferParam(_shader, _kernelClearTriLocks, "_TriLocks", _triLocks);
            Dispatch(cb, _shader, _kernelClearTriLocks, (_triCount + 255) / 256, 1, 1, MarkerPrefix + "ClearTriLocks");
        }

        private void DispatchClearDirtyVertexFlags(CommandBuffer cb) {
            cb.SetComputeBufferParam(_shader, _kernelClearDirtyVertexFlags, "_DirtyVertexFlags", _dirtyVertexFlags);
            Dispatch(cb, _shader, _kernelClearDirtyVertexFlags, (_realVertexCount + 255) / 256, 1, 1, MarkerPrefix + "ClearDirtyVertexFlags");
        }

        private static void Dispatch(CommandBuffer cb, ComputeShader shader, int kernel, int x, int y, int z, string marker) {
            cb.BeginSample(marker);
            cb.DispatchCompute(shader, kernel, x, y, z);
            cb.EndSample(marker);
        }

        private static void Dispatch(ComputeShader shader, int kernel, int x, int y, int z, string marker) {
            Profiler.BeginSample(marker);
            shader.Dispatch(kernel, x, y, z);
            Profiler.EndSample();
        }

        /// <summary>
        /// Enqueues a full rebuild of vertex neighbour lists and the triangle-to-halfedge map for a specific slot.
        /// </summary>
        public void EnqueueRebuildVertexAdjacencyAndTriMap(CommandBuffer cb, int slot) {
            if (cb == null) throw new ArgumentNullException(nameof(cb));
            if (slot < 0 || slot > 2) throw new ArgumentOutOfRangeException(nameof(slot));

            SetCommonParams(cb);

            cb.SetComputeBufferParam(_shader, _kernelClearVertexToEdge, "_VToE", _vToE);
            cb.SetComputeBufferParam(_shader, _kernelBuildVertexToEdge, "_VToE", _vToE);
            cb.SetComputeBufferParam(_shader, _kernelBuildNeighbors, "_VToE", _vToE);
            cb.SetComputeBufferParam(_shader, _kernelBuildNeighbors, "_Neighbors", _neighbors);
            cb.SetComputeBufferParam(_shader, _kernelBuildNeighbors, "_NeighborCounts", _neighborCounts);

            cb.SetComputeBufferParam(_shader, _kernelBuildVertexToEdge, "_HalfEdges", _halfEdges[slot]);
            cb.SetComputeBufferParam(_shader, _kernelBuildNeighbors, "_HalfEdges", _halfEdges[slot]);
            cb.SetComputeBufferParam(_shader, _kernelBuildNeighbors, "_Positions", _positions[slot]);
            cb.SetComputeBufferParam(_shader, _kernelClearTriToHE, "_TriToHE", _triToHE[slot]);
            cb.SetComputeBufferParam(_shader, _kernelBuildRenderableTriToHE, "_HalfEdges", _halfEdges[slot]);
            cb.SetComputeBufferParam(_shader, _kernelBuildRenderableTriToHE, "_TriToHE", _triToHE[slot]);

            Dispatch(cb, _shader, _kernelClearVertexToEdge, (_vertexCount + 255) / 256, 1, 1, MarkerPrefix + "ClearVertexToEdge");
            Dispatch(cb, _shader, _kernelBuildVertexToEdge, (_halfEdgeCount + 255) / 256, 1, 1, MarkerPrefix + "BuildVertexToEdge");
            Dispatch(cb, _shader, _kernelBuildNeighbors, (_realVertexCount + 255) / 256, 1, 1, MarkerPrefix + "BuildNeighbors");
            Dispatch(cb, _shader, _kernelClearTriToHE, (_triCount + 255) / 256, 1, 1, MarkerPrefix + "ClearTriToHE");
            Dispatch(cb, _shader, _kernelBuildRenderableTriToHE, (_halfEdgeCount + 255) / 256, 1, 1, MarkerPrefix + "BuildRenderableTriToHE");
        }

        /// <summary>
        /// Synchronous version of EnqueueRebuildVertexAdjacencyAndTriMap (used during init).
        /// </summary>
        private void RebuildVertexAdjacencyAndTriMapSync(int slot) {
            _shader.SetInt("_VertexCount", _vertexCount);
            _shader.SetInt("_RealVertexCount", _realVertexCount);
            _shader.SetInt("_HalfEdgeCount", _halfEdgeCount);
            _shader.SetInt("_TriCount", _triCount);
            _shader.SetInt("_NeighborCount", _neighborCount);
            _shader.SetInt("_UseSupportRadiusFilter", _useSupportRadiusFilter ? 1 : 0);
            _shader.SetFloat("_SupportRadius2", _supportRadius2);

            _shader.SetBuffer(_kernelClearVertexToEdge, "_VToE", _vToE);
            _shader.SetBuffer(_kernelBuildVertexToEdge, "_VToE", _vToE);
            _shader.SetBuffer(_kernelBuildNeighbors, "_VToE", _vToE);
            _shader.SetBuffer(_kernelBuildNeighbors, "_Neighbors", _neighbors);
            _shader.SetBuffer(_kernelBuildNeighbors, "_NeighborCounts", _neighborCounts);

            _shader.SetBuffer(_kernelBuildVertexToEdge, "_HalfEdges", _halfEdges[slot]);
            _shader.SetBuffer(_kernelBuildNeighbors, "_HalfEdges", _halfEdges[slot]);
            _shader.SetBuffer(_kernelBuildNeighbors, "_Positions", _positions[slot]);
            _shader.SetBuffer(_kernelClearTriToHE, "_TriToHE", _triToHE[slot]);
            _shader.SetBuffer(_kernelBuildRenderableTriToHE, "_HalfEdges", _halfEdges[slot]);
            _shader.SetBuffer(_kernelBuildRenderableTriToHE, "_TriToHE", _triToHE[slot]);

            Dispatch(_shader, _kernelClearVertexToEdge, (_vertexCount + 255) / 256, 1, 1, MarkerPrefix + "ClearVertexToEdgeSync");
            Dispatch(_shader, _kernelBuildVertexToEdge, (_halfEdgeCount + 255) / 256, 1, 1, MarkerPrefix + "BuildVertexToEdgeSync");
            Dispatch(_shader, _kernelBuildNeighbors, (_realVertexCount + 255) / 256, 1, 1, MarkerPrefix + "BuildNeighborsSync");
            Dispatch(_shader, _kernelClearTriToHE, (_triCount + 255) / 256, 1, 1, MarkerPrefix + "ClearTriToHESync");
            Dispatch(_shader, _kernelBuildRenderableTriToHE, (_halfEdgeCount + 255) / 256, 1, 1, MarkerPrefix + "BuildRenderableTriToHESync");
        }
    }
}
