using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace GPU.Delaunay {
    public sealed class DT : IDisposable {
        public struct HalfEdge {
            public int v;
            public int next;
            public int twin;
            public int t;
        }

        readonly ComputeShader shader;
        readonly bool ownsShaderInstance;

        // Triple‑buffered for renderer
        ComputeBuffer[] positions = new ComputeBuffer[3];
        ComputeBuffer[] halfEdges = new ComputeBuffer[3];
        ComputeBuffer[] triToHE = new ComputeBuffer[3];

        // Single working buffers (shared across all slots)
        ComputeBuffer triLocks;
        ComputeBuffer vToE;
        ComputeBuffer neighbors;
        ComputeBuffer neighborCounts;
        ComputeBuffer flipCount;

        float2[] positionScratch;
        float2[] superPoints;
        readonly uint[] flipScratch = { 0u };

        int _renderSlot;                // slot used for rendering (0,1,2)

        int vertexCount;
        int realVertexCount;
        int halfEdgeCount;
        int triCount;

        readonly int kClearTriLocks;
        readonly int kClearVertexToEdge;
        readonly int kBuildVertexToEdge;
        readonly int kBuildNeighbors;
        readonly int kFixHalfEdges;
        readonly int kLegalizeHalfEdges;
        readonly int kClearTriToHE;
        readonly int kBuildRenderableTriToHE;

        public int HalfEdgeCount => halfEdgeCount;
        public int TriCount => triCount;
        public int NeighborCount { get; private set; }

        public ComputeBuffer NeighborsBuffer => neighbors;
        public ComputeBuffer NeighborCountsBuffer => neighborCounts;
        // Renderer-facing properties – use the current render slot.
        public ComputeBuffer PositionsBuffer => positions[_renderSlot];
        public ComputeBuffer HalfEdgesBuffer => halfEdges[_renderSlot];
        public ComputeBuffer TriToHEBuffer => triToHE[_renderSlot];

        public DT(ComputeShader shader) {
            if (!shader) throw new ArgumentNullException(nameof(shader));
            this.shader = UnityEngine.Object.Instantiate(shader);
            ownsShaderInstance = true;

            kClearTriLocks = this.shader.FindKernel("ClearTriLocks");
            kClearVertexToEdge = this.shader.FindKernel("ClearVertexToEdge");
            kBuildVertexToEdge = this.shader.FindKernel("BuildVertexToEdge");
            kBuildNeighbors = this.shader.FindKernel("BuildNeighbors");
            kFixHalfEdges = this.shader.FindKernel("FixHalfEdges");
            kLegalizeHalfEdges = this.shader.FindKernel("LegalizeHalfEdges");
            kClearTriToHE = this.shader.FindKernel("ClearTriToHE");
            kBuildRenderableTriToHE = this.shader.FindKernel("BuildRenderableTriToHE");
        }

        public ComputeBuffer GetPositionsBuffer(int slot) => positions[slot];
        public ComputeBuffer GetHalfEdgesBuffer(int slot) => halfEdges[slot];
        public ComputeBuffer GetTriToHEBuffer(int slot) => triToHE[slot];

        // Shared working buffers
        ComputeBuffer GetTriLocksBuffer() => triLocks;
        ComputeBuffer GetVToEBuffer() => vToE;
        ComputeBuffer GetNeighborsBuffer() => neighbors;
        ComputeBuffer GetNeighborCountsBuffer() => neighborCounts;
        ComputeBuffer GetFlipCountBuffer() => flipCount;

        public void Init(
            IReadOnlyList<float2> allPoints,
            int realPointCount,
            DTBuilder.HalfEdge[] initialHalfEdges,
            int triangleCount,
            int neighborCount
        ) {
            DisposeBuffers();

            if (allPoints == null) throw new ArgumentNullException(nameof(allPoints));
            if (initialHalfEdges == null) throw new ArgumentNullException(nameof(initialHalfEdges));
            if (realPointCount <= 0) throw new ArgumentOutOfRangeException(nameof(realPointCount));
            if (triangleCount < 0) throw new ArgumentOutOfRangeException(nameof(triangleCount));
            if (neighborCount <= 0) throw new ArgumentOutOfRangeException(nameof(neighborCount));

            vertexCount = allPoints.Count;
            realVertexCount = realPointCount;

            if (vertexCount < realVertexCount) throw new ArgumentException("VertexCount < realPointCount.", nameof(allPoints));
            if (vertexCount != realVertexCount + 3) throw new ArgumentException("Expected real points + 3 super points.", nameof(allPoints));

            halfEdgeCount = initialHalfEdges.Length;
            if (halfEdgeCount == 0) throw new ArgumentException("Half-edge buffer is empty.", nameof(initialHalfEdges));
            if ((halfEdgeCount % 3) != 0) throw new ArgumentException("Half-edge count must be a multiple of 3.", nameof(initialHalfEdges));

            triCount = triangleCount;
            NeighborCount = neighborCount;

            positionScratch = new float2[vertexCount];
            superPoints = new float2[3] {
                allPoints[realVertexCount + 0],
                allPoints[realVertexCount + 1],
                allPoints[realVertexCount + 2],
            };

            // Create three copies of render buffers
            for (int i = 0; i < 3; i++) {
                positions[i] = new ComputeBuffer(vertexCount, sizeof(float) * 2, ComputeBufferType.Structured);
                halfEdges[i] = new ComputeBuffer(halfEdgeCount, sizeof(int) * 4, ComputeBufferType.Structured);
                triToHE[i] = new ComputeBuffer(triCount, sizeof(int), ComputeBufferType.Structured);
            }

            // Create single working buffers
            triLocks = new ComputeBuffer(triCount, sizeof(int), ComputeBufferType.Structured);
            vToE = new ComputeBuffer(vertexCount, sizeof(int), ComputeBufferType.Structured);
            neighbors = new ComputeBuffer(realVertexCount * neighborCount, sizeof(int), ComputeBufferType.Structured);
            neighborCounts = new ComputeBuffer(realVertexCount, sizeof(int), ComputeBufferType.Structured);
            flipCount = new ComputeBuffer(1, sizeof(uint), ComputeBufferType.Structured);

            // Fill position data into all three slots
            for (int idx = 0; idx < vertexCount; idx++)
                positionScratch[idx] = allPoints[idx];
            for (int i = 0; i < 3; i++)
                positions[i].SetData(positionScratch);

            // Fill half-edge data into all three slots
            var he = new HalfEdge[halfEdgeCount];
            for (int idx = 0; idx < halfEdgeCount; idx++) {
                he[idx] = new HalfEdge {
                    v = initialHalfEdges[idx].v,
                    next = initialHalfEdges[idx].next,
                    twin = initialHalfEdges[idx].twin,
                    t = initialHalfEdges[idx].t,
                };
            }
            for (int i = 0; i < 3; i++)
                halfEdges[i].SetData(he);

            // Initial adjacency and tri map for all slots (uses shared working buffers)
            for (int i = 0; i < 3; i++) {
                RebuildVertexAdjacencyAndTriMapForSlotSync(i);
            }

            _renderSlot = 0;
        }

        void BindCommon() {
            shader.SetInt("_VertexCount", vertexCount);
            shader.SetInt("_RealVertexCount", realVertexCount);
            shader.SetInt("_HalfEdgeCount", halfEdgeCount);
            shader.SetInt("_TriCount", triCount);
            shader.SetInt("_NeighborCount", NeighborCount);
        }

        void EnqueueSetCommonParams(CommandBuffer cb) {
            cb.SetComputeIntParam(shader, "_VertexCount", vertexCount);
            cb.SetComputeIntParam(shader, "_RealVertexCount", realVertexCount);
            cb.SetComputeIntParam(shader, "_HalfEdgeCount", halfEdgeCount);
            cb.SetComputeIntParam(shader, "_TriCount", triCount);
            cb.SetComputeIntParam(shader, "_NeighborCount", NeighborCount);
        }

        void EnqueueDispatchClearTriLocks(CommandBuffer cb) {
            cb.SetComputeBufferParam(shader, kClearTriLocks, "_TriLocks", GetTriLocksBuffer());
            cb.DispatchCompute(shader, kClearTriLocks, (triCount + 255) / 256, 1, 1);
        }

        /// <summary>
        /// Enqueues maintenance on the specified slot.
        /// </summary>
        public void EnqueueMaintain(
            CommandBuffer cb,
            ComputeBuffer positionsForMaintain,
            int writeSlot,
            int fixIterations,
            int legalizeIterations,
            bool rebuildAdjacencyAndTriMap = true
        ) {
            if (cb == null) throw new ArgumentNullException(nameof(cb));
            if (positionsForMaintain == null) throw new ArgumentNullException(nameof(positionsForMaintain));
            if (writeSlot < 0 || writeSlot > 2) throw new ArgumentOutOfRangeException(nameof(writeSlot));

            EnqueueSetCommonParams(cb);

            // Bind shared working buffers (same for all slots)
            cb.SetComputeBufferParam(shader, kClearTriLocks, "_TriLocks", GetTriLocksBuffer());
            cb.SetComputeBufferParam(shader, kFixHalfEdges, "_TriLocks", GetTriLocksBuffer());
            cb.SetComputeBufferParam(shader, kFixHalfEdges, "_FlipCount", GetFlipCountBuffer());
            cb.SetComputeBufferParam(shader, kLegalizeHalfEdges, "_TriLocks", GetTriLocksBuffer());
            cb.SetComputeBufferParam(shader, kLegalizeHalfEdges, "_FlipCount", GetFlipCountBuffer());

            cb.SetComputeBufferParam(shader, kClearVertexToEdge, "_VToE", GetVToEBuffer());
            cb.SetComputeBufferParam(shader, kBuildVertexToEdge, "_VToE", GetVToEBuffer());
            cb.SetComputeBufferParam(shader, kBuildNeighbors, "_VToE", GetVToEBuffer());
            cb.SetComputeBufferParam(shader, kBuildNeighbors, "_Neighbors", GetNeighborsBuffer());
            cb.SetComputeBufferParam(shader, kBuildNeighbors, "_NeighborCounts", GetNeighborCountsBuffer());

            // Bind slot‑specific render buffers (the ones we're writing to)
            cb.SetComputeBufferParam(shader, kFixHalfEdges, "_HalfEdges", GetHalfEdgesBuffer(writeSlot));
            cb.SetComputeBufferParam(shader, kLegalizeHalfEdges, "_HalfEdges", GetHalfEdgesBuffer(writeSlot));
            cb.SetComputeBufferParam(shader, kBuildVertexToEdge, "_HalfEdges", GetHalfEdgesBuffer(writeSlot));
            cb.SetComputeBufferParam(shader, kBuildNeighbors, "_HalfEdges", GetHalfEdgesBuffer(writeSlot));
            cb.SetComputeBufferParam(shader, kClearTriToHE, "_TriToHE", GetTriToHEBuffer(writeSlot));
            cb.SetComputeBufferParam(shader, kBuildRenderableTriToHE, "_HalfEdges", GetHalfEdgesBuffer(writeSlot));
            cb.SetComputeBufferParam(shader, kBuildRenderableTriToHE, "_TriToHE", GetTriToHEBuffer(writeSlot));

            // Bind positions (may be same as slot's positions or a separate velocity buffer)
            cb.SetComputeBufferParam(shader, kFixHalfEdges, "_Positions", positionsForMaintain);
            cb.SetComputeBufferParam(shader, kLegalizeHalfEdges, "_Positions", positionsForMaintain);

            int groups = (halfEdgeCount + 255) / 256;
            var flipCountBuffer = GetFlipCountBuffer();

            for (int i = 0; i < fixIterations; i++) {
                EnqueueDispatchClearTriLocks(cb);
                cb.SetBufferData(flipCountBuffer, flipScratch);
                cb.DispatchCompute(shader, kFixHalfEdges, groups, 1, 1);
            }

            for (int i = 0; i < legalizeIterations; i++) {
                EnqueueDispatchClearTriLocks(cb);
                cb.SetBufferData(flipCountBuffer, flipScratch);
                cb.DispatchCompute(shader, kLegalizeHalfEdges, groups, 1, 1);
            }

            if (rebuildAdjacencyAndTriMap) {
                EnqueueRebuildVertexAdjacencyAndTriMap(cb, writeSlot);
            }
        }

        /// <summary>
        /// Enqueues rebuild of adjacency and tri‑to‑HE map for a specific slot.
        /// </summary>
        public void EnqueueRebuildVertexAdjacencyAndTriMap(CommandBuffer cb, int slot) {
            if (cb == null) throw new ArgumentNullException(nameof(cb));
            if (slot < 0 || slot > 2) throw new ArgumentOutOfRangeException(nameof(slot));

            EnqueueSetCommonParams(cb);

            // Shared working buffers
            cb.SetComputeBufferParam(shader, kClearVertexToEdge, "_VToE", GetVToEBuffer());
            cb.SetComputeBufferParam(shader, kBuildVertexToEdge, "_VToE", GetVToEBuffer());
            cb.SetComputeBufferParam(shader, kBuildNeighbors, "_VToE", GetVToEBuffer());
            cb.SetComputeBufferParam(shader, kBuildNeighbors, "_Neighbors", GetNeighborsBuffer());
            cb.SetComputeBufferParam(shader, kBuildNeighbors, "_NeighborCounts", GetNeighborCountsBuffer());

            // Slot‑specific buffers
            cb.SetComputeBufferParam(shader, kBuildVertexToEdge, "_HalfEdges", GetHalfEdgesBuffer(slot));
            cb.SetComputeBufferParam(shader, kBuildNeighbors, "_HalfEdges", GetHalfEdgesBuffer(slot));
            cb.SetComputeBufferParam(shader, kClearTriToHE, "_TriToHE", GetTriToHEBuffer(slot));
            cb.SetComputeBufferParam(shader, kBuildRenderableTriToHE, "_HalfEdges", GetHalfEdgesBuffer(slot));
            cb.SetComputeBufferParam(shader, kBuildRenderableTriToHE, "_TriToHE", GetTriToHEBuffer(slot));

            cb.DispatchCompute(shader, kClearVertexToEdge, (vertexCount + 255) / 256, 1, 1);
            cb.DispatchCompute(shader, kBuildVertexToEdge, (halfEdgeCount + 255) / 256, 1, 1);
            cb.DispatchCompute(shader, kBuildNeighbors, (realVertexCount + 255) / 256, 1, 1);
            cb.DispatchCompute(shader, kClearTriToHE, (triCount + 255) / 256, 1, 1);
            cb.DispatchCompute(shader, kBuildRenderableTriToHE, (halfEdgeCount + 255) / 256, 1, 1);
        }

        // Synchronous rebuild (used during init)
        void RebuildVertexAdjacencyAndTriMapForSlotSync(int slot) {
            BindCommon();

            shader.SetBuffer(kClearVertexToEdge, "_VToE", GetVToEBuffer());
            shader.SetBuffer(kBuildVertexToEdge, "_VToE", GetVToEBuffer());
            shader.SetBuffer(kBuildNeighbors, "_VToE", GetVToEBuffer());
            shader.SetBuffer(kBuildNeighbors, "_Neighbors", GetNeighborsBuffer());
            shader.SetBuffer(kBuildNeighbors, "_NeighborCounts", GetNeighborCountsBuffer());

            shader.SetBuffer(kBuildVertexToEdge, "_HalfEdges", GetHalfEdgesBuffer(slot));
            shader.SetBuffer(kBuildNeighbors, "_HalfEdges", GetHalfEdgesBuffer(slot));
            shader.SetBuffer(kClearTriToHE, "_TriToHE", GetTriToHEBuffer(slot));
            shader.SetBuffer(kBuildRenderableTriToHE, "_HalfEdges", GetHalfEdgesBuffer(slot));
            shader.SetBuffer(kBuildRenderableTriToHE, "_TriToHE", GetTriToHEBuffer(slot));

            shader.Dispatch(kClearVertexToEdge, (vertexCount + 255) / 256, 1, 1);
            shader.Dispatch(kBuildVertexToEdge, (halfEdgeCount + 255) / 256, 1, 1);
            shader.Dispatch(kBuildNeighbors, (realVertexCount + 255) / 256, 1, 1);
            shader.Dispatch(kClearTriToHE, (triCount + 255) / 256, 1, 1);
            shader.Dispatch(kBuildRenderableTriToHE, (halfEdgeCount + 255) / 256, 1, 1);
        }

        public void Dispose() {
            DisposeBuffers();
            if (ownsShaderInstance && shader)
                UnityEngine.Object.Destroy(shader);
        }

        void DisposeBuffers() {
            for (int i = 0; i < 3; i++) {
                positions[i]?.Dispose(); positions[i] = null;
                halfEdges[i]?.Dispose(); halfEdges[i] = null;
                triToHE[i]?.Dispose(); triToHE[i] = null;
            }

            triLocks?.Dispose(); triLocks = null;
            vToE?.Dispose(); vToE = null;
            neighbors?.Dispose(); neighbors = null;
            neighborCounts?.Dispose(); neighborCounts = null;
            flipCount?.Dispose(); flipCount = null;

            positionScratch = null;
            superPoints = null;

            vertexCount = 0;
            realVertexCount = 0;
            halfEdgeCount = 0;
            triCount = 0;
            NeighborCount = 0;

            _renderSlot = 0;
        }
    }
}