using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace GPU.Delaunay {
    /// <summary>
    /// GPU-side maintenance for a 2D Delaunay triangulation stored as a half-edge structure.
    /// Maintains topology using two phases:
    /// 1) Fixing (restore a valid triangulation after point motion).
    /// 2) Legalizing (Delaunay edge flips based on in-circle test).
    /// </summary>
    public sealed class DT : IDisposable {
        public struct HalfEdge {
            public int v;
            public int next;
            public int twin;
            public int t;
        }

        readonly ComputeShader shader;
        readonly bool ownsShaderInstance;

        ComputeBuffer positions0;
        ComputeBuffer positions1;

        ComputeBuffer halfEdges0;
        ComputeBuffer halfEdges1;

        ComputeBuffer triLocks0;
        ComputeBuffer triLocks1;

        ComputeBuffer vToE0;
        ComputeBuffer vToE1;

        ComputeBuffer neighbors0;
        ComputeBuffer neighbors1;

        ComputeBuffer neighborCounts0;
        ComputeBuffer neighborCounts1;

        ComputeBuffer flipCount0;
        ComputeBuffer flipCount1;

        // Rendering support: tri -> representative half-edge (or -1 if not renderable).
        ComputeBuffer triToHE0;
        ComputeBuffer triToHE1;

        uint adjacencyVersion0;
        uint adjacencyVersion1;

        float2[] positionScratch;
        float2[] superPoints;
        readonly uint[] flipScratch = { 0u };

        int renderPing;
        int pendingRenderPing;
        bool hasPendingRenderPing;

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

        public int VertexCount => vertexCount;
        public int RealVertexCount => realVertexCount;
        public int HalfEdgeCount => halfEdgeCount;
        public int TriCount => triCount;
        public int NeighborCount { get; private set; }

        public int RenderPing => renderPing;

        public uint AdjacencyVersion => GetAdjacencyVersion(renderPing);

        public ComputeBuffer PositionsCurrBuffer => GetPositionsBuffer(renderPing);
        public ComputeBuffer PositionsPrevBuffer => GetPositionsBuffer(renderPing ^ 1);

        // Back-compat: treat "PositionsBuffer" as "current".
        public ComputeBuffer PositionsBuffer => PositionsCurrBuffer;

        // Back-compat: "write" means "other ping" relative to current render ping.
        public ComputeBuffer PositionsWriteBuffer => GetPositionsBuffer(renderPing ^ 1);

        public ComputeBuffer HalfEdgesBuffer => GetHalfEdgesBuffer(renderPing);
        public ComputeBuffer TriToHEBuffer => GetTriToHEBuffer(renderPing);

        public ComputeBuffer NeighborsBuffer => GetNeighborsBuffer(renderPing);
        public ComputeBuffer NeighborCountsBuffer => GetNeighborCountsBuffer(renderPing);

        public DT(ComputeShader shader) {
            if (!shader) throw new ArgumentNullException(nameof(shader));

            // Buffer bindings are stored on the ComputeShader instance.
            // Multiple hierarchy levels must not share one shader object.
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

        public ComputeBuffer GetPositionsBuffer(int ping) => (ping & 1) == 0 ? positions0 : positions1;
        public ComputeBuffer GetHalfEdgesBuffer(int ping) => (ping & 1) == 0 ? halfEdges0 : halfEdges1;
        public ComputeBuffer GetTriToHEBuffer(int ping) => (ping & 1) == 0 ? triToHE0 : triToHE1;
        public ComputeBuffer GetNeighborsBuffer(int ping) => (ping & 1) == 0 ? neighbors0 : neighbors1;
        public ComputeBuffer GetNeighborCountsBuffer(int ping) => (ping & 1) == 0 ? neighborCounts0 : neighborCounts1;

        public uint GetAdjacencyVersion(int ping) => (ping & 1) == 0 ? adjacencyVersion0 : adjacencyVersion1;

        ComputeBuffer GetTriLocksBuffer(int ping) => (ping & 1) == 0 ? triLocks0 : triLocks1;
        ComputeBuffer GetVToEBuffer(int ping) => (ping & 1) == 0 ? vToE0 : vToE1;
        ComputeBuffer GetFlipCountBuffer(int ping) => (ping & 1) == 0 ? flipCount0 : flipCount1;

        public void SetRenderPing(int ping) {
            renderPing = ping & 1;
        }

        public void SetPendingRenderPing(int ping) {
            pendingRenderPing = ping & 1;
            hasPendingRenderPing = true;
        }

        public void CommitPendingRenderPing() {
            if (!hasPendingRenderPing) return;
            renderPing = pendingRenderPing;
            hasPendingRenderPing = false;
        }

        /// <summary>
        /// Initializes GPU buffers for a triangulation that includes 3 "super" vertices.
        /// The super vertices must be appended at the end of <paramref name="allPoints"/>.
        /// They are kept fixed across updates and provide a "virtual outside face" so hull edges can flip.
        /// </summary>
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

            positions0 = new ComputeBuffer(vertexCount, sizeof(float) * 2, ComputeBufferType.Structured);
            positions1 = new ComputeBuffer(vertexCount, sizeof(float) * 2, ComputeBufferType.Structured);

            halfEdges0 = new ComputeBuffer(halfEdgeCount, sizeof(int) * 4, ComputeBufferType.Structured);
            halfEdges1 = new ComputeBuffer(halfEdgeCount, sizeof(int) * 4, ComputeBufferType.Structured);

            triLocks0 = new ComputeBuffer(triCount, sizeof(int), ComputeBufferType.Structured);
            triLocks1 = new ComputeBuffer(triCount, sizeof(int), ComputeBufferType.Structured);

            vToE0 = new ComputeBuffer(vertexCount, sizeof(int), ComputeBufferType.Structured);
            vToE1 = new ComputeBuffer(vertexCount, sizeof(int), ComputeBufferType.Structured);

            neighbors0 = new ComputeBuffer(realVertexCount * neighborCount, sizeof(int), ComputeBufferType.Structured);
            neighbors1 = new ComputeBuffer(realVertexCount * neighborCount, sizeof(int), ComputeBufferType.Structured);

            neighborCounts0 = new ComputeBuffer(realVertexCount, sizeof(int), ComputeBufferType.Structured);
            neighborCounts1 = new ComputeBuffer(realVertexCount, sizeof(int), ComputeBufferType.Structured);

            flipCount0 = new ComputeBuffer(1, sizeof(uint), ComputeBufferType.Structured);
            flipCount1 = new ComputeBuffer(1, sizeof(uint), ComputeBufferType.Structured);

            triToHE0 = new ComputeBuffer(triCount, sizeof(int), ComputeBufferType.Structured);
            triToHE1 = new ComputeBuffer(triCount, sizeof(int), ComputeBufferType.Structured);

            for (int i = 0; i < vertexCount; i++)
                positionScratch[i] = allPoints[i];

            positions0.SetData(positionScratch);
            positions1.SetData(positionScratch);

            var he = new HalfEdge[halfEdgeCount];
            for (int i = 0; i < halfEdgeCount; i++) {
                he[i] = new HalfEdge {
                    v = initialHalfEdges[i].v,
                    next = initialHalfEdges[i].next,
                    twin = initialHalfEdges[i].twin,
                    t = initialHalfEdges[i].t,
                };
            }

            halfEdges0.SetData(he);
            halfEdges1.SetData(he);

            renderPing = 0;
            pendingRenderPing = 0;
            hasPendingRenderPing = false;

            adjacencyVersion0 = 0;
            adjacencyVersion1 = 0;

            BindCommon();
            RebuildVertexAdjacencyAndTriMapForPingSync(0);
            RebuildVertexAdjacencyAndTriMapForPingSync(1);
        }

        void BindCommon() {
            shader.SetInt("_VertexCount", vertexCount);
            shader.SetInt("_RealVertexCount", realVertexCount);
            shader.SetInt("_HalfEdgeCount", halfEdgeCount);
            shader.SetInt("_TriCount", triCount);
            shader.SetInt("_NeighborCount", NeighborCount);
        }

        public void BindPositionsForMaintain(ComputeBuffer pos) {
            shader.SetBuffer(kFixHalfEdges, "_Positions", pos);
            shader.SetBuffer(kLegalizeHalfEdges, "_Positions", pos);
        }

        public void EnqueueBindPositionsForMaintain(CommandBuffer cb, ComputeBuffer pos) {
            cb.SetComputeBufferParam(shader, kFixHalfEdges, "_Positions", pos);
            cb.SetComputeBufferParam(shader, kLegalizeHalfEdges, "_Positions", pos);
        }

        void BindBuffersForPing(CommandBuffer cb, int ping, ComputeBuffer positionsForMaintain) {
            EnqueueSetCommonParams(cb);

            cb.SetComputeBufferParam(shader, kClearTriLocks, "_TriLocks", GetTriLocksBuffer(ping));

            cb.SetComputeBufferParam(shader, kFixHalfEdges, "_HalfEdges", GetHalfEdgesBuffer(ping));
            cb.SetComputeBufferParam(shader, kFixHalfEdges, "_TriLocks", GetTriLocksBuffer(ping));
            cb.SetComputeBufferParam(shader, kFixHalfEdges, "_FlipCount", GetFlipCountBuffer(ping));
            cb.SetComputeBufferParam(shader, kFixHalfEdges, "_Positions", positionsForMaintain);

            cb.SetComputeBufferParam(shader, kLegalizeHalfEdges, "_HalfEdges", GetHalfEdgesBuffer(ping));
            cb.SetComputeBufferParam(shader, kLegalizeHalfEdges, "_TriLocks", GetTriLocksBuffer(ping));
            cb.SetComputeBufferParam(shader, kLegalizeHalfEdges, "_FlipCount", GetFlipCountBuffer(ping));
            cb.SetComputeBufferParam(shader, kLegalizeHalfEdges, "_Positions", positionsForMaintain);

            cb.SetComputeBufferParam(shader, kClearVertexToEdge, "_VToE", GetVToEBuffer(ping));

            cb.SetComputeBufferParam(shader, kBuildVertexToEdge, "_HalfEdges", GetHalfEdgesBuffer(ping));
            cb.SetComputeBufferParam(shader, kBuildVertexToEdge, "_VToE", GetVToEBuffer(ping));

            cb.SetComputeBufferParam(shader, kBuildNeighbors, "_HalfEdges", GetHalfEdgesBuffer(ping));
            cb.SetComputeBufferParam(shader, kBuildNeighbors, "_VToE", GetVToEBuffer(ping));
            cb.SetComputeBufferParam(shader, kBuildNeighbors, "_Neighbors", GetNeighborsBuffer(ping));
            cb.SetComputeBufferParam(shader, kBuildNeighbors, "_NeighborCounts", GetNeighborCountsBuffer(ping));

            cb.SetComputeBufferParam(shader, kClearTriToHE, "_TriToHE", GetTriToHEBuffer(ping));

            cb.SetComputeBufferParam(shader, kBuildRenderableTriToHE, "_HalfEdges", GetHalfEdgesBuffer(ping));
            cb.SetComputeBufferParam(shader, kBuildRenderableTriToHE, "_TriToHE", GetTriToHEBuffer(ping));
        }

        void EnqueueSetCommonParams(CommandBuffer cb) {
            cb.SetComputeIntParam(shader, "_VertexCount", vertexCount);
            cb.SetComputeIntParam(shader, "_RealVertexCount", realVertexCount);
            cb.SetComputeIntParam(shader, "_HalfEdgeCount", halfEdgeCount);
            cb.SetComputeIntParam(shader, "_TriCount", triCount);
            cb.SetComputeIntParam(shader, "_NeighborCount", NeighborCount);
        }

        void EnqueueDispatchClearTriLocks(CommandBuffer cb, int ping) {
            cb.SetComputeBufferParam(shader, kClearTriLocks, "_TriLocks", GetTriLocksBuffer(ping));
            cb.DispatchCompute(shader, kClearTriLocks, (triCount + 255) / 256, 1, 1);
        }

        public void EnqueueMaintain(
            CommandBuffer cb,
            ComputeBuffer positionsForMaintain,
            int writePing,
            int fixIterations,
            int legalizeIterations,
            bool rebuildAdjacencyAndTriMap = true
        ) {
            if (cb == null) throw new ArgumentNullException(nameof(cb));
            if (positionsForMaintain == null) throw new ArgumentNullException(nameof(positionsForMaintain));

            int ping = writePing & 1;

            BindBuffersForPing(cb, ping, positionsForMaintain);

            int groups = (halfEdgeCount + 255) / 256;
            var flipCount = GetFlipCountBuffer(ping);

            for (int i = 0; i < fixIterations; i++) {
                EnqueueDispatchClearTriLocks(cb, ping);
                cb.SetBufferData(flipCount, flipScratch);
                cb.DispatchCompute(shader, kFixHalfEdges, groups, 1, 1);
            }

            for (int i = 0; i < legalizeIterations; i++) {
                EnqueueDispatchClearTriLocks(cb, ping);
                cb.SetBufferData(flipCount, flipScratch);
                cb.DispatchCompute(shader, kLegalizeHalfEdges, groups, 1, 1);
            }

            if (rebuildAdjacencyAndTriMap) {
                EnqueueRebuildVertexAdjacencyAndTriMap(cb, ping);

                if (ping == 0) adjacencyVersion0++;
                else adjacencyVersion1++;
            }
        }

        public void EnqueueRebuildVertexAdjacencyAndTriMap(CommandBuffer cb, int writePing) {
            if (cb == null) throw new ArgumentNullException(nameof(cb));

            int ping = writePing & 1;

            EnqueueSetCommonParams(cb);

            cb.SetComputeBufferParam(shader, kClearVertexToEdge, "_VToE", GetVToEBuffer(ping));

            cb.SetComputeBufferParam(shader, kBuildVertexToEdge, "_HalfEdges", GetHalfEdgesBuffer(ping));
            cb.SetComputeBufferParam(shader, kBuildVertexToEdge, "_VToE", GetVToEBuffer(ping));

            cb.SetComputeBufferParam(shader, kBuildNeighbors, "_HalfEdges", GetHalfEdgesBuffer(ping));
            cb.SetComputeBufferParam(shader, kBuildNeighbors, "_VToE", GetVToEBuffer(ping));
            cb.SetComputeBufferParam(shader, kBuildNeighbors, "_Neighbors", GetNeighborsBuffer(ping));
            cb.SetComputeBufferParam(shader, kBuildNeighbors, "_NeighborCounts", GetNeighborCountsBuffer(ping));

            cb.SetComputeBufferParam(shader, kClearTriToHE, "_TriToHE", GetTriToHEBuffer(ping));

            cb.SetComputeBufferParam(shader, kBuildRenderableTriToHE, "_HalfEdges", GetHalfEdgesBuffer(ping));
            cb.SetComputeBufferParam(shader, kBuildRenderableTriToHE, "_TriToHE", GetTriToHEBuffer(ping));

            cb.DispatchCompute(shader, kClearVertexToEdge, (vertexCount + 255) / 256, 1, 1);
            cb.DispatchCompute(shader, kBuildVertexToEdge, (halfEdgeCount + 255) / 256, 1, 1);
            cb.DispatchCompute(shader, kBuildNeighbors, (realVertexCount + 255) / 256, 1, 1);

            cb.DispatchCompute(shader, kClearTriToHE, (triCount + 255) / 256, 1, 1);
            cb.DispatchCompute(shader, kBuildRenderableTriToHE, (halfEdgeCount + 255) / 256, 1, 1);
        }

        public void Maintain(int fixIterations, int legalizeIterations) {
            int writePing = renderPing ^ 1;
            int groups = (halfEdgeCount + 255) / 256;

            BindCommon();
            shader.SetBuffer(kFixHalfEdges, "_Positions", GetPositionsBuffer(writePing));
            shader.SetBuffer(kLegalizeHalfEdges, "_Positions", GetPositionsBuffer(writePing));

            shader.SetBuffer(kClearTriLocks, "_TriLocks", GetTriLocksBuffer(writePing));

            shader.SetBuffer(kFixHalfEdges, "_HalfEdges", GetHalfEdgesBuffer(writePing));
            shader.SetBuffer(kFixHalfEdges, "_TriLocks", GetTriLocksBuffer(writePing));
            shader.SetBuffer(kFixHalfEdges, "_FlipCount", GetFlipCountBuffer(writePing));

            shader.SetBuffer(kLegalizeHalfEdges, "_HalfEdges", GetHalfEdgesBuffer(writePing));
            shader.SetBuffer(kLegalizeHalfEdges, "_TriLocks", GetTriLocksBuffer(writePing));
            shader.SetBuffer(kLegalizeHalfEdges, "_FlipCount", GetFlipCountBuffer(writePing));

            shader.SetBuffer(kClearVertexToEdge, "_VToE", GetVToEBuffer(writePing));

            shader.SetBuffer(kBuildVertexToEdge, "_HalfEdges", GetHalfEdgesBuffer(writePing));
            shader.SetBuffer(kBuildVertexToEdge, "_VToE", GetVToEBuffer(writePing));

            shader.SetBuffer(kBuildNeighbors, "_HalfEdges", GetHalfEdgesBuffer(writePing));
            shader.SetBuffer(kBuildNeighbors, "_VToE", GetVToEBuffer(writePing));
            shader.SetBuffer(kBuildNeighbors, "_Neighbors", GetNeighborsBuffer(writePing));
            shader.SetBuffer(kBuildNeighbors, "_NeighborCounts", GetNeighborCountsBuffer(writePing));

            shader.SetBuffer(kClearTriToHE, "_TriToHE", GetTriToHEBuffer(writePing));

            shader.SetBuffer(kBuildRenderableTriToHE, "_HalfEdges", GetHalfEdgesBuffer(writePing));
            shader.SetBuffer(kBuildRenderableTriToHE, "_TriToHE", GetTriToHEBuffer(writePing));

            for (int i = 0; i < fixIterations; i++) {
                shader.Dispatch(kClearTriLocks, (triCount + 255) / 256, 1, 1);
                flipScratch[0] = 0u;
                GetFlipCountBuffer(writePing).SetData(flipScratch);
                shader.Dispatch(kFixHalfEdges, groups, 1, 1);
            }

            for (int i = 0; i < legalizeIterations; i++) {
                shader.Dispatch(kClearTriLocks, (triCount + 255) / 256, 1, 1);
                flipScratch[0] = 0u;
                GetFlipCountBuffer(writePing).SetData(flipScratch);
                shader.Dispatch(kLegalizeHalfEdges, groups, 1, 1);
            }

            RebuildVertexAdjacencyAndTriMapForPingSync(writePing);

            if (writePing == 0) adjacencyVersion0++;
            else adjacencyVersion1++;

            renderPing = writePing;
        }

        void RebuildVertexAdjacencyAndTriMapForPingSync(int ping) {
            BindCommon();

            shader.SetBuffer(kClearVertexToEdge, "_VToE", GetVToEBuffer(ping));

            shader.SetBuffer(kBuildVertexToEdge, "_HalfEdges", GetHalfEdgesBuffer(ping));
            shader.SetBuffer(kBuildVertexToEdge, "_VToE", GetVToEBuffer(ping));

            shader.SetBuffer(kBuildNeighbors, "_HalfEdges", GetHalfEdgesBuffer(ping));
            shader.SetBuffer(kBuildNeighbors, "_VToE", GetVToEBuffer(ping));
            shader.SetBuffer(kBuildNeighbors, "_Neighbors", GetNeighborsBuffer(ping));
            shader.SetBuffer(kBuildNeighbors, "_NeighborCounts", GetNeighborCountsBuffer(ping));

            shader.SetBuffer(kClearTriToHE, "_TriToHE", GetTriToHEBuffer(ping));

            shader.SetBuffer(kBuildRenderableTriToHE, "_HalfEdges", GetHalfEdgesBuffer(ping));
            shader.SetBuffer(kBuildRenderableTriToHE, "_TriToHE", GetTriToHEBuffer(ping));

            shader.Dispatch(kClearVertexToEdge, (vertexCount + 255) / 256, 1, 1);
            shader.Dispatch(kBuildVertexToEdge, (halfEdgeCount + 255) / 256, 1, 1);
            shader.Dispatch(kBuildNeighbors, (realVertexCount + 255) / 256, 1, 1);

            shader.Dispatch(kClearTriToHE, (triCount + 255) / 256, 1, 1);
            shader.Dispatch(kBuildRenderableTriToHE, (halfEdgeCount + 255) / 256, 1, 1);
        }

        public void UpdatePositionsFromNodesPrefix(List<Node> nodes, float2 center, float invHalfExtent) {
            if (nodes == null) throw new ArgumentNullException(nameof(nodes));
            if (nodes.Count < realVertexCount) throw new ArgumentException("Node count is smaller than RealVertexCount.", nameof(nodes));

            for (int i = 0; i < realVertexCount; i++)
                positionScratch[i] = (nodes[i].pos - center) * invHalfExtent;

            positionScratch[realVertexCount + 0] = superPoints[0];
            positionScratch[realVertexCount + 1] = superPoints[1];
            positionScratch[realVertexCount + 2] = superPoints[2];

            positions0.SetData(positionScratch);
            positions1.SetData(positionScratch);
            renderPing = 0;
        }

        public uint GetLastFlipCount() {
            var fc = GetFlipCountBuffer(renderPing);
            if (fc == null) return 0u;
            fc.GetData(flipScratch);
            return flipScratch[0];
        }

        // Back-compat for old call sites (no longer used by the async solver).
        public void SwapPositionsAfterTick() => renderPing ^= 1;
        public void SwapTopologyAfterTick() => renderPing ^= 1;

        public void GetHalfEdges(HalfEdge[] dst) {
            if (dst == null) throw new ArgumentNullException(nameof(dst));
            if (dst.Length != halfEdgeCount) throw new ArgumentException("Wrong destination length.", nameof(dst));
            GetHalfEdgesBuffer(renderPing).GetData(dst);
        }

        public void Dispose() {
            DisposeBuffers();

            if (ownsShaderInstance && shader)
                UnityEngine.Object.Destroy(shader);
        }

        void DisposeBuffers() {
            positions0?.Dispose(); positions0 = null;
            positions1?.Dispose(); positions1 = null;

            halfEdges0?.Dispose(); halfEdges0 = null;
            halfEdges1?.Dispose(); halfEdges1 = null;

            triLocks0?.Dispose(); triLocks0 = null;
            triLocks1?.Dispose(); triLocks1 = null;

            vToE0?.Dispose(); vToE0 = null;
            vToE1?.Dispose(); vToE1 = null;

            neighbors0?.Dispose(); neighbors0 = null;
            neighbors1?.Dispose(); neighbors1 = null;

            neighborCounts0?.Dispose(); neighborCounts0 = null;
            neighborCounts1?.Dispose(); neighborCounts1 = null;

            flipCount0?.Dispose(); flipCount0 = null;
            flipCount1?.Dispose(); flipCount1 = null;

            triToHE0?.Dispose(); triToHE0 = null;
            triToHE1?.Dispose(); triToHE1 = null;

            positionScratch = null;
            superPoints = null;

            vertexCount = 0;
            realVertexCount = 0;
            halfEdgeCount = 0;
            triCount = 0;

            NeighborCount = 0;

            renderPing = 0;
            pendingRenderPing = 0;
            hasPendingRenderPing = false;

            adjacencyVersion0 = 0;
            adjacencyVersion1 = 0;
        }
    }
}
