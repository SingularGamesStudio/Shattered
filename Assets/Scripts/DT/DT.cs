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

        ComputeBuffer positionsA;
        ComputeBuffer positionsB;
        bool positionsAIsCurrent = true;

        // Topology/adjacency is double-buffered so rendering can read "current"
        // while async compute writes into "write", then we swap on fence.
        ComputeBuffer halfEdgesA;
        ComputeBuffer halfEdgesB;
        ComputeBuffer triLocksA;
        ComputeBuffer triLocksB;

        ComputeBuffer vToEA;
        ComputeBuffer vToEB;
        ComputeBuffer neighborsA;
        ComputeBuffer neighborsB;
        ComputeBuffer neighborCountsA;
        ComputeBuffer neighborCountsB;

        ComputeBuffer flipCountA;
        ComputeBuffer flipCountB;

        // Rendering support: tri -> representative half-edge (or -1 if not renderable).
        ComputeBuffer triToHEA;
        ComputeBuffer triToHEB;

        bool topologyAIsCurrent = true;
        uint adjacencyVersionA;
        uint adjacencyVersionB;

        float2[] positionScratch;
        float2[] superPoints;
        readonly uint[] flipScratch = { 0u };

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

        public uint AdjacencyVersion => topologyAIsCurrent ? adjacencyVersionA : adjacencyVersionB;

        public ComputeBuffer PositionsCurrBuffer => positionsAIsCurrent ? positionsA : positionsB;
        public ComputeBuffer PositionsPrevBuffer => positionsAIsCurrent ? positionsB : positionsA;
        public ComputeBuffer PositionsWriteBuffer => PositionsPrevBuffer;

        // Back-compat: treat "PositionsBuffer" as "current".
        public ComputeBuffer PositionsBuffer => PositionsCurrBuffer;

        public ComputeBuffer HalfEdgesBuffer => topologyAIsCurrent ? halfEdgesA : halfEdgesB;
        ComputeBuffer HalfEdgesWriteBuffer => topologyAIsCurrent ? halfEdgesB : halfEdgesA;

        public ComputeBuffer TriToHEBuffer => topologyAIsCurrent ? triToHEA : triToHEB;
        ComputeBuffer TriToHEWriteBuffer => topologyAIsCurrent ? triToHEB : triToHEA;

        public ComputeBuffer NeighborsBuffer => topologyAIsCurrent ? neighborsA : neighborsB;
        ComputeBuffer NeighborsWriteBuffer => topologyAIsCurrent ? neighborsB : neighborsA;

        public ComputeBuffer NeighborCountsBuffer => topologyAIsCurrent ? neighborCountsA : neighborCountsB;

        // Expose write-side buffers so solver can explicitly bind them for async recording.
        public ComputeBuffer HalfEdgesA => halfEdgesA;
        public ComputeBuffer HalfEdgesB => halfEdgesB;
        public ComputeBuffer NeighborsA => neighborsA;
        public ComputeBuffer NeighborsB => neighborsB;
        public ComputeBuffer NeighborCountsA => neighborCountsA;
        public ComputeBuffer NeighborCountsB => neighborCountsB;

        public bool TopologyAIsCurrent => topologyAIsCurrent;
        public bool PositionsAIsCurrent => positionsAIsCurrent;

        ComputeBuffer NeighborCountsWriteBuffer => topologyAIsCurrent ? neighborCountsB : neighborCountsA;

        ComputeBuffer TriLocksWriteBuffer => topologyAIsCurrent ? triLocksB : triLocksA;
        ComputeBuffer VToEWriteBuffer => topologyAIsCurrent ? vToEB : vToEA;
        ComputeBuffer FlipCountWriteBuffer => topologyAIsCurrent ? flipCountB : flipCountA;

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

            positionsA = new ComputeBuffer(vertexCount, sizeof(float) * 2, ComputeBufferType.Structured);
            positionsB = new ComputeBuffer(vertexCount, sizeof(float) * 2, ComputeBufferType.Structured);
            positionsAIsCurrent = true;

            halfEdgesA = new ComputeBuffer(halfEdgeCount, sizeof(int) * 4, ComputeBufferType.Structured);
            halfEdgesB = new ComputeBuffer(halfEdgeCount, sizeof(int) * 4, ComputeBufferType.Structured);

            triLocksA = new ComputeBuffer(triCount, sizeof(int), ComputeBufferType.Structured);
            triLocksB = new ComputeBuffer(triCount, sizeof(int), ComputeBufferType.Structured);

            vToEA = new ComputeBuffer(vertexCount, sizeof(int), ComputeBufferType.Structured);
            vToEB = new ComputeBuffer(vertexCount, sizeof(int), ComputeBufferType.Structured);

            neighborsA = new ComputeBuffer(realVertexCount * neighborCount, sizeof(int), ComputeBufferType.Structured);
            neighborsB = new ComputeBuffer(realVertexCount * neighborCount, sizeof(int), ComputeBufferType.Structured);

            neighborCountsA = new ComputeBuffer(realVertexCount, sizeof(int), ComputeBufferType.Structured);
            neighborCountsB = new ComputeBuffer(realVertexCount, sizeof(int), ComputeBufferType.Structured);

            flipCountA = new ComputeBuffer(1, sizeof(uint), ComputeBufferType.Structured);
            flipCountB = new ComputeBuffer(1, sizeof(uint), ComputeBufferType.Structured);

            triToHEA = new ComputeBuffer(triCount, sizeof(int), ComputeBufferType.Structured);
            triToHEB = new ComputeBuffer(triCount, sizeof(int), ComputeBufferType.Structured);

            for (int i = 0; i < vertexCount; i++)
                positionScratch[i] = allPoints[i];

            positionsA.SetData(positionScratch);
            positionsB.SetData(positionScratch);

            var he = new HalfEdge[halfEdgeCount];
            for (int i = 0; i < halfEdgeCount; i++) {
                he[i] = new HalfEdge {
                    v = initialHalfEdges[i].v,
                    next = initialHalfEdges[i].next,
                    twin = initialHalfEdges[i].twin,
                    t = initialHalfEdges[i].t,
                };
            }

            halfEdgesA.SetData(he);
            halfEdgesB.SetData(he);

            topologyAIsCurrent = true;
            adjacencyVersionA = 0;
            adjacencyVersionB = 0;

            BindCommon();

            // Build adjacency/tri-map for BOTH topology buffers so the first async swap
            // doesn't reveal an uninitialized "other" buffer.
            BindTopologyForMaintainSync(halfEdgesA, triLocksA, flipCountA, vToEA, neighborsA, neighborCountsA, triToHEA, PositionsCurrBuffer);
            DispatchClearTriLocks();
            DispatchRebuildVertexAdjacencyAndTriMapSync(vertexCount, halfEdgeCount, triCount, realVertexCount);
            adjacencyVersionA = 1;

            BindTopologyForMaintainSync(halfEdgesB, triLocksB, flipCountB, vToEB, neighborsB, neighborCountsB, triToHEB, PositionsCurrBuffer);
            DispatchClearTriLocks();
            DispatchRebuildVertexAdjacencyAndTriMapSync(vertexCount, halfEdgeCount, triCount, realVertexCount);
            adjacencyVersionB = 1;

            // Restore bindings to "current" topology for anyone using shader.Dispatch without rebinding.
            BindTopologyForMaintainSync(HalfEdgesBuffer,
                topologyAIsCurrent ? triLocksA : triLocksB,
                topologyAIsCurrent ? flipCountA : flipCountB,
                topologyAIsCurrent ? vToEA : vToEB,
                NeighborsBuffer,
                NeighborCountsBuffer,
                TriToHEBuffer,
                PositionsCurrBuffer);
        }

        void BindCommon() {
            shader.SetInt("_VertexCount", vertexCount);
            shader.SetInt("_RealVertexCount", realVertexCount);
            shader.SetInt("_HalfEdgeCount", halfEdgeCount);
            shader.SetInt("_TriCount", triCount);
            shader.SetInt("_NeighborCount", NeighborCount);

            BindPositionsForMaintain(PositionsCurrBuffer);
        }

        public void BindPositionsForMaintain(ComputeBuffer pos) {
            shader.SetBuffer(kFixHalfEdges, "_Positions", pos);
            shader.SetBuffer(kLegalizeHalfEdges, "_Positions", pos);
        }

        public void EnqueueBindPositionsForMaintain(CommandBuffer cb, ComputeBuffer pos) {
            cb.SetComputeBufferParam(shader, kFixHalfEdges, "_Positions", pos);
            cb.SetComputeBufferParam(shader, kLegalizeHalfEdges, "_Positions", pos);
        }

        public void SwapPositionsAfterTick() {
            positionsAIsCurrent = !positionsAIsCurrent;
            BindPositionsForMaintain(PositionsCurrBuffer);
        }

        public void SwapTopologyAfterTick() {
            topologyAIsCurrent = !topologyAIsCurrent;
        }

        void BindTopologyForMaintainSync(
            ComputeBuffer halfEdges,
            ComputeBuffer triLocks,
            ComputeBuffer flipCount,
            ComputeBuffer vToE,
            ComputeBuffer neighbors,
            ComputeBuffer neighborCounts,
            ComputeBuffer triToHE,
            ComputeBuffer positionsForMaintain
        ) {
            shader.SetInt("_VertexCount", vertexCount);
            shader.SetInt("_RealVertexCount", realVertexCount);
            shader.SetInt("_HalfEdgeCount", halfEdgeCount);
            shader.SetInt("_TriCount", triCount);
            shader.SetInt("_NeighborCount", NeighborCount);

            shader.SetBuffer(kClearTriLocks, "_TriLocks", triLocks);

            shader.SetBuffer(kFixHalfEdges, "_HalfEdges", halfEdges);
            shader.SetBuffer(kFixHalfEdges, "_TriLocks", triLocks);
            shader.SetBuffer(kFixHalfEdges, "_FlipCount", flipCount);
            shader.SetBuffer(kFixHalfEdges, "_Positions", positionsForMaintain);

            shader.SetBuffer(kLegalizeHalfEdges, "_HalfEdges", halfEdges);
            shader.SetBuffer(kLegalizeHalfEdges, "_TriLocks", triLocks);
            shader.SetBuffer(kLegalizeHalfEdges, "_FlipCount", flipCount);
            shader.SetBuffer(kLegalizeHalfEdges, "_Positions", positionsForMaintain);

            shader.SetBuffer(kClearVertexToEdge, "_VToE", vToE);

            shader.SetBuffer(kBuildVertexToEdge, "_HalfEdges", halfEdges);
            shader.SetBuffer(kBuildVertexToEdge, "_VToE", vToE);

            shader.SetBuffer(kBuildNeighbors, "_HalfEdges", halfEdges);
            shader.SetBuffer(kBuildNeighbors, "_VToE", vToE);
            shader.SetBuffer(kBuildNeighbors, "_Neighbors", neighbors);
            shader.SetBuffer(kBuildNeighbors, "_NeighborCounts", neighborCounts);

            shader.SetBuffer(kClearTriToHE, "_TriToHE", triToHE);

            shader.SetBuffer(kBuildRenderableTriToHE, "_HalfEdges", halfEdges);
            shader.SetBuffer(kBuildRenderableTriToHE, "_TriToHE", triToHE);
        }

        void EnqueueSetCommonParams(CommandBuffer cb) {
            cb.SetComputeIntParam(shader, "_VertexCount", vertexCount);
            cb.SetComputeIntParam(shader, "_RealVertexCount", realVertexCount);
            cb.SetComputeIntParam(shader, "_HalfEdgeCount", halfEdgeCount);
            cb.SetComputeIntParam(shader, "_TriCount", triCount);
            cb.SetComputeIntParam(shader, "_NeighborCount", NeighborCount);
        }

        void EnqueueSetWriteBuffers(CommandBuffer cb, ComputeBuffer positionsForMaintain) {
            var he = HalfEdgesWriteBuffer;
            var tl = TriLocksWriteBuffer;
            var fc = FlipCountWriteBuffer;
            var vte = VToEWriteBuffer;
            var n = NeighborsWriteBuffer;
            var nc = NeighborCountsWriteBuffer;
            var t2 = TriToHEWriteBuffer;

            cb.SetComputeBufferParam(shader, kClearTriLocks, "_TriLocks", tl);

            cb.SetComputeBufferParam(shader, kFixHalfEdges, "_HalfEdges", he);
            cb.SetComputeBufferParam(shader, kFixHalfEdges, "_TriLocks", tl);
            cb.SetComputeBufferParam(shader, kFixHalfEdges, "_FlipCount", fc);
            cb.SetComputeBufferParam(shader, kFixHalfEdges, "_Positions", positionsForMaintain);

            cb.SetComputeBufferParam(shader, kLegalizeHalfEdges, "_HalfEdges", he);
            cb.SetComputeBufferParam(shader, kLegalizeHalfEdges, "_TriLocks", tl);
            cb.SetComputeBufferParam(shader, kLegalizeHalfEdges, "_FlipCount", fc);
            cb.SetComputeBufferParam(shader, kLegalizeHalfEdges, "_Positions", positionsForMaintain);

            cb.SetComputeBufferParam(shader, kClearVertexToEdge, "_VToE", vte);

            cb.SetComputeBufferParam(shader, kBuildVertexToEdge, "_HalfEdges", he);
            cb.SetComputeBufferParam(shader, kBuildVertexToEdge, "_VToE", vte);

            cb.SetComputeBufferParam(shader, kBuildNeighbors, "_HalfEdges", he);
            cb.SetComputeBufferParam(shader, kBuildNeighbors, "_VToE", vte);
            cb.SetComputeBufferParam(shader, kBuildNeighbors, "_Neighbors", n);
            cb.SetComputeBufferParam(shader, kBuildNeighbors, "_NeighborCounts", nc);

            cb.SetComputeBufferParam(shader, kClearTriToHE, "_TriToHE", t2);

            cb.SetComputeBufferParam(shader, kBuildRenderableTriToHE, "_HalfEdges", he);
            cb.SetComputeBufferParam(shader, kBuildRenderableTriToHE, "_TriToHE", t2);
        }

        public void UpdatePositionsFromNodes(List<Node> nodes) {
            if (nodes == null) throw new ArgumentNullException(nameof(nodes));
            if (nodes.Count != realVertexCount) throw new ArgumentException("Node count doesn't match RealVertexCount.", nameof(nodes));

            for (int i = 0; i < realVertexCount; i++)
                positionScratch[i] = nodes[i].pos;

            positionScratch[realVertexCount + 0] = superPoints[0];
            positionScratch[realVertexCount + 1] = superPoints[1];
            positionScratch[realVertexCount + 2] = superPoints[2];

            positionsA.SetData(positionScratch);
            positionsB.SetData(positionScratch);
            positionsAIsCurrent = true;
            BindPositionsForMaintain(PositionsCurrBuffer);
        }

        public void UpdatePositionsFromNodes(List<Node> nodes, float2 center, float invHalfExtent) {
            if (nodes == null) throw new ArgumentNullException(nameof(nodes));
            if (nodes.Count != realVertexCount) throw new ArgumentException("Node count doesn't match RealVertexCount.", nameof(nodes));

            for (int i = 0; i < realVertexCount; i++)
                positionScratch[i] = (nodes[i].pos - center) * invHalfExtent;

            positionScratch[realVertexCount + 0] = superPoints[0];
            positionScratch[realVertexCount + 1] = superPoints[1];
            positionScratch[realVertexCount + 2] = superPoints[2];

            positionsA.SetData(positionScratch);
            positionsB.SetData(positionScratch);
            positionsAIsCurrent = true;
            BindPositionsForMaintain(PositionsCurrBuffer);
        }

        // Used by the hierarchy: the DT only owns the prefix [0..RealVertexCount).
        public void UpdatePositionsFromNodesPrefix(List<Node> nodes, float2 center, float invHalfExtent) {
            if (nodes == null) throw new ArgumentNullException(nameof(nodes));
            if (nodes.Count < realVertexCount) throw new ArgumentException("Node count is smaller than RealVertexCount.", nameof(nodes));

            for (int i = 0; i < realVertexCount; i++)
                positionScratch[i] = (nodes[i].pos - center) * invHalfExtent;

            positionScratch[realVertexCount + 0] = superPoints[0];
            positionScratch[realVertexCount + 1] = superPoints[1];
            positionScratch[realVertexCount + 2] = superPoints[2];

            positionsA.SetData(positionScratch);
            positionsB.SetData(positionScratch);
            positionsAIsCurrent = true;
            BindPositionsForMaintain(PositionsCurrBuffer);
        }

        public uint GetLastFlipCount() {
            var fc = topologyAIsCurrent ? flipCountA : flipCountB;
            if (fc == null) return 0u;
            fc.GetData(flipScratch);
            return flipScratch[0];
        }

        public void Maintain(int fixIterations, int legalizeIterations) {
            // Sync path: write into the non-current topology buffers, then swap.
            BindTopologyForMaintainSync(
                HalfEdgesWriteBuffer,
                TriLocksWriteBuffer,
                FlipCountWriteBuffer,
                VToEWriteBuffer,
                NeighborsWriteBuffer,
                NeighborCountsWriteBuffer,
                TriToHEWriteBuffer,
                PositionsCurrBuffer
            );

            int groups = (halfEdgeCount + 255) / 256;

            for (int i = 0; i < fixIterations; i++) {
                DispatchClearTriLocks();
                flipScratch[0] = 0u;
                FlipCountWriteBuffer.SetData(flipScratch);
                shader.Dispatch(kFixHalfEdges, groups, 1, 1);
            }

            for (int i = 0; i < legalizeIterations; i++) {
                DispatchClearTriLocks();
                flipScratch[0] = 0u;
                FlipCountWriteBuffer.SetData(flipScratch);
                shader.Dispatch(kLegalizeHalfEdges, groups, 1, 1);
            }

            DispatchRebuildVertexAdjacencyAndTriMapSync(vertexCount, halfEdgeCount, triCount, realVertexCount);

            uint next = AdjacencyVersion + 1;
            if (topologyAIsCurrent) adjacencyVersionB = next;
            else adjacencyVersionA = next;

            SwapTopologyAfterTick();
        }

        public void EnqueueMaintain(
            CommandBuffer cb,
            ComputeBuffer positionsForMaintain,
            int fixIterations,
            int legalizeIterations,
            bool rebuildAdjacencyAndTriMap = true
        ) {
            if (cb == null) throw new ArgumentNullException(nameof(cb));
            if (positionsForMaintain == null) throw new ArgumentNullException(nameof(positionsForMaintain));

            EnqueueSetCommonParams(cb);
            EnqueueSetWriteBuffers(cb, positionsForMaintain);

            int groups = (halfEdgeCount + 255) / 256;

            for (int i = 0; i < fixIterations; i++) {
                EnqueueDispatchClearTriLocks(cb);
                cb.SetBufferData(FlipCountWriteBuffer, flipScratch);
                cb.DispatchCompute(shader, kFixHalfEdges, groups, 1, 1);
            }

            for (int i = 0; i < legalizeIterations; i++) {
                EnqueueDispatchClearTriLocks(cb);
                cb.SetBufferData(FlipCountWriteBuffer, flipScratch);
                cb.DispatchCompute(shader, kLegalizeHalfEdges, groups, 1, 1);
            }

            if (rebuildAdjacencyAndTriMap) {
                EnqueueRebuildVertexAdjacencyAndTriMap(cb);

                uint next = AdjacencyVersion + 1;
                if (topologyAIsCurrent) adjacencyVersionB = next;
                else adjacencyVersionA = next;
            }
        }

        public void RebuildVertexAdjacencyAndTriMap() {
            // Sync path: rebuild into write buffers, then swap.
            BindTopologyForMaintainSync(
                HalfEdgesWriteBuffer,
                TriLocksWriteBuffer,
                FlipCountWriteBuffer,
                VToEWriteBuffer,
                NeighborsWriteBuffer,
                NeighborCountsWriteBuffer,
                TriToHEWriteBuffer,
                PositionsCurrBuffer
            );

            DispatchRebuildVertexAdjacencyAndTriMapSync(vertexCount, halfEdgeCount, triCount, realVertexCount);

            uint next = AdjacencyVersion + 1;
            if (topologyAIsCurrent) adjacencyVersionB = next;
            else adjacencyVersionA = next;

            SwapTopologyAfterTick();
        }

        public void EnqueueRebuildVertexAdjacencyAndTriMap(CommandBuffer cb) {
            if (cb == null) throw new ArgumentNullException(nameof(cb));

            EnqueueSetCommonParams(cb);

            cb.SetComputeBufferParam(shader, kClearVertexToEdge, "_VToE", VToEWriteBuffer);

            cb.SetComputeBufferParam(shader, kBuildVertexToEdge, "_HalfEdges", HalfEdgesWriteBuffer);
            cb.SetComputeBufferParam(shader, kBuildVertexToEdge, "_VToE", VToEWriteBuffer);

            cb.SetComputeBufferParam(shader, kBuildNeighbors, "_HalfEdges", HalfEdgesWriteBuffer);
            cb.SetComputeBufferParam(shader, kBuildNeighbors, "_VToE", VToEWriteBuffer);
            cb.SetComputeBufferParam(shader, kBuildNeighbors, "_Neighbors", NeighborsWriteBuffer);
            cb.SetComputeBufferParam(shader, kBuildNeighbors, "_NeighborCounts", NeighborCountsWriteBuffer);

            cb.SetComputeBufferParam(shader, kClearTriToHE, "_TriToHE", TriToHEWriteBuffer);

            cb.SetComputeBufferParam(shader, kBuildRenderableTriToHE, "_HalfEdges", HalfEdgesWriteBuffer);
            cb.SetComputeBufferParam(shader, kBuildRenderableTriToHE, "_TriToHE", TriToHEWriteBuffer);

            cb.DispatchCompute(shader, kClearVertexToEdge, (vertexCount + 255) / 256, 1, 1);
            cb.DispatchCompute(shader, kBuildVertexToEdge, (halfEdgeCount + 255) / 256, 1, 1);
            cb.DispatchCompute(shader, kBuildNeighbors, (realVertexCount + 255) / 256, 1, 1);

            cb.DispatchCompute(shader, kClearTriToHE, (triCount + 255) / 256, 1, 1);
            cb.DispatchCompute(shader, kBuildRenderableTriToHE, (halfEdgeCount + 255) / 256, 1, 1);
        }

        void DispatchRebuildVertexAdjacencyAndTriMapSync(int vCount, int heCount, int tCount, int rvCount) {
            shader.Dispatch(kClearVertexToEdge, (vCount + 255) / 256, 1, 1);
            shader.Dispatch(kBuildVertexToEdge, (heCount + 255) / 256, 1, 1);
            shader.Dispatch(kBuildNeighbors, (rvCount + 255) / 256, 1, 1);

            shader.Dispatch(kClearTriToHE, (tCount + 255) / 256, 1, 1);
            shader.Dispatch(kBuildRenderableTriToHE, (heCount + 255) / 256, 1, 1);
        }

        void DispatchClearTriLocks() {
            shader.Dispatch(kClearTriLocks, (triCount + 255) / 256, 1, 1);
        }

        void EnqueueDispatchClearTriLocks(CommandBuffer cb) {
            cb.SetComputeBufferParam(shader, kClearTriLocks, "_TriLocks", TriLocksWriteBuffer);
            cb.DispatchCompute(shader, kClearTriLocks, (triCount + 255) / 256, 1, 1);
        }

        public void GetHalfEdges(HalfEdge[] dst) {
            if (dst == null) throw new ArgumentNullException(nameof(dst));
            if (dst.Length != halfEdgeCount) throw new ArgumentException("Wrong destination length.", nameof(dst));
            HalfEdgesBuffer.GetData(dst);
        }

        public void Dispose() {
            DisposeBuffers();

            if (ownsShaderInstance && shader)
                UnityEngine.Object.Destroy(shader);
        }

        void DisposeBuffers() {
            positionsA?.Dispose(); positionsA = null;
            positionsB?.Dispose(); positionsB = null;

            halfEdgesA?.Dispose(); halfEdgesA = null;
            halfEdgesB?.Dispose(); halfEdgesB = null;

            triLocksA?.Dispose(); triLocksA = null;
            triLocksB?.Dispose(); triLocksB = null;

            vToEA?.Dispose(); vToEA = null;
            vToEB?.Dispose(); vToEB = null;

            neighborsA?.Dispose(); neighborsA = null;
            neighborsB?.Dispose(); neighborsB = null;

            neighborCountsA?.Dispose(); neighborCountsA = null;
            neighborCountsB?.Dispose(); neighborCountsB = null;

            flipCountA?.Dispose(); flipCountA = null;
            flipCountB?.Dispose(); flipCountB = null;

            triToHEA?.Dispose(); triToHEA = null;
            triToHEB?.Dispose(); triToHEB = null;

            positionScratch = null;
            superPoints = null;

            vertexCount = 0;
            realVertexCount = 0;
            halfEdgeCount = 0;
            triCount = 0;

            positionsAIsCurrent = true;
            topologyAIsCurrent = true;
            adjacencyVersionA = 0;
            adjacencyVersionB = 0;
        }
    }
}
