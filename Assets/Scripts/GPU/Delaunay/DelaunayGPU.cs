using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

namespace GPU.Delaunay {
    /// <summary>
    /// GPU-side maintenance for a 2D Delaunay triangulation stored as a half-edge structure.
    /// Maintains topology using two phases:
    /// 1) Fixing (restore a valid triangulation after point motion).
    /// 2) Legalizing (Delaunay edge flips based on in-circle test).
    /// </summary>
    public sealed class DelaunayGpu : IDisposable {
        public struct HalfEdge {
            public int v;
            public int next;
            public int twin;
            public int t;
        }

        readonly ComputeShader shader;
        readonly bool ownsShaderInstance;

        ComputeBuffer positions;
        ComputeBuffer halfEdges;
        ComputeBuffer triLocks;

        ComputeBuffer vToE;
        ComputeBuffer neighbors;
        ComputeBuffer neighborCounts;

        ComputeBuffer flipCount;

        // Rendering support: tri -> representative half-edge (or -1 if not renderable).
        ComputeBuffer triToHE;

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

        public ComputeBuffer PositionsBuffer => positions;
        public ComputeBuffer HalfEdgesBuffer => halfEdges;
        public ComputeBuffer TriToHEBuffer => triToHE;

        public DelaunayGpu(ComputeShader shader) {
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

            positions = new ComputeBuffer(vertexCount, sizeof(float) * 2, ComputeBufferType.Structured);
            halfEdges = new ComputeBuffer(halfEdgeCount, sizeof(int) * 4, ComputeBufferType.Structured);
            triLocks = new ComputeBuffer(triCount, sizeof(int), ComputeBufferType.Structured);

            vToE = new ComputeBuffer(vertexCount, sizeof(int), ComputeBufferType.Structured);
            neighbors = new ComputeBuffer(realVertexCount * neighborCount, sizeof(int), ComputeBufferType.Structured);
            neighborCounts = new ComputeBuffer(realVertexCount, sizeof(int), ComputeBufferType.Structured);

            flipCount = new ComputeBuffer(1, sizeof(uint), ComputeBufferType.Structured);

            triToHE = new ComputeBuffer(triCount, sizeof(int), ComputeBufferType.Structured);

            for (int i = 0; i < vertexCount; i++)
                positionScratch[i] = allPoints[i];

            positions.SetData(positionScratch);

            var he = new HalfEdge[halfEdgeCount];
            for (int i = 0; i < halfEdgeCount; i++) {
                he[i] = new HalfEdge {
                    v = initialHalfEdges[i].v,
                    next = initialHalfEdges[i].next,
                    twin = initialHalfEdges[i].twin,
                    t = initialHalfEdges[i].t,
                };
            }
            halfEdges.SetData(he);

            BindCommon();
            DispatchClearTriLocks();
            RebuildVertexAdjacencyAndTriMap();
        }

        void BindCommon() {
            shader.SetInt("_VertexCount", vertexCount);
            shader.SetInt("_RealVertexCount", realVertexCount);
            shader.SetInt("_HalfEdgeCount", halfEdgeCount);
            shader.SetInt("_TriCount", triCount);
            shader.SetInt("_NeighborCount", NeighborCount);

            shader.SetBuffer(kClearTriLocks, "_TriLocks", triLocks);

            shader.SetBuffer(kClearVertexToEdge, "_VToE", vToE);

            shader.SetBuffer(kBuildVertexToEdge, "_HalfEdges", halfEdges);
            shader.SetBuffer(kBuildVertexToEdge, "_VToE", vToE);

            shader.SetBuffer(kBuildNeighbors, "_HalfEdges", halfEdges);
            shader.SetBuffer(kBuildNeighbors, "_VToE", vToE);
            shader.SetBuffer(kBuildNeighbors, "_Neighbors", neighbors);
            shader.SetBuffer(kBuildNeighbors, "_NeighborCounts", neighborCounts);

            shader.SetBuffer(kFixHalfEdges, "_Positions", positions);
            shader.SetBuffer(kFixHalfEdges, "_HalfEdges", halfEdges);
            shader.SetBuffer(kFixHalfEdges, "_TriLocks", triLocks);
            shader.SetBuffer(kFixHalfEdges, "_FlipCount", flipCount);

            shader.SetBuffer(kLegalizeHalfEdges, "_Positions", positions);
            shader.SetBuffer(kLegalizeHalfEdges, "_HalfEdges", halfEdges);
            shader.SetBuffer(kLegalizeHalfEdges, "_TriLocks", triLocks);
            shader.SetBuffer(kLegalizeHalfEdges, "_FlipCount", flipCount);

            shader.SetBuffer(kClearTriToHE, "_TriToHE", triToHE);

            shader.SetBuffer(kBuildRenderableTriToHE, "_HalfEdges", halfEdges);
            shader.SetBuffer(kBuildRenderableTriToHE, "_TriToHE", triToHE);
        }

        public void UpdatePositionsFromNodes(List<Node> nodes) {
            if (nodes == null) throw new ArgumentNullException(nameof(nodes));
            if (nodes.Count != realVertexCount) throw new ArgumentException("Node count doesn't match RealVertexCount.", nameof(nodes));

            for (int i = 0; i < realVertexCount; i++)
                positionScratch[i] = nodes[i].pos;

            positionScratch[realVertexCount + 0] = superPoints[0];
            positionScratch[realVertexCount + 1] = superPoints[1];
            positionScratch[realVertexCount + 2] = superPoints[2];

            positions.SetData(positionScratch);
        }

        public void UpdatePositionsFromNodes(List<Node> nodes, float2 center, float invHalfExtent) {
            if (nodes == null) throw new ArgumentNullException(nameof(nodes));
            if (nodes.Count != realVertexCount) throw new ArgumentException("Node count doesn't match RealVertexCount.", nameof(nodes));

            for (int i = 0; i < realVertexCount; i++)
                positionScratch[i] = (nodes[i].pos - center) * invHalfExtent;

            positionScratch[realVertexCount + 0] = superPoints[0];
            positionScratch[realVertexCount + 1] = superPoints[1];
            positionScratch[realVertexCount + 2] = superPoints[2];

            positions.SetData(positionScratch);
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

            positions.SetData(positionScratch);
        }

        public uint GetLastFlipCount() {
            if (flipCount == null) return 0u;
            flipCount.GetData(flipScratch);
            return flipScratch[0];
        }

        public void Maintain(int fixIterations, int legalizeIterations) {
            int groups = (halfEdgeCount + 255) / 256;

            for (int i = 0; i < fixIterations; i++) {
                DispatchClearTriLocks();
                flipScratch[0] = 0u;
                flipCount.SetData(flipScratch);
                shader.Dispatch(kFixHalfEdges, groups, 1, 1);
            }

            for (int i = 0; i < legalizeIterations; i++) {
                DispatchClearTriLocks();
                flipScratch[0] = 0u;
                flipCount.SetData(flipScratch);
                shader.Dispatch(kLegalizeHalfEdges, groups, 1, 1);
            }

            RebuildVertexAdjacencyAndTriMap();
        }

        public void RebuildVertexAdjacencyAndTriMap() {
            shader.Dispatch(kClearVertexToEdge, (vertexCount + 255) / 256, 1, 1);
            shader.Dispatch(kBuildVertexToEdge, (halfEdgeCount + 255) / 256, 1, 1);
            shader.Dispatch(kBuildNeighbors, (realVertexCount + 255) / 256, 1, 1);

            shader.Dispatch(kClearTriToHE, (triCount + 255) / 256, 1, 1);
            shader.Dispatch(kBuildRenderableTriToHE, (halfEdgeCount + 255) / 256, 1, 1);
        }

        void DispatchClearTriLocks() {
            shader.Dispatch(kClearTriLocks, (triCount + 255) / 256, 1, 1);
        }

        public void GetHalfEdges(HalfEdge[] dst) {
            if (dst == null) throw new ArgumentNullException(nameof(dst));
            if (dst.Length != halfEdgeCount) throw new ArgumentException("Wrong destination length.", nameof(dst));
            halfEdges.GetData(dst);
        }

        public void GetNeighbors(int[] dst) {
            if (dst == null) throw new ArgumentNullException(nameof(dst));
            int expected = realVertexCount * NeighborCount;
            if (dst.Length != expected) throw new ArgumentException($"Wrong destination length (expected {expected}).", nameof(dst));
            neighbors.GetData(dst);
        }

        public void GetNeighborCounts(int[] dst) {
            if (dst == null) throw new ArgumentNullException(nameof(dst));
            if (dst.Length != realVertexCount) throw new ArgumentException($"Wrong destination length (expected {realVertexCount}).", nameof(dst));
            neighborCounts.GetData(dst);
        }

        public void Dispose() {
            DisposeBuffers();

            if (ownsShaderInstance && shader)
                UnityEngine.Object.Destroy(shader);
        }

        void DisposeBuffers() {
            positions?.Dispose(); positions = null;
            halfEdges?.Dispose(); halfEdges = null;
            triLocks?.Dispose(); triLocks = null;
            vToE?.Dispose(); vToE = null;
            neighbors?.Dispose(); neighbors = null;
            neighborCounts?.Dispose(); neighborCounts = null;
            flipCount?.Dispose(); flipCount = null;
            triToHE?.Dispose(); triToHE = null;

            positionScratch = null;
            superPoints = null;

            vertexCount = 0;
            realVertexCount = 0;
            halfEdgeCount = 0;
            triCount = 0;
        }
    }
}
