using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

namespace GPU.Delaunay {
    public sealed class DTHierarchy : IDisposable {
        readonly ComputeShader shader;

        DT[] levels;
        int[] levelEndIndex;
        int maxLevel;

        int neighborCount;

        readonly List<DTBuilder.Triangle> triangles = new List<DTBuilder.Triangle>(4096);
        readonly List<int> bad = new List<int>(128);
        readonly HashSet<DirEdge> boundary = new HashSet<DirEdge>(2048);

        float2[] gpuAllScratch;

        struct DirEdge : IEquatable<DirEdge> {
            public int a, b;
            public DirEdge(int a, int b) { this.a = a; this.b = b; }
            public bool Equals(DirEdge other) => a == other.a && b == other.b;
            public override bool Equals(object obj) => obj is DirEdge other && Equals(other);
            public override int GetHashCode() => unchecked((a * 73856093) ^ (b * 19349663));
        }

        public int MaxLevel => maxLevel;
        public int LevelCount => levels?.Length ?? 0;

        public DTHierarchy(ComputeShader shader) {
            this.shader = shader ? shader : throw new ArgumentNullException(nameof(shader));
        }

        public DT GetLevelDt(int level) {
            if (levels == null) return null;
            if ((uint)level >= (uint)levels.Length) return null;
            return levels[level];
        }

        public int GetLevelRealVertexCount(int level) {
            if ((uint)level >= (uint)LevelCount) return 0;
            return levelEndIndex[level];
        }

        public int GetLevelHalfEdgeCount(int level) {
            if ((uint)level >= (uint)LevelCount) return 0;
            return levels[level]?.HalfEdgeCount ?? 0;
        }

        public void InitFromMeshlessNodes(
            List<Node> nodes,
            float2 normCenter,
            float normInvHalfExtent,
            float2 super0,
            float2 super1,
            float2 super2,
            int neighborCount,
            int warmupFixIterations,
            int warmupLegalizeIterations
        ) {
            if (nodes == null) throw new ArgumentNullException(nameof(nodes));
            if (nodes.Count < 3) throw new ArgumentException("Need at least 3 nodes.", nameof(nodes));
            if (neighborCount <= 0) throw new ArgumentOutOfRangeException(nameof(neighborCount));

            this.neighborCount = neighborCount;

            ComputeLevelEndIndex(nodes, out maxLevel, out levelEndIndex);

            DisposeLevels();
            levels = new DT[maxLevel + 1];

            for (int level = 0; level <= maxLevel; level++) {
                int n = levelEndIndex[level];
                if (n < 3) {
                    levels[level] = null;
                    continue;
                }

                EnsureScratch(n + 3);

                for (int i = 0; i < n; i++)
                    gpuAllScratch[i] = (nodes[i].pos - normCenter) * normInvHalfExtent;

                gpuAllScratch[n + 0] = super0;
                gpuAllScratch[n + 1] = super1;
                gpuAllScratch[n + 2] = super2;

                BuildBowyerWatsonWithFixedSuper(gpuAllScratch, n, triangles);

                var he = DTBuilder.BuildHalfEdges(gpuAllScratch, triangles);

                var dt = new DT(shader);
                dt.Init(gpuAllScratch, n, he, triangles.Count, neighborCount);
                dt.Maintain(warmupFixIterations, warmupLegalizeIterations);

                levels[level] = dt;
            }
        }

        public void UpdatePositionsFromNodesAllLevels(List<Node> nodes, float2 normCenter, float normInvHalfExtent) {
            if (levels == null) return;

            for (int level = 0; level < levels.Length; level++) {
                var dt = levels[level];
                if (dt == null) continue;
                dt.UpdatePositionsFromNodesPrefix(nodes, normCenter, normInvHalfExtent);
            }
        }

        public void MaintainAllLevels(int fixIterations, int legalizeIterations) {
            if (levels == null) return;

            for (int level = 0; level < levels.Length; level++) {
                var dt = levels[level];
                if (dt == null) continue;
                dt.Maintain(fixIterations, legalizeIterations);
            }
        }

        public void GetHalfEdges(int level, DT.HalfEdge[] dst) {
            if ((uint)level >= (uint)LevelCount) throw new ArgumentOutOfRangeException(nameof(level));
            if (levels[level] == null) throw new InvalidOperationException("Level DT is not initialized (too few vertices).");
            levels[level].GetHalfEdges(dst);
        }

        public void Dispose() {
            DisposeLevels();
            levels = null;
            levelEndIndex = null;
            maxLevel = 0;
            gpuAllScratch = null;
        }

        void DisposeLevels() {
            if (levels == null) return;
            for (int i = 0; i < levels.Length; i++) {
                levels[i]?.Dispose();
                levels[i] = null;
            }
        }

        void EnsureScratch(int count) {
            if (gpuAllScratch == null || gpuAllScratch.Length != count)
                gpuAllScratch = new float2[count];
        }

        static void ComputeLevelEndIndex(List<Node> nodes, out int maxLevel, out int[] levelEndIndex) {
            maxLevel = 0;
            for (int i = 0; i < nodes.Count; i++)
                maxLevel = math.max(maxLevel, nodes[i].maxLayer);

            levelEndIndex = new int[maxLevel + 1];
            int idx = 0;
            for (int level = maxLevel; level >= 0; level--) {
                for (; idx < nodes.Count && nodes[idx].maxLayer >= level; idx++) { }
                levelEndIndex[level] = idx;
            }
        }

        static float Orient2D(float2 a, float2 b, float2 c) {
            float2 ab = b - a;
            float2 ac = c - a;
            return ab.x * ac.y - ab.y * ac.x;
        }

        static bool InCircleCCW(float2 a, float2 b, float2 c, float2 p) {
            float2 ap = a - p;
            float2 bp = b - p;
            float2 cp = c - p;

            float a2 = math.dot(ap, ap);
            float b2 = math.dot(bp, bp);
            float c2 = math.dot(cp, cp);

            float det =
                ap.x * (bp.y * c2 - b2 * cp.y) -
                ap.y * (bp.x * c2 - b2 * cp.x) +
                a2 * (bp.x * cp.y - bp.y * cp.x);

            return det > 0f;
        }

        static void ToggleEdge(HashSet<DirEdge> set, int a, int b) {
            if (!set.Remove(new DirEdge(b, a)))
                set.Add(new DirEdge(a, b));
        }

        void BuildBowyerWatsonWithFixedSuper(float2[] pointsWithSuper, int realCount, List<DTBuilder.Triangle> outTriangles) {
            outTriangles.Clear();

            int s0 = realCount + 0;
            int s1 = realCount + 1;
            int s2 = realCount + 2;

            if (Orient2D(pointsWithSuper[s0], pointsWithSuper[s1], pointsWithSuper[s2]) < 0f)
                outTriangles.Add(new DTBuilder.Triangle(s0, s2, s1));
            else
                outTriangles.Add(new DTBuilder.Triangle(s0, s1, s2));

            for (int pi = 0; pi < realCount; pi++) {
                float2 p = pointsWithSuper[pi];

                bad.Clear();
                for (int ti = 0; ti < outTriangles.Count; ti++) {
                    var t = outTriangles[ti];

                    float2 a = pointsWithSuper[t.a];
                    float2 b = pointsWithSuper[t.b];
                    float2 c = pointsWithSuper[t.c];

                    if (Orient2D(a, b, c) <= 0f)
                        continue;

                    if (InCircleCCW(a, b, c, p))
                        bad.Add(ti);
                }

                if (bad.Count == 0)
                    continue;

                boundary.Clear();
                for (int bi = 0; bi < bad.Count; bi++) {
                    var t = outTriangles[bad[bi]];
                    ToggleEdge(boundary, t.a, t.b);
                    ToggleEdge(boundary, t.b, t.c);
                    ToggleEdge(boundary, t.c, t.a);
                }

                bad.Sort();
                for (int bi = bad.Count - 1; bi >= 0; bi--)
                    outTriangles.RemoveAt(bad[bi]);

                foreach (var e in boundary) {
                    int a = e.a;
                    int b = e.b;

                    if (Orient2D(pointsWithSuper[a], pointsWithSuper[b], p) < 0f) {
                        int tmp = a; a = b; b = tmp;
                    }

                    outTriangles.Add(new DTBuilder.Triangle(a, b, pi));
                }
            }
        }
    }
}
