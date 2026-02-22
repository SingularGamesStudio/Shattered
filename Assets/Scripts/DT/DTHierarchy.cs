using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using System.Diagnostics;

namespace GPU.Delaunay {
    public sealed class DTHierarchy : IDisposable {
        readonly ComputeShader shader;
        DT[] levels;
        int[] levelEndIndex;
        int maxLevel;
        float2[] gpuAllScratch;

        readonly List<DTBuilder.Triangle> triangles = new List<DTBuilder.Triangle>(4096);

        public DTHierarchy(ComputeShader shader) => this.shader = shader ? shader : throw new ArgumentNullException(nameof(shader));

        public DT GetLevelDt(int level) => (levels != null && (uint)level < (uint)levels.Length) ? levels[level] : null;

        public void InitFromMeshlessNodes(
            List<Node> nodes,
            int[] precomputedLevelEndIndex,
            int precomputedMaxLevel,
            float2 normCenter,
            float normInvHalfExtent,
            float2 super0,
            float2 super1,
            float2 super2,
            int neighborCount) {
            if (nodes == null) throw new ArgumentNullException(nameof(nodes));
            if (nodes.Count < 3) throw new ArgumentException("Need at least 3 nodes.");
            if (neighborCount <= 0) throw new ArgumentOutOfRangeException(nameof(neighborCount));
            this.levelEndIndex = precomputedLevelEndIndex;
            this.maxLevel = precomputedMaxLevel;

            DisposeLevels();
            levels = new DT[maxLevel + 1];

            float worldAreaScale = 1f / (normInvHalfExtent * normInvHalfExtent);

            for (int level = 0; level <= maxLevel; level++) {
                int n = levelEndIndex[level];
                if (n < 3) {
                    levels[level] = null;
                    continue;
                }

                EnsureScratch(n + 3);

                // Normalize points
                for (int i = 0; i < n; i++)
                    gpuAllScratch[i] = (nodes[i].pos - normCenter) * normInvHalfExtent;

                gpuAllScratch[n + 0] = (super0 - normCenter) * normInvHalfExtent;
                gpuAllScratch[n + 1] = (super1 - normCenter) * normInvHalfExtent;
                gpuAllScratch[n + 2] = (super2 - normCenter) * normInvHalfExtent;

                // Build triangulation using the unified builder
                triangles.Clear();
                DTBuilder.BuildDelaunay(gpuAllScratch, n, triangles);

                if (level == 0) {
                    for (int i = 0; i < n; i++)
                        nodes[i].restVolume = 0f;

                    foreach (var tri in triangles) {
                        if ((uint)tri.a >= (uint)n || (uint)tri.b >= (uint)n || (uint)tri.c >= (uint)n)
                            continue;
                        float2 a = gpuAllScratch[tri.a];
                        float2 b = gpuAllScratch[tri.b];
                        float2 c = gpuAllScratch[tri.c];
                        float areaNorm = 0.5f * math.abs((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
                        float areaWorld = areaNorm * worldAreaScale;
                        float share = areaWorld / 3f;
                        nodes[tri.a].restVolume += share;
                        nodes[tri.b].restVolume += share;
                        nodes[tri.c].restVolume += share;
                    }
                }

                // Build half‑edges and create the GPU DT

                var he = DTBuilder.BuildHalfEdges(gpuAllScratch, triangles);
                var dt = new DT(shader);
                dt.Init(gpuAllScratch, n, he, triangles.Count, neighborCount);
                levels[level] = dt;
            }
        }

        void EnsureScratch(int required) {
            if (gpuAllScratch == null || gpuAllScratch.Length != required)
                gpuAllScratch = new float2[required];
        }

        void DisposeLevels() {
            if (levels == null) return;
            foreach (var dt in levels)
                dt?.Dispose();
        }

        public void Dispose() {
            DisposeLevels();
            levels = null;
            levelEndIndex = null;
            gpuAllScratch = null;
        }
    }
}