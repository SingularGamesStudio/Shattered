using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

namespace GPU.Delaunay {
    [RequireComponent(typeof(Meshless))]
    public sealed class DelaunayGpuStressTest : MonoBehaviour {
        [Header("GPU")]
        public ComputeShader delaunayShader;
        public int fixIterationsPerSubstep = 2;
        public int legalizeIterationsPerSubstep = 2;
        public int warmupFixIterations = 64;
        public int warmupLegalizeIterations = 128;

        [Header("Point cloud (world)")]
        public int seed = 1;
        public int pointCount = 256;
        public Vector2 boundsMin = new Vector2(-5f, -5f);
        public Vector2 boundsMax = new Vector2(5f, 5f);
        public float minDistance = 0.15f;
        public bool buildMeshlessHnsw = true;

        [Header("Mover")]
        public int movingIndex = 0;
        public float moveSpeed = 1.0f;
        public float maxMoveStep = 0.05f;
        public bool clampMoverToBounds = true;

        [Header("GPU normalization")]
        [Tooltip("Expands the normalization AABB. Must include all motion, otherwise rebuild or clamp.")]
        public float normalizePadding = 2f;
        [Tooltip("If the moving point ever leaves the normalization AABB, rebuild DT (expensive but robust).")]
        public bool rebuildIfOutsideNormalizeBounds = true;

        [Header("Wireframe (Debug.DrawLine)")]
        public bool drawEveryFrame = true;
        public bool depthTest = false;
        public Color lineColor = new Color(0.1f, 0.8f, 1f, 1f);
        public bool drawSuperEdges = false;
        public Color superEdgeColor = new Color(1f, 0.2f, 0.2f, 1f);
        public int readbackIntervalFrames = 1;

        [Header("Diagnostics")]
        public bool logFlips = true;
        public int logEveryNFrames = 60;

        Meshless meshless;
        DelaunayGpu dt;

        DelaunayGpu.HalfEdge[] halfEdgeCpu;

        float2 normCenter;
        float normInvHalfExtent;

        float2[] worldRealScratch;
        float2[] gpuAllScratch;
        float2[] gpuSuper;

        readonly List<DTBuilder.Triangle> triangles = new List<DTBuilder.Triangle>(4096);
        readonly List<int> bad = new List<int>(128);
        readonly HashSet<DirEdge> boundary = new HashSet<DirEdge>(2048);

        uint frameIndex;
        float2 moverVel;

        void OnEnable() => meshless = GetComponent<Meshless>();

        void Start() {
            if (!delaunayShader) { enabled = false; return; }

            GeneratePointCloudWorld();
            RecomputeNormalization();
            BuildAndUploadDT();

            moverVel = math.normalizesafe(new float2(1f, 0.35f)) * moveSpeed;
        }

        void OnDisable() {
            dt?.Dispose();
            dt = null;
        }

        void GeneratePointCloudWorld() {
            meshless.nodes.Clear();
            meshless.maxLayer = 0;
            meshless.levelEndIndex = null;
            meshless.hnsw = null;

            var rng = new Unity.Mathematics.Random((uint)math.max(1, seed));

            float2 min = new float2(boundsMin.x, boundsMin.y);
            float2 max = new float2(boundsMax.x, boundsMax.y);
            float2 size = max - min;

            float minDist2 = minDistance * minDistance;

            for (int i = 0; i < pointCount; i++) {
                float2 p = min + rng.NextFloat2() * size;

                int tries = 0;
                while (tries++ < 32) {
                    bool ok = true;
                    for (int j = 0; j < meshless.nodes.Count; j++) {
                        float2 d = meshless.nodes[j].pos - p;
                        if (math.dot(d, d) < minDist2) { ok = false; break; }
                    }

                    if (ok) break;
                    p = min + rng.NextFloat2() * size;
                }

                meshless.Add(p);

                var node = meshless.nodes[meshless.nodes.Count - 1];
                node.maxLayer = 0;
                node.parentIndex = -1;
                node.vel = float2.zero;
                node.isFixed = false;
                node.invMass = 1f;
            }

            if (buildMeshlessHnsw) meshless.Build();
            else {
                meshless.maxLayer = 0;
                meshless.levelEndIndex = new int[1] { meshless.nodes.Count };
            }

            int n = meshless.nodes.Count;

            if (worldRealScratch == null || worldRealScratch.Length != n)
                worldRealScratch = new float2[n];

            if (gpuAllScratch == null || gpuAllScratch.Length != n + 3)
                gpuAllScratch = new float2[n + 3];

            gpuSuper ??= new float2[3];
        }

        /// <summary>
        /// Defines the normalization transform used for all GPU predicate evaluation:
        ///   p_gpu = (p_world - center) * invHalfExtent.
        /// The super triangle is defined in normalized GPU coordinates to keep determinants well-conditioned.
        /// </summary>
        void RecomputeNormalization() {
            float2 min = new float2(boundsMin.x - normalizePadding, boundsMin.y - normalizePadding);
            float2 max = new float2(boundsMax.x + normalizePadding, boundsMax.y + normalizePadding);

            normCenter = 0.5f * (min + max);

            float2 extent = max - min;
            float half = 0.5f * math.max(extent.x, extent.y);
            normInvHalfExtent = 1f / math.max(1e-6f, half);

            gpuSuper[0] = new float2(0f, 3f);
            gpuSuper[1] = new float2(-3f, -3f);
            gpuSuper[2] = new float2(3f, -3f);
        }

        float2 WorldToGpu(float2 p) => (p - normCenter) * normInvHalfExtent;

        float2 GpuToWorld(float2 p) => p / normInvHalfExtent + normCenter;

        bool IsInsideNormalizeBounds(float2 pWorld) {
            float2 min = new float2(boundsMin.x - normalizePadding, boundsMin.y - normalizePadding);
            float2 max = new float2(boundsMax.x + normalizePadding, boundsMax.y + normalizePadding);
            return pWorld.x >= min.x && pWorld.y >= min.y && pWorld.x <= max.x && pWorld.y <= max.y;
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

        struct DirEdge : IEquatable<DirEdge> {
            public int a, b;
            public DirEdge(int a, int b) { this.a = a; this.b = b; }
            public bool Equals(DirEdge other) => a == other.a && b == other.b;
            public override bool Equals(object obj) => obj is DirEdge other && Equals(other);
            public override int GetHashCode() => unchecked((a * 73856093) ^ (b * 19349663));
        }

        static void ToggleEdge(HashSet<DirEdge> set, int a, int b) {
            var e = new DirEdge(a, b);
            if (!set.Remove(new DirEdge(b, a)))
                set.Add(e);
        }

        /// <summary>
        /// CPU Bowyer-Watson in normalized coordinates, keeping the super-triangle triangles.
        /// This gives the GPU maintenance passes a "virtual outside face", so hull edges are flippable.
        /// </summary>
        void BuildAndUploadDT() {
            int n = meshless.nodes.Count;
            if (n < 3) { enabled = false; return; }

            for (int i = 0; i < n; i++) {
                float2 p = meshless.nodes[i].pos;
                worldRealScratch[i] = p;
                gpuAllScratch[i] = WorldToGpu(p);
            }

            gpuAllScratch[n + 0] = gpuSuper[0];
            gpuAllScratch[n + 1] = gpuSuper[1];
            gpuAllScratch[n + 2] = gpuSuper[2];

            triangles.Clear();

            if (Orient2D(gpuAllScratch[n + 0], gpuAllScratch[n + 1], gpuAllScratch[n + 2]) < 0f)
                triangles.Add(new DTBuilder.Triangle(n + 0, n + 2, n + 1));
            else
                triangles.Add(new DTBuilder.Triangle(n + 0, n + 1, n + 2));

            for (int pi = 0; pi < n; pi++) {
                float2 p = gpuAllScratch[pi];

                bad.Clear();
                for (int ti = 0; ti < triangles.Count; ti++) {
                    var t = triangles[ti];

                    float2 a = gpuAllScratch[t.a];
                    float2 b = gpuAllScratch[t.b];
                    float2 c = gpuAllScratch[t.c];

                    if (Orient2D(a, b, c) <= 0f)
                        continue;

                    if (InCircleCCW(a, b, c, p))
                        bad.Add(ti);
                }

                if (bad.Count == 0)
                    continue;

                boundary.Clear();
                for (int bi = 0; bi < bad.Count; bi++) {
                    var t = triangles[bad[bi]];
                    ToggleEdge(boundary, t.a, t.b);
                    ToggleEdge(boundary, t.b, t.c);
                    ToggleEdge(boundary, t.c, t.a);
                }

                bad.Sort();
                for (int bi = bad.Count - 1; bi >= 0; bi--)
                    triangles.RemoveAt(bad[bi]);

                foreach (var e in boundary) {
                    int a = e.a;
                    int b = e.b;

                    if (Orient2D(gpuAllScratch[a], gpuAllScratch[b], p) < 0f) {
                        int tmp = a; a = b; b = tmp;
                    }

                    triangles.Add(new DTBuilder.Triangle(a, b, pi));
                }
            }

            var he = DTBuilder.BuildHalfEdges(gpuAllScratch, triangles);

            dt?.Dispose();
            dt = new DelaunayGpu(delaunayShader);

            dt.Init(gpuAllScratch, n, he, triangles.Count, Const.NeighborCount);
            dt.Maintain(warmupFixIterations, warmupLegalizeIterations);

            halfEdgeCpu = new DelaunayGpu.HalfEdge[dt.HalfEdgeCount];
            dt.GetHalfEdges(halfEdgeCpu);
        }

        void Update() {
            if (dt == null) return;

            frameIndex++;

            SubstepMoveAndMaintain();

            if (readbackIntervalFrames > 0 && (frameIndex % (uint)readbackIntervalFrames) == 0) {
                if (halfEdgeCpu == null || halfEdgeCpu.Length != dt.HalfEdgeCount)
                    halfEdgeCpu = new DelaunayGpu.HalfEdge[dt.HalfEdgeCount];

                dt.GetHalfEdges(halfEdgeCpu);
            }

            if (drawEveryFrame && readbackIntervalFrames > 0)
                DrawDebugWireframe();

            if (logFlips && (Time.frameCount % math.max(1, logEveryNFrames)) == 0)
                Debug.Log($"DT flips (last iter): {dt.GetLastFlipCount()}");
        }

        /// <summary>
        /// Moves a single vertex using small substeps (important for local maintenance),
        /// optionally rebuilds if leaving the normalization domain, then runs Fix+Legalize.
        /// </summary>
        void SubstepMoveAndMaintain() {
            float dtFrame = Time.deltaTime;
            if (dtFrame <= 0f) return;

            int n = meshless.nodes.Count;
            if ((uint)movingIndex >= (uint)n) return;

            float2 min = new float2(boundsMin.x, boundsMin.y);
            float2 max = new float2(boundsMax.x, boundsMax.y);

            float2 totalMove = moverVel * dtFrame;
            float len = math.length(totalMove);
            if (len <= 1e-12f) return;

            int steps = math.max(1, (int)math.ceil(len / math.max(1e-6f, maxMoveStep)));
            float2 stepMove = totalMove / steps;

            for (int s = 0; s < steps; s++) {
                var node = meshless.nodes[movingIndex];
                float2 p = node.pos + stepMove;

                if (clampMoverToBounds) {
                    p = math.clamp(p, min, max);
                } else {
                    if (p.x < min.x) { p.x = min.x; moverVel.x = math.abs(moverVel.x); } else if (p.x > max.x) { p.x = max.x; moverVel.x = -math.abs(moverVel.x); }

                    if (p.y < min.y) { p.y = min.y; moverVel.y = math.abs(moverVel.y); } else if (p.y > max.y) { p.y = max.y; moverVel.y = -math.abs(moverVel.y); }
                }

                node.pos = p;
                node.vel = float2.zero;

                if (rebuildIfOutsideNormalizeBounds && !IsInsideNormalizeBounds(p)) {
                    RecomputeNormalization();
                    BuildAndUploadDT();
                    return;
                }

                dt.UpdatePositionsFromNodes(meshless.nodes, normCenter, normInvHalfExtent);
                dt.Maintain(fixIterationsPerSubstep, legalizeIterationsPerSubstep);
            }
        }

        static int Dest(DelaunayGpu.HalfEdge[] he, int idx) => he[he[idx].next].v;

        float2 GetDrawPosWorld(int v) {
            int real = meshless.nodes.Count;
            return (uint)v < (uint)real ? meshless.nodes[v].pos : GpuToWorld(gpuSuper[v - real]);
        }

        void DrawDebugWireframe() {
            if (halfEdgeCpu == null || halfEdgeCpu.Length == 0) return;

            int real = meshless.nodes.Count;
            if (real != dt.RealVertexCount) return;

            for (int he0 = 0; he0 < halfEdgeCpu.Length; he0++) {
                int tw = halfEdgeCpu[he0].twin;
                if (tw >= 0 && he0 > tw) continue;

                int v0 = halfEdgeCpu[he0].v;
                int v1 = Dest(halfEdgeCpu, he0);

                bool v0Real = (uint)v0 < (uint)real;
                bool v1Real = (uint)v1 < (uint)real;

                if (!drawSuperEdges) {
                    if (!v0Real || !v1Real) continue;
                } else {
                    if (!v0Real && !v1Real) continue;
                }

                float2 a = GetDrawPosWorld(v0);
                float2 b = GetDrawPosWorld(v1);

                Debug.DrawLine(
                    new Vector3(a.x, a.y, 0f),
                    new Vector3(b.x, b.y, 0f),
                    (v0Real && v1Real) ? lineColor : superEdgeColor,
                    0f,
                    depthTest
                );
            }
        }
    }
}
