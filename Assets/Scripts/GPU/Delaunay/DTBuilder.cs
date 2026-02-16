using System;
using System.Collections.Generic;
using Unity.Mathematics;

namespace GPU.Delaunay {
    public static class DTBuilder {
        public struct Triangle {
            public int a, b, c; // CCW preferred
            public Triangle(int a, int b, int c) { this.a = a; this.b = b; this.c = c; }
        }

        public struct HalfEdge {
            public int v;
            public int next;
            public int twin;
            public int t;
        }

        struct DirectedEdge : IEquatable<DirectedEdge> {
            public readonly int a;
            public readonly int b;

            public DirectedEdge(int a, int b) { this.a = a; this.b = b; }

            public bool Equals(DirectedEdge other) => a == other.a && b == other.b;
            public override bool Equals(object obj) => obj is DirectedEdge other && Equals(other);
            public override int GetHashCode() => unchecked((a * 73856093) ^ (b * 19349663));
        }

        struct EdgeKey : IEquatable<EdgeKey> {
            public readonly int a;
            public readonly int b;

            public EdgeKey(int a, int b) { this.a = a; this.b = b; }

            public bool Equals(EdgeKey other) => a == other.a && b == other.b;
            public override bool Equals(object obj) => obj is EdgeKey other && Equals(other);
            public override int GetHashCode() => unchecked((a * 73856093) ^ (b * 19349663));
        }

        static float Orient2D(in float2 a, in float2 b, in float2 c) {
            float2 ab = b - a;
            float2 ac = c - a;
            return ab.x * ac.y - ab.y * ac.x;
        }

        static bool InCircleCCW(in float2 a, in float2 b, in float2 c, in float2 p) {
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

        static void ToggleEdge(HashSet<DirectedEdge> set, int a, int b) {
            if (!set.Remove(new DirectedEdge(b, a)))
                set.Add(new DirectedEdge(a, b));
        }

        /// <summary>
        /// Bowyer-Watson triangulation with a persistent super triangle.
        /// Unlike typical implementations, this keeps triangles incident to the super vertices,
        /// which is required for dynamic maintenance: hull edges become interior (paired) and thus flippable.
        ///
        /// superBoundsMin/superBoundsMax must cover the entire expected motion domain of real points.
        /// </summary>
        public static void BuildBowyerWatsonWithSuper(
            IReadOnlyList<float2> points,
            float2 superBoundsMin,
            float2 superBoundsMax,
            float superScale,
            out float2[] allPoints,
            out List<Triangle> triangles,
            out int realPointCount) {
            if (points == null) throw new ArgumentNullException(nameof(points));

            int n = points.Count;
            realPointCount = n;

            if (n < 3) {
                allPoints = n == 0 ? Array.Empty<float2>() : new float2[n];
                for (int i = 0; i < n; i++) allPoints[i] = points[i];
                triangles = new List<Triangle>(0);
                return;
            }

            ComputeSuperTriangle(superBoundsMin, superBoundsMax, superScale, out float2 p0, out float2 p1, out float2 p2);

            allPoints = new float2[n + 3];
            for (int i = 0; i < n; i++) allPoints[i] = points[i];
            allPoints[n + 0] = p0;
            allPoints[n + 1] = p1;
            allPoints[n + 2] = p2;

            triangles = new List<Triangle>(math.max(16, 2 * n));
            if (Orient2D(allPoints[n + 0], allPoints[n + 1], allPoints[n + 2]) < 0f)
                triangles.Add(new Triangle(n + 0, n + 2, n + 1));
            else
                triangles.Add(new Triangle(n + 0, n + 1, n + 2));

            var bad = new List<int>(64);
            var boundary = new HashSet<DirectedEdge>(256);

            for (int pi = 0; pi < n; pi++) {
                float2 p = allPoints[pi];

                bad.Clear();
                for (int ti = 0; ti < triangles.Count; ti++) {
                    var t = triangles[ti];

                    float2 a = allPoints[t.a];
                    float2 b = allPoints[t.b];
                    float2 c = allPoints[t.c];

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

                    if (Orient2D(allPoints[a], allPoints[b], p) < 0f) {
                        int tmp = a; a = b; b = tmp;
                    }

                    triangles.Add(new Triangle(a, b, pi));
                }
            }
        }

        /// <summary>
        /// Builds a half-edge structure for a triangle list.
        /// Each triangle contributes 3 half-edges with 'next' cycling around the face and 'twin' linking opposite directed edges.
        /// </summary>
        public static HalfEdge[] BuildHalfEdges(IReadOnlyList<float2> points, IReadOnlyList<Triangle> triangles) {
            if (points == null) throw new ArgumentNullException(nameof(points));
            if (triangles == null) throw new ArgumentNullException(nameof(triangles));

            int triCount = triangles.Count;
            var halfEdges = new HalfEdge[triCount * 3];
            var map = new Dictionary<EdgeKey, int>(triCount * 3);

            for (int t = 0; t < triCount; t++) {
                var tri = triangles[t];
                int a = tri.a;
                int b = tri.b;
                int c = tri.c;

                if (Orient2D(points[a], points[b], points[c]) < 0f) {
                    int tmp = b; b = c; c = tmp;
                }

                int he0 = t * 3 + 0;
                int he1 = t * 3 + 1;
                int he2 = t * 3 + 2;

                halfEdges[he0] = new HalfEdge { v = a, next = he1, twin = -1, t = t };
                halfEdges[he1] = new HalfEdge { v = b, next = he2, twin = -1, t = t };
                halfEdges[he2] = new HalfEdge { v = c, next = he0, twin = -1, t = t };

                AddTwin(map, halfEdges, he0, a, b);
                AddTwin(map, halfEdges, he1, b, c);
                AddTwin(map, halfEdges, he2, c, a);
            }

            return halfEdges;
        }

        /// <summary>
        /// Constructs a triangle that fully encloses the given AABB, scaled by <paramref name="scale"/>.
        /// This should be large enough to contain all point motion for the entire simulation run (or until rebuild).
        /// </summary>
        static void ComputeSuperTriangle(float2 min, float2 max, float scale, out float2 p0, out float2 p1, out float2 p2) {
            float2 center = 0.5f * (min + max);
            float2 extent = max - min;

            float d = math.max(extent.x, extent.y);
            float s = math.max(1f, scale) * math.max(1e-6f, d);

            p0 = center + new float2(0f, 2f * s);
            p1 = center + new float2(-2f * s, -2f * s);
            p2 = center + new float2(2f * s, -2f * s);
        }

        static void AddTwin(Dictionary<EdgeKey, int> map, HalfEdge[] halfEdges, int he, int u, int v) {
            var rev = new EdgeKey(v, u);
            if (map.TryGetValue(rev, out int other)) {
                halfEdges[he].twin = other;
                halfEdges[other].twin = he;
                map.Remove(rev);
                return;
            }

            map.Add(new EdgeKey(u, v), he);
        }
    }
}
