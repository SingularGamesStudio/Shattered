using System;
using System.Collections.Generic;
using Unity.Mathematics;

namespace GPU.Delaunay {
    public static partial class DTBuilder {
        public struct Triangle { public int a, b, c; public Triangle(int a, int b, int c) { this.a = a; this.b = b; this.c = c; } }
        public struct HalfEdge { public int v, next, twin, t; }

        struct EdgeKey : IEquatable<EdgeKey> {
            public readonly int a, b;
            public EdgeKey(int a, int b) { this.a = a; this.b = b; }
            public bool Equals(EdgeKey other) => a == other.a && b == other.b;
            public override int GetHashCode() => unchecked((a * 73856093) ^ (b * 19349663));
        }

        static float Orient2D(float2 a, float2 b, float2 c) {
            float2 ab = b - a, ac = c - a;
            return ab.x * ac.y - ab.y * ac.x;
        }

        /// <summary>
        /// Builds a Delaunay triangulation. The last three entries of pointsWithSuper are the super‑triangle vertices.
        /// </summary>
        public static void BuildDelaunay(float2[] pointsWithSuper, int realCount, List<Triangle> outTriangles) {
            outTriangles.Clear();
            int total = realCount + 3;
            if (pointsWithSuper == null || pointsWithSuper.Length < total || total < 3)
                return;

            TriangulateFast(pointsWithSuper, total, outTriangles);
        }

        /// <summary> Builds a Delaunay triangulation with a super triangle computed from the given bounds. </summary>
        public static void BuildBowyerWatsonWithSuper(
            IReadOnlyList<float2> points,
            float2 superBoundsMin,
            float2 superBoundsMax,
            float superScale,
            out float2[] allPoints,
            out List<Triangle> triangles,
            out int realPointCount) {

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
            allPoints[n] = p0;
            allPoints[n + 1] = p1;
            allPoints[n + 2] = p2;

            triangles = new List<Triangle>(math.max(16, 2 * n));
            TriangulateFast(allPoints, n + 3, triangles);
        }

        static void ComputeSuperTriangle(float2 min, float2 max, float scale, out float2 p0, out float2 p1, out float2 p2) {
            float2 center = 0.5f * (min + max);
            float d = math.max(max.x - min.x, max.y - min.y);
            float s = math.max(1f, scale) * math.max(1e-6f, d);
            p0 = center + new float2(0f, 2f * s);
            p1 = center + new float2(-2f * s, -2f * s);
            p2 = center + new float2(2f * s, -2f * s);
        }

        /// <summary> Builds a half‑edge representation from a triangle list. </summary>
        public static HalfEdge[] BuildHalfEdges(IReadOnlyList<float2> points, IReadOnlyList<Triangle> triangles) {
            int triCount = triangles.Count;
            var halfEdges = new HalfEdge[triCount * 3];
            var map = new Dictionary<EdgeKey, int>(triCount * 3);

            for (int t = 0; t < triCount; t++) {
                var tri = triangles[t];
                int a = tri.a, b = tri.b, c = tri.c;

                // ensure CCW order (same as your old code path)
                if (Orient2D(points[a], points[b], points[c]) < 0f) { int tmp = b; b = c; c = tmp; }

                int he0 = t * 3, he1 = he0 + 1, he2 = he0 + 2;
                halfEdges[he0] = new HalfEdge { v = a, next = he1, twin = -1, t = t };
                halfEdges[he1] = new HalfEdge { v = b, next = he2, twin = -1, t = t };
                halfEdges[he2] = new HalfEdge { v = c, next = he0, twin = -1, t = t };

                AddTwin(map, halfEdges, he0, a, b);
                AddTwin(map, halfEdges, he1, b, c);
                AddTwin(map, halfEdges, he2, c, a);
            }

            return halfEdges;
        }

        static void AddTwin(Dictionary<EdgeKey, int> map, HalfEdge[] halfEdges, int he, int u, int v) {
            var rev = new EdgeKey(v, u);
            if (map.TryGetValue(rev, out int other)) {
                halfEdges[he].twin = other;
                halfEdges[other].twin = he;
                map.Remove(rev);
            } else {
                map.Add(new EdgeKey(u, v), he);
            }
        }

    }
}
