using System;
using System.Collections.Generic;
using Unity.Mathematics;

namespace GPU.Delaunay {
    public static class DTBuilder {
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

        // ----------------------------
        // Fast incremental triangulator
        // ----------------------------

        // Sign conventions here are intentionally matched: Orient(...) and InCircle(...) must agree.
        // This is the same convention used in known-good ports. [page:2]

        static bool Orient(double px, double py, double qx, double qy, double rx, double ry) =>
            (qy - py) * (rx - qx) - (qx - px) * (ry - qy) < 0.0;

        static bool InCircle(double ax, double ay, double bx, double by, double cx, double cy, double px, double py) {
            double dx = ax - px;
            double dy = ay - py;
            double ex = bx - px;
            double ey = by - py;
            double fx = cx - px;
            double fy = cy - py;

            double ap = dx * dx + dy * dy;
            double bp = ex * ex + ey * ey;
            double cp = fx * fx + fy * fy;

            return dx * (ey * cp - bp * fy) -
                   dy * (ex * cp - bp * fx) +
                   ap * (ex * fy - ey * fx) < 0.0;
        }

        static double Circumradius(double ax, double ay, double bx, double by, double cx, double cy) {
            double dx = bx - ax;
            double dy = by - ay;
            double ex = cx - ax;
            double ey = cy - ay;
            double bl = dx * dx + dy * dy;
            double cl = ex * ex + ey * ey;
            double d = 0.5 / (dx * ey - dy * ex);
            double x = (ey * bl - dy * cl) * d;
            double y = (dx * cl - ex * bl) * d;
            return x * x + y * y;
        }

        static void Circumcenter(double ax, double ay, double bx, double by, double cx, double cy, out double x, out double y) {
            double dx = bx - ax;
            double dy = by - ay;
            double ex = cx - ax;
            double ey = cy - ay;
            double bl = dx * dx + dy * dy;
            double cl = ex * ex + ey * ey;
            double d = 0.5 / (dx * ey - dy * ex);
            x = ax + (ey * bl - dy * cl) * d;
            y = ay + (dx * cl - ex * bl) * d;
        }

        static double PseudoAngle(double dx, double dy) {
            double p = dx / (math.abs(dx) + math.abs(dy));
            return (dy > 0 ? 3 - p : 1 + p) / 4;
        }

        static void TriangulateFast(float2[] points, int count, List<Triangle> outTriangles) {
            outTriangles.Clear();
            if (count < 3) return;

            // coords (double)
            var coords = new double[count * 2];
            for (int i = 0; i < count; i++) {
                coords[2 * i] = points[i].x;
                coords[2 * i + 1] = points[i].y;
            }

            const double EPSILON = 2.2204460492503131e-16; // 2^-52, same idea as common ports [page:2]

            // bbox + ids
            var ids = new int[count];
            double minX = double.PositiveInfinity, minY = double.PositiveInfinity;
            double maxX = double.NegativeInfinity, maxY = double.NegativeInfinity;

            for (int i = 0; i < count; i++) {
                double x = coords[2 * i], y = coords[2 * i + 1];
                if (x < minX) minX = x;
                if (y < minY) minY = y;
                if (x > maxX) maxX = x;
                if (y > maxY) maxY = y;
                ids[i] = i;
            }

            double cx = (minX + maxX) * 0.5;
            double cy = (minY + maxY) * 0.5;

            // pick a seed point close to the center
            int i0 = 0;
            double minDist = double.PositiveInfinity;
            for (int i = 0; i < count; i++) {
                double dx = coords[2 * i] - cx;
                double dy = coords[2 * i + 1] - cy;
                double d = dx * dx + dy * dy;
                if (d < minDist) { i0 = i; minDist = d; }
            }

            double i0x = coords[2 * i0], i0y = coords[2 * i0 + 1];

            // find the point closest to the seed
            int i1 = 0;
            minDist = double.PositiveInfinity;
            for (int i = 0; i < count; i++) {
                if (i == i0) continue;
                double dx = coords[2 * i] - i0x;
                double dy = coords[2 * i + 1] - i0y;
                double d = dx * dx + dy * dy;
                if (d < minDist && d > 0) { i1 = i; minDist = d; }
            }

            double i1x = coords[2 * i1], i1y = coords[2 * i1 + 1];

            // find the third point which forms the smallest circumcircle with the first two
            int i2 = 0;
            double minRadius = double.PositiveInfinity;
            for (int i = 0; i < count; i++) {
                if (i == i0 || i == i1) continue;
                double r = Circumradius(i0x, i0y, i1x, i1y, coords[2 * i], coords[2 * i + 1]);
                if (r < minRadius) { i2 = i; minRadius = r; }
            }
            if (double.IsInfinity(minRadius)) return;

            double i2x = coords[2 * i2], i2y = coords[2 * i2 + 1];

            // ensure seed triangle is CCW under this Orient convention
            if (Orient(i0x, i0y, i1x, i1y, i2x, i2y)) {
                int tmp = i1; i1 = i2; i2 = tmp;
                i1x = coords[2 * i1]; i1y = coords[2 * i1 + 1];
                i2x = coords[2 * i2]; i2y = coords[2 * i2 + 1];
            }

            Circumcenter(i0x, i0y, i1x, i1y, i2x, i2y, out double ccx, out double ccy);

            // sort points by distance from seed triangle circumcenter (deterministic tie-break by index)
            var dists = new double[count];
            for (int i = 0; i < count; i++) {
                double dx = coords[2 * i] - ccx;
                double dy = coords[2 * i + 1] - ccy;
                dists[i] = dx * dx + dy * dy;
            }
            Array.Sort(ids, (a, b) => {
                int c = dists[a].CompareTo(dists[b]);
                return c != 0 ? c : a.CompareTo(b);
            });

            int maxTriangles = 2 * count - 5;
            var triangles = new int[maxTriangles * 3];
            var halfedges = new int[maxTriangles * 3];
            for (int i = 0; i < halfedges.Length; i++) halfedges[i] = -1;

            int trianglesLen = 0;

            int hashSize = math.max(1, (int)math.ceil(math.sqrt(count)));
            var hullPrev = new int[count];
            var hullNext = new int[count];
            var hullTri = new int[count];
            var hullHash = new int[hashSize];
            for (int i = 0; i < hashSize; i++) hullHash[i] = -1;

            int hullStart = i0;
            int hullSize = 3;

            hullNext[i0] = hullPrev[i2] = i1;
            hullNext[i1] = hullPrev[i0] = i2;
            hullNext[i2] = hullPrev[i1] = i0;

            hullTri[i0] = 0;
            hullTri[i1] = 1;
            hullTri[i2] = 2;

            int HashKey(double x, double y) =>
                (int)(math.floor(PseudoAngle(x - ccx, y - ccy) * hashSize) % hashSize);

            void Link(int a, int b) {
                halfedges[a] = b;
                if (b != -1) halfedges[b] = a;
            }

            int AddTriangle(int a, int b, int c, int ha, int hb, int hc) {
                int t = trianglesLen;
                triangles[t] = a;
                triangles[t + 1] = b;
                triangles[t + 2] = c;
                Link(t, ha);
                Link(t + 1, hb);
                Link(t + 2, hc);
                trianglesLen += 3;
                return t;
            }

            var edgeStack = new int[512];

            int Legalize(int a) {
                int i = 0;
                int ar;

                while (true) {
                    int b = halfedges[a];

                    int a0 = a - a % 3;
                    ar = a0 + (a + 2) % 3;

                    if (b == -1) {
                        if (i == 0) break;
                        a = edgeStack[--i];
                        continue;
                    }

                    int b0 = b - b % 3;
                    int al = a0 + (a + 1) % 3;
                    int bl = b0 + (b + 2) % 3;

                    int p0 = triangles[ar];
                    int pr = triangles[a];
                    int pl = triangles[al];
                    int p1 = triangles[bl];

                    bool illegal = InCircle(
                        coords[2 * p0], coords[2 * p0 + 1],
                        coords[2 * pr], coords[2 * pr + 1],
                        coords[2 * pl], coords[2 * pl + 1],
                        coords[2 * p1], coords[2 * p1 + 1]);

                    if (illegal) {
                        triangles[a] = p1;
                        triangles[b] = p0;

                        int hbl = halfedges[bl];

                        // edge swapped on the other side of the hull (rare); fix the halfedge reference
                        if (hbl == -1) {
                            int e = hullStart;
                            do {
                                if (hullTri[e] == bl) { hullTri[e] = a; break; }
                                e = hullPrev[e];
                            } while (e != hullStart);
                        }

                        Link(a, hbl);
                        Link(b, halfedges[ar]);
                        Link(ar, bl);

                        int br = b0 + (b + 1) % 3;
                        if (i < edgeStack.Length) edgeStack[i++] = br;
                    } else {
                        if (i == 0) break;
                        a = edgeStack[--i];
                    }
                }

                return ar;
            }

            hullHash[HashKey(i0x, i0y)] = i0;
            hullHash[HashKey(i1x, i1y)] = i1;
            hullHash[HashKey(i2x, i2y)] = i2;

            AddTriangle(i0, i1, i2, -1, -1, -1);

            double xp = 0, yp = 0;

            for (int k = 0; k < ids.Length; k++) {
                int i = ids[k];
                double x = coords[2 * i];
                double y = coords[2 * i + 1];

                // skip near-duplicate points (same policy as common ports)
                if (k > 0 && math.abs(x - xp) <= EPSILON && math.abs(y - yp) <= EPSILON) continue;
                xp = x; yp = y;

                // skip seed triangle points
                if (i == i0 || i == i1 || i == i2) continue;

                // find a visible edge on the convex hull using edge hash
                int start = 0;
                int key = HashKey(x, y);
                for (int j = 0; j < hashSize; j++) {
                    start = hullHash[(key + j) % hashSize];
                    if (start != -1 && start != hullNext[start]) break;
                }

                start = hullPrev[start];
                int e0 = start;
                int e = start;
                int q = hullNext[e];

                while (!Orient(x, y, coords[2 * e], coords[2 * e + 1], coords[2 * q], coords[2 * q + 1])) {
                    e = q;
                    if (e == e0) { e = int.MaxValue; break; }
                    q = hullNext[e];
                }

                if (e == int.MaxValue) continue;

                // add the first triangle from the point
                int t = AddTriangle(e, i, hullNext[e], -1, -1, hullTri[e]);

                // flip triangles until Delaunay
                hullTri[i] = Legalize(t + 2);
                hullTri[e] = t;
                hullSize++;

                // walk forward
                int next = hullNext[e];
                q = hullNext[next];
                while (Orient(x, y, coords[2 * next], coords[2 * next + 1], coords[2 * q], coords[2 * q + 1])) {
                    t = AddTriangle(next, i, q, hullTri[i], -1, hullTri[next]);
                    hullTri[i] = Legalize(t + 2);
                    hullNext[next] = next; // mark as removed
                    hullSize--;
                    next = q;
                    q = hullNext[next];
                }

                // walk backward
                if (e == e0) {
                    q = hullPrev[e];
                    while (Orient(x, y, coords[2 * q], coords[2 * q + 1], coords[2 * e], coords[2 * e + 1])) {
                        t = AddTriangle(q, i, e, -1, hullTri[e], hullTri[q]);
                        Legalize(t + 2);
                        hullTri[q] = t;
                        hullNext[e] = e; // mark as removed
                        hullSize--;
                        e = q;
                        q = hullPrev[e];
                    }
                }

                // update hull indices
                hullStart = hullPrev[i] = e;
                hullNext[e] = hullPrev[next] = i;
                hullNext[i] = next;

                // save new edges in hash
                hullHash[HashKey(x, y)] = i;
                hullHash[HashKey(coords[2 * e], coords[2 * e + 1])] = e;
            }

            int triCount = trianglesLen / 3;
            outTriangles.Capacity = math.max(outTriangles.Capacity, triCount);
            for (int t = 0; t < triCount; t++) {
                int e = 3 * t;
                outTriangles.Add(new Triangle(triangles[e], triangles[e + 1], triangles[e + 2]));
            }
        }
    }
}
