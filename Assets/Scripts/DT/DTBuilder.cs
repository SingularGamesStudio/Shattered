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

        static void ComputeSuperTriangle(float2 min, float2 max, float scale, out float2 p0, out float2 p1, out float2 p2) {
            float2 center = 0.5f * (min + max);
            float d = math.max(max.x - min.x, max.y - min.y);
            float s = math.max(1f, scale) * math.max(1e-6f, d);
            p0 = center + new float2(0f, 2f * s);
            p1 = center + new float2(-2f * s, -2f * s);
            p2 = center + new float2(2f * s, -2f * s);
        }

        /// <summary>
        /// Builds a Delaunay triangulation.
        /// The last three entries of pointsWithSuper are the super‑triangle vertices.
        /// </summary>
        public static void BuildDelaunay(float2[] pointsWithSuper, int realCount, List<Triangle> outTriangles) {
            outTriangles.Clear();
            if (realCount < 3) return;

            TriangulateFast(pointsWithSuper, realCount, outTriangles);
        }

        /// <summary> Builds a half‑edge representation from a triangle list. </summary>
        public static HalfEdge[] BuildHalfEdges(IReadOnlyList<float2> points, IReadOnlyList<Triangle> triangles) {
            int triCount = triangles.Count;
            var halfEdges = new HalfEdge[triCount * 3];
            var map = new Dictionary<EdgeKey, int>(triCount * 3);

            for (int t = 0; t < triCount; t++) {
                var tri = triangles[t];
                int a = tri.a, b = tri.b, c = tri.c;

                // ensure CCW order
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

        static int NextHalfedge(int e) => e % 3 == 2 ? e - 2 : e + 1;

        static double Orient(double ax, double ay, double bx, double by, double cx, double cy) {
            return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
        }

        static bool InCircle(
            double ax, double ay,
            double bx, double by,
            double cx, double cy,
            double px, double py) {

            double apx = ax - px, apy = ay - py;
            double bpx = bx - px, bpy = by - py;
            double cpx = cx - px, cpy = cy - py;

            double a2 = apx * apx + apy * apy;
            double b2 = bpx * bpx + bpy * bpy;
            double c2 = cpx * cpx + cpy * cpy;

            double det =
                apx * (bpy * c2 - b2 * cpy) -
                apy * (bpx * c2 - b2 * cpx) +
                a2 * (bpx * cpy - bpy * cpx);

            return det > 0.0;
        }

        static double PseudoAngle(double dx, double dy) {
            double p = dx / (math.abs(dx) + math.abs(dy));
            return (dy > 0 ? (3 - p) : (1 + p)) / 4;
        }

        static void Circumcenter(
            double ax, double ay,
            double bx, double by,
            double cx, double cy,
            out double x, out double y) {

            double dx = bx - ax, dy = by - ay;
            double ex = cx - ax, ey = cy - ay;

            double bl = dx * dx + dy * dy;
            double cl = ex * ex + ey * ey;

            double d = 2.0 * (dx * ey - dy * ex);
            if (d == 0) { x = ax; y = ay; return; }

            double inv = 1.0 / d;
            x = ax + (ey * bl - dy * cl) * inv;
            y = ay + (dx * cl - ex * bl) * inv;
        }

        static double Circumradius2(
            double ax, double ay,
            double bx, double by,
            double cx, double cy) {

            if (math.abs(Orient(ax, ay, bx, by, cx, cy)) < 1e-18) return double.PositiveInfinity;
            Circumcenter(ax, ay, bx, by, cx, cy, out double x, out double y);
            double dx = ax - x, dy = ay - y;
            return dx * dx + dy * dy;
        }

        static void TriangulateFast(float2[] points, int count, List<Triangle> outTriangles) {
            outTriangles.Clear();
            if (count < 3) return;

            // coords (double for stability)
            var coords = new double[count * 2];
            for (int i = 0; i < count; i++) {
                coords[2 * i] = points[i].x;
                coords[2 * i + 1] = points[i].y;
            }

            // bbox
            double minX = coords[0], minY = coords[1];
            double maxX = minX, maxY = minY;
            for (int i = 1; i < count; i++) {
                double x = coords[2 * i], y = coords[2 * i + 1];
                if (x < minX) minX = x;
                if (y < minY) minY = y;
                if (x > maxX) maxX = x;
                if (y > maxY) maxY = y;
            }
            double cx0 = (minX + maxX) * 0.5;
            double cy0 = (minY + maxY) * 0.5;

            // pick i0 closest to bbox center
            int i0 = 0;
            double minDist = double.PositiveInfinity;
            for (int i = 0; i < count; i++) {
                double dx = coords[2 * i] - cx0;
                double dy = coords[2 * i + 1] - cy0;
                double d = dx * dx + dy * dy;
                if (d < minDist) { i0 = i; minDist = d; }
            }

            // pick i1 closest to i0
            int i1 = -1;
            minDist = double.PositiveInfinity;
            double i0x = coords[2 * i0], i0y = coords[2 * i0 + 1];
            for (int i = 0; i < count; i++) {
                if (i == i0) continue;
                double dx = coords[2 * i] - i0x;
                double dy = coords[2 * i + 1] - i0y;
                double d = dx * dx + dy * dy;
                if (d > 0 && d < minDist) { i1 = i; minDist = d; }
            }
            if (i1 == -1) return;

            // pick i2 with minimum circumradius
            int i2 = -1;
            double minRadius = double.PositiveInfinity;
            double i1x = coords[2 * i1], i1y = coords[2 * i1 + 1];
            for (int i = 0; i < count; i++) {
                if (i == i0 || i == i1) continue;
                double x = coords[2 * i], y = coords[2 * i + 1];
                double r = Circumradius2(i0x, i0y, i1x, i1y, x, y);
                if (r < minRadius) { i2 = i; minRadius = r; }
            }
            if (i2 == -1 || double.IsInfinity(minRadius)) return;

            double i2x = coords[2 * i2], i2y = coords[2 * i2 + 1];

            // ensure seed triangle is CCW
            if (Orient(i0x, i0y, i1x, i1y, i2x, i2y) < 0) {
                int tmp = i1; i1 = i2; i2 = tmp;
                i1x = coords[2 * i1]; i1y = coords[2 * i1 + 1];
                i2x = coords[2 * i2]; i2y = coords[2 * i2 + 1];
            }

            Circumcenter(i0x, i0y, i1x, i1y, i2x, i2y, out double ccx, out double ccy);

            // sort point ids by distance to circumcenter
            var ids = new int[count];
            var dists = new double[count];
            for (int i = 0; i < count; i++) {
                ids[i] = i;
                double dx = coords[2 * i] - ccx;
                double dy = coords[2 * i + 1] - ccy;
                dists[i] = dx * dx + dy * dy;
            }
            Array.Sort(ids, (a, b) => dists[a].CompareTo(dists[b]));

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

            hullPrev[i0] = i2;
            hullNext[i0] = i1;
            hullPrev[i1] = i0;
            hullNext[i1] = i2;
            hullPrev[i2] = i1;
            hullNext[i2] = i0;

            int HashKey(double dx, double dy) {
                int key = (int)math.floor(hashSize * PseudoAngle(dx, dy));
                return key < hashSize ? key : hashSize - 1;
            }

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
                int stackSize = 0;

                while (true) {
                    int b = halfedges[a];

                    int a0 = a - a % 3;
                    int ar = a0 + (a + 2) % 3;

                    if (b == -1) {
                        if (stackSize == 0) return ar;
                        a = edgeStack[--stackSize];
                        continue;
                    }

                    int b0 = b - b % 3;
                    int al = a0 + (a + 1) % 3;
                    int bl = b0 + (b + 2) % 3;

                    int p0 = triangles[ar];
                    int pr = triangles[a];
                    int pl = triangles[al];
                    int p1 = triangles[bl];

                    double p0x = coords[2 * p0], p0y = coords[2 * p0 + 1];
                    double prx = coords[2 * pr], pry = coords[2 * pr + 1];
                    double plx = coords[2 * pl], ply = coords[2 * pl + 1];
                    double p1x = coords[2 * p1], p1y = coords[2 * p1 + 1];

                    if (InCircle(p0x, p0y, prx, pry, plx, ply, p1x, p1y)) {
                        triangles[a] = p1;
                        triangles[b] = p0;

                        int hbl = halfedges[bl];
                        if (hbl == -1) {
                            int e = hullStart;
                            do {
                                if (hullTri[e] == bl) { hullTri[e] = a; break; }
                                e = hullNext[e];
                            } while (e != hullStart);
                        }

                        Link(a, hbl);
                        Link(b, halfedges[ar]);
                        Link(ar, bl);

                        int br = b0 + (b + 1) % 3;
                        if (stackSize == edgeStack.Length) Array.Resize(ref edgeStack, edgeStack.Length * 2);
                        edgeStack[stackSize++] = br;
                        continue;
                    }

                    if (stackSize == 0) return ar;
                    a = edgeStack[--stackSize];
                }
            }

            // seed triangle
            hullTri[i0] = 0;
            hullTri[i1] = 1;
            hullTri[i2] = 2;

            hullHash[HashKey(coords[2 * i0] - ccx, coords[2 * i0 + 1] - ccy)] = i0;
            hullHash[HashKey(coords[2 * i1] - ccx, coords[2 * i1 + 1] - ccy)] = i1;
            hullHash[HashKey(coords[2 * i2] - ccx, coords[2 * i2 + 1] - ccy)] = i2;

            AddTriangle(i0, i1, i2, -1, -1, -1);

            double prevX = double.NaN, prevY = double.NaN;

            for (int k = 0; k < ids.Length; k++) {
                int i = ids[k];
                if (i == i0 || i == i1 || i == i2) continue;

                double x = coords[2 * i], y = coords[2 * i + 1];
                if (x == prevX && y == prevY) continue;
                prevX = x; prevY = y;

                // find a visible edge on the hull
                int start = -1;
                int key = HashKey(x - ccx, y - ccy);
                for (int j = 0; j < hashSize; j++) {
                    int s = hullHash[(key + j) % hashSize];
                    if (s != -1 && s != hullNext[s]) { start = s; break; }
                }
                if (start == -1) start = hullStart;

                int e = start;
                while (true) {
                    int q = hullNext[e];
                    double ex = coords[2 * e], ey = coords[2 * e + 1];
                    double qx = coords[2 * q], qy = coords[2 * q + 1];
                    if (Orient(ex, ey, qx, qy, x, y) < 0) break;

                    e = q;
                    if (e == start) { e = -1; break; }
                }
                if (e == -1) continue;

                int next = hullNext[e];

                // add first triangle from the visible edge
                int t = AddTriangle(e, i, next, -1, -1, hullTri[e]);
                hullTri[i] = Legalize(t + 2);
                hullTri[e] = t;

                // walk forward and add triangles
                int n = next;
                while (true) {
                    int q = hullNext[n];
                    double nx = coords[2 * n], ny = coords[2 * n + 1];
                    double qx = coords[2 * q], qy = coords[2 * q + 1];
                    if (Orient(nx, ny, qx, qy, x, y) >= 0) break;

                    t = AddTriangle(n, i, q, hullTri[i], -1, hullTri[n]);
                    hullTri[i] = Legalize(t + 2);
                    hullNext[n] = n; // mark removed
                    n = q;
                }

                // walk backward and add triangles
                int prev = e;
                while (true) {
                    int q = hullPrev[prev];
                    double qx = coords[2 * q], qy = coords[2 * q + 1];
                    double px = coords[2 * prev], py = coords[2 * prev + 1];
                    if (Orient(qx, qy, px, py, x, y) >= 0) break;

                    t = AddTriangle(q, i, prev, -1, hullTri[prev], hullTri[q]);
                    Legalize(t + 2);
                    hullTri[q] = t;
                    hullNext[prev] = prev; // mark removed
                    prev = q;
                }

                // splice the new point into the hull between prev and n
                hullStart = prev;
                hullPrev[i] = prev;
                hullNext[i] = n;
                hullNext[prev] = i;
                hullPrev[n] = i;

                // update hash table
                hullHash[HashKey(x - ccx, y - ccy)] = i;
                hullHash[HashKey(coords[2 * prev] - ccx, coords[2 * prev + 1] - ccy)] = prev;
                hullHash[HashKey(coords[2 * n] - ccx, coords[2 * n + 1] - ccy)] = n;
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
