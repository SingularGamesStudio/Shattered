using System;
using System.Collections.Generic;
using Unity.Mathematics;

namespace GPU.Delaunay {
    public static partial class DTBuilder {
        static void TriangulateFast(float2[] points, int count, List<Triangle> outTriangles) {
            outTriangles.Clear();
            if (count < 3) return;

            var coords = new double[count * 2];
            for (int i = 0; i < count; i++) {
                coords[2 * i] = points[i].x;
                coords[2 * i + 1] = points[i].y;
            }

            const double EPSILON = 2.2204460492503131e-16;

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

            int i0 = 0;
            double minDist = double.PositiveInfinity;
            for (int i = 0; i < count; i++) {
                double dx = coords[2 * i] - cx;
                double dy = coords[2 * i + 1] - cy;
                double d = dx * dx + dy * dy;
                if (d < minDist) { i0 = i; minDist = d; }
            }

            double i0x = coords[2 * i0], i0y = coords[2 * i0 + 1];

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

            int i2 = 0;
            double minRadius = double.PositiveInfinity;
            for (int i = 0; i < count; i++) {
                if (i == i0 || i == i1) continue;
                double r = Circumradius(i0x, i0y, i1x, i1y, coords[2 * i], coords[2 * i + 1]);
                if (r < minRadius) { i2 = i; minRadius = r; }
            }
            if (double.IsInfinity(minRadius)) return;

            double i2x = coords[2 * i2], i2y = coords[2 * i2 + 1];

            if (Orient(i0x, i0y, i1x, i1y, i2x, i2y)) {
                int tmp = i1; i1 = i2; i2 = tmp;
                i1x = coords[2 * i1]; i1y = coords[2 * i1 + 1];
                i2x = coords[2 * i2]; i2y = coords[2 * i2 + 1];
            }

            Circumcenter(i0x, i0y, i1x, i1y, i2x, i2y, out double ccx, out double ccy);

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

                if (k > 0 && math.abs(x - xp) <= EPSILON && math.abs(y - yp) <= EPSILON) continue;
                xp = x; yp = y;

                if (i == i0 || i == i1 || i == i2) continue;

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

                int t = AddTriangle(e, i, hullNext[e], -1, -1, hullTri[e]);

                hullTri[i] = Legalize(t + 2);
                hullTri[e] = t;
                hullSize++;

                int next = hullNext[e];
                q = hullNext[next];
                while (Orient(x, y, coords[2 * next], coords[2 * next + 1], coords[2 * q], coords[2 * q + 1])) {
                    t = AddTriangle(next, i, q, hullTri[i], -1, hullTri[next]);
                    hullTri[i] = Legalize(t + 2);
                    hullNext[next] = next;
                    hullSize--;
                    next = q;
                    q = hullNext[next];
                }

                if (e == e0) {
                    q = hullPrev[e];
                    while (Orient(x, y, coords[2 * q], coords[2 * q + 1], coords[2 * e], coords[2 * e + 1])) {
                        t = AddTriangle(q, i, e, -1, hullTri[e], hullTri[q]);
                        Legalize(t + 2);
                        hullTri[q] = t;
                        hullNext[e] = e;
                        hullSize--;
                        e = q;
                        q = hullPrev[e];
                    }
                }

                hullStart = hullPrev[i] = e;
                hullNext[e] = hullPrev[next] = i;
                hullNext[i] = next;

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
