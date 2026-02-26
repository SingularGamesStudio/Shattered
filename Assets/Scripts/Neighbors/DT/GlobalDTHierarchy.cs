using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

namespace GPU.Delaunay {
    public sealed class GlobalDTHierarchy : IDisposable {
        struct LayerData {
            public DT dt;
            public int[] ownerBodyByLocal;
            public int[] globalNodeByLocal;
            public int[] globalFineNodeByLocal;
            public int activeCount;
            public int fineCount;
            public float layerKernelH;
        }

        readonly ComputeShader shader;
        LayerData[] layers;
        int maxLayer = -1;

        float2 normCenter;
        float normInvHalfExtent;

        readonly List<DTBuilder.Triangle> triangles = new List<DTBuilder.Triangle>(4096);

        public GlobalDTHierarchy(ComputeShader shader) {
            this.shader = shader ? shader : throw new ArgumentNullException(nameof(shader));
        }

        public int MaxLayer => maxLayer;
        public float2 NormCenter => normCenter;
        public float NormInvHalfExtent => normInvHalfExtent;

        public bool TryGetLayerDt(int layer, out DT dt) {
            dt = null;
            if (layers == null || layer < 0 || layer >= layers.Length)
                return false;

            dt = layers[layer].dt;
            return dt != null;
        }

        public bool TryGetLayerMappings(int layer, out int[] ownerBodyByLocal, out int[] globalNodeByLocal, out int[] globalFineNodeByLocal, out int activeCount, out int fineCount) {
            ownerBodyByLocal = null;
            globalNodeByLocal = null;
            globalFineNodeByLocal = null;
            activeCount = 0;
            fineCount = 0;

            if (layers == null || layer < 0 || layer >= layers.Length)
                return false;

            ownerBodyByLocal = layers[layer].ownerBodyByLocal;
            globalNodeByLocal = layers[layer].globalNodeByLocal;
            globalFineNodeByLocal = layers[layer].globalFineNodeByLocal;
            activeCount = layers[layer].activeCount;
            fineCount = layers[layer].fineCount;
            return activeCount > 0;
        }

        public bool TryGetLayerExecutionContext(int layer, out int activeCount, out int fineCount, out float layerKernelH) {
            activeCount = 0;
            fineCount = 0;
            layerKernelH = 0f;

            if (layers == null || layer < 0 || layer >= layers.Length)
                return false;

            LayerData data = layers[layer];
            activeCount = data.activeCount;
            fineCount = data.fineCount;
            layerKernelH = data.layerKernelH;
            return activeCount > 0;
        }

        public bool TryGetBodyLayerSegment(int layer, int bodyIndex, out int localBase, out int activeCount) {
            localBase = 0;
            activeCount = 0;

            if (layers == null || layer < 0 || layer >= layers.Length)
                return false;

            int[] owners = layers[layer].ownerBodyByLocal;
            if (owners == null || owners.Length == 0)
                return false;

            int first = -1;
            int last = -1;
            for (int i = 0; i < owners.Length; i++) {
                if (owners[i] != bodyIndex)
                    continue;

                if (first < 0)
                    first = i;
                last = i;
            }

            if (first < 0 || last < first)
                return false;

            localBase = first;
            activeCount = last - first + 1;
            return activeCount > 0;
        }

        public void Rebuild(IReadOnlyList<Meshless> meshes, IReadOnlyList<int> baseOffsets, bool allowCrossBodyTopology, int maxLayerOverride = -1) {
            if (meshes == null) throw new ArgumentNullException(nameof(meshes));
            if (baseOffsets == null) throw new ArgumentNullException(nameof(baseOffsets));
            if (meshes.Count != baseOffsets.Count) throw new ArgumentException("meshes/baseOffsets count mismatch.");

            int localMaxLayer = -1;
            for (int i = 0; i < meshes.Count; i++) {
                var m = meshes[i];
                if (m == null || m.nodes == null || m.nodes.Count <= 0)
                    continue;

                localMaxLayer = math.max(localMaxLayer, m.maxLayer);
            }

            if (maxLayerOverride >= 0)
                localMaxLayer = math.min(localMaxLayer, maxLayerOverride);

            DisposeLayers();
            maxLayer = localMaxLayer;
            if (maxLayer < 0) {
                layers = null;
                return;
            }

            layers = new LayerData[maxLayer + 1];

            ComputeGlobalNormalization(meshes);
            float2 super0;
            float2 super1;
            float2 super2;
            ComputeSuperTriangle(meshes, 2f, out super0, out super1, out super2);

            for (int layer = 0; layer <= maxLayer; layer++) {
                var points = new List<float2>(1024);
                var owners = new List<int>(1024);
                var globals = new List<int>(1024);
                var fineGlobals = new List<int>(1024);
                float layerKernelH = 0f;
                bool hasLayerKernelH = false;

                if (layer == 0) {
                    int totalNodes = CountAllNodes(meshes);
                    float2[] pointsByGlobal = new float2[math.max(1, totalNodes)];
                    int[] ownerByGlobal = new int[math.max(1, totalNodes)];
                    bool[] filledByGlobal = new bool[math.max(1, totalNodes)];

                    for (int meshIdx = 0; meshIdx < meshes.Count; meshIdx++) {
                        Meshless m = meshes[meshIdx];
                        if (m == null || m.nodes == null || m.nodes.Count <= 0)
                            continue;

                        int baseOffset = baseOffsets[meshIdx];
                        float meshLayerKernelH = m.GetLayerKernelH(layer);
                        if (!hasLayerKernelH) {
                            layerKernelH = meshLayerKernelH;
                            hasLayerKernelH = true;
                        } else {
                            layerKernelH = math.max(layerKernelH, meshLayerKernelH);
                        }

                        for (int i = 0; i < m.nodes.Count; i++) {
                            int gi = baseOffset + i;
                            if ((uint)gi >= (uint)totalNodes)
                                continue;

                            pointsByGlobal[gi] = (m.nodes[i].pos - normCenter) * normInvHalfExtent;
                            ownerByGlobal[gi] = meshIdx;
                            filledByGlobal[gi] = true;
                        }
                    }

                    for (int gi = 0; gi < totalNodes; gi++) {
                        if (!filledByGlobal[gi])
                            continue;

                        points.Add(pointsByGlobal[gi]);
                        owners.Add(ownerByGlobal[gi]);
                        globals.Add(gi);
                        fineGlobals.Add(gi);
                    }
                } else {
                    var fineTailGlobals = new List<int>(1024);

                    for (int meshIdx = 0; meshIdx < meshes.Count; meshIdx++) {
                        Meshless m = meshes[meshIdx];
                        if (m == null || m.nodes == null || m.nodes.Count <= 0)
                            continue;
                        if (layer > m.maxLayer)
                            continue;

                        int active = m.NodeCount(layer);
                        int fine = layer == 1 ? m.nodes.Count : (layer > 1 ? m.NodeCount(layer - 1) : active);
                        int baseOffset = baseOffsets[meshIdx];
                        float meshLayerKernelH = m.GetLayerKernelH(layer);
                        if (!hasLayerKernelH) {
                            layerKernelH = meshLayerKernelH;
                            hasLayerKernelH = true;
                        } else {
                            layerKernelH = math.max(layerKernelH, meshLayerKernelH);
                        }

                        for (int i = 0; i < active; i++) {
                            float2 p = (m.nodes[i].pos - normCenter) * normInvHalfExtent;
                            points.Add(p);
                            owners.Add(meshIdx);
                            globals.Add(baseOffset + i);
                            fineGlobals.Add(baseOffset + i);
                        }

                        for (int i = active; i < fine; i++) {
                            fineTailGlobals.Add(baseOffset + i);
                        }
                    }

                    for (int i = 0; i < fineTailGlobals.Count; i++)
                        fineGlobals.Add(fineTailGlobals[i]);
                }

                int activeCount = points.Count;
                int fineCount = fineGlobals.Count;

                if (activeCount < 3) {
                    layers[layer] = new LayerData {
                        dt = null,
                        ownerBodyByLocal = owners.ToArray(),
                        globalNodeByLocal = globals.ToArray(),
                        globalFineNodeByLocal = fineGlobals.ToArray(),
                        activeCount = activeCount,
                        fineCount = fineCount,
                        layerKernelH = hasLayerKernelH ? layerKernelH : 0f,
                    };
                    continue;
                }

                List<float2> dtPoints;
                float2 super0Norm = (super0 - normCenter) * normInvHalfExtent;
                float2 super1Norm = (super1 - normCenter) * normInvHalfExtent;
                float2 super2Norm = (super2 - normCenter) * normInvHalfExtent;
                if (allowCrossBodyTopology || !HasMultipleOwners(owners, activeCount)) {
                    points.Add(super0Norm);
                    points.Add(super1Norm);
                    points.Add(super2Norm);

                    triangles.Clear();
                    DTBuilder.BuildDelaunay(points.ToArray(), activeCount, triangles);
                    dtPoints = points;
                } else {
                    dtPoints = new List<float2>(activeCount + math.max(3, meshes.Count * 3));
                    for (int i = 0; i < activeCount; i++)
                        dtPoints.Add(points[i]);

                    triangles.Clear();
                    int segmentStart = 0;
                    while (segmentStart < activeCount) {
                        int segmentEnd = segmentStart + 1;
                        while (segmentEnd < activeCount && owners[segmentEnd] == owners[segmentStart])
                            segmentEnd++;

                        int segmentCount = segmentEnd - segmentStart;
                        if (segmentCount >= 3) {
                            float2 localMin = points[segmentStart];
                            float2 localMax = localMin;
                            for (int i = segmentStart + 1; i < segmentEnd; i++) {
                                float2 p = points[i];
                                localMin = math.min(localMin, p);
                                localMax = math.max(localMax, p);
                            }

                            ComputeSuperTriangleForRange(localMin, localMax, 2f, out float2 ls0, out float2 ls1, out float2 ls2);

                            var segmentPoints = new float2[segmentCount + 3];
                            for (int i = 0; i < segmentCount; i++)
                                segmentPoints[i] = points[segmentStart + i];
                            segmentPoints[segmentCount] = ls0;
                            segmentPoints[segmentCount + 1] = ls1;
                            segmentPoints[segmentCount + 2] = ls2;

                            var segmentTriangles = new List<DTBuilder.Triangle>(math.max(4, segmentCount * 2));
                            DTBuilder.BuildDelaunay(segmentPoints, segmentCount, segmentTriangles);

                            int superBase = dtPoints.Count;
                            dtPoints.Add(ls0);
                            dtPoints.Add(ls1);
                            dtPoints.Add(ls2);

                            for (int ti = 0; ti < segmentTriangles.Count; ti++) {
                                DTBuilder.Triangle t = segmentTriangles[ti];
                                triangles.Add(new DTBuilder.Triangle(
                                    MapSegmentVertexIndex(t.a, segmentStart, segmentCount, superBase),
                                    MapSegmentVertexIndex(t.b, segmentStart, segmentCount, superBase),
                                    MapSegmentVertexIndex(t.c, segmentStart, segmentCount, superBase)
                                ));
                            }
                        }

                        segmentStart = segmentEnd;
                    }
                }

                var he = DTBuilder.BuildHalfEdges(dtPoints, triangles);
                var dt = new DT(shader);
                dt.Init(dtPoints, activeCount, he, triangles.Count, Const.NeighborCount);

                if (!allowCrossBodyTopology && HasMultipleOwners(owners, activeCount))
                    FilterNeighborsByOwner(dt, owners, activeCount);

                layers[layer] = new LayerData {
                    dt = dt,
                    ownerBodyByLocal = owners.ToArray(),
                    globalNodeByLocal = globals.ToArray(),
                    globalFineNodeByLocal = fineGlobals.ToArray(),
                    activeCount = activeCount,
                    fineCount = fineCount,
                    layerKernelH = hasLayerKernelH ? layerKernelH : 0f,
                };
            }
        }

        static int CountAllNodes(IReadOnlyList<Meshless> meshes) {
            int total = 0;
            for (int i = 0; i < meshes.Count; i++) {
                Meshless m = meshes[i];
                if (m == null || m.nodes == null)
                    continue;
                total += m.nodes.Count;
            }

            return total;
        }

        void ComputeGlobalNormalization(IReadOnlyList<Meshless> meshes) {
            bool any = false;
            float2 min = 0f;
            float2 max = 0f;

            for (int i = 0; i < meshes.Count; i++) {
                Meshless m = meshes[i];
                if (m == null || m.nodes == null || m.nodes.Count <= 0)
                    continue;

                for (int j = 0; j < m.nodes.Count; j++) {
                    float2 p = m.nodes[j].pos;
                    if (!any) {
                        min = p;
                        max = p;
                        any = true;
                    } else {
                        min = math.min(min, p);
                        max = math.max(max, p);
                    }
                }
            }

            if (!any) {
                normCenter = 0f;
                normInvHalfExtent = 1f;
                return;
            }

            float2 padding = new float2(2f, 2f);
            min -= padding;
            max += padding;

            normCenter = 0.5f * (min + max);
            float2 extent = max - min;
            float half = 0.5f * math.max(extent.x, extent.y);
            normInvHalfExtent = 1f / math.max(1e-6f, half);
        }

        static void ComputeSuperTriangle(IReadOnlyList<Meshless> meshes, float scale, out float2 p0, out float2 p1, out float2 p2) {
            bool any = false;
            float2 min = 0f;
            float2 max = 0f;

            for (int i = 0; i < meshes.Count; i++) {
                Meshless m = meshes[i];
                if (m == null || m.nodes == null || m.nodes.Count <= 0)
                    continue;

                for (int j = 0; j < m.nodes.Count; j++) {
                    float2 p = m.nodes[j].pos;
                    if (!any) {
                        min = p;
                        max = p;
                        any = true;
                    } else {
                        min = math.min(min, p);
                        max = math.max(max, p);
                    }
                }
            }

            if (!any) {
                p0 = new float2(0f, 2f);
                p1 = new float2(-2f, -2f);
                p2 = new float2(2f, -2f);
                return;
            }

            float2 center = 0.5f * (min + max);
            float2 extent = max - min;
            float d = math.max(extent.x, extent.y);
            float s = math.max(1f, scale) * math.max(1e-6f, d);
            p0 = center + new float2(0f, 2f * s);
            p1 = center + new float2(-2f * s, -2f * s);
            p2 = center + new float2(2f * s, -2f * s);
        }

        static void ComputeSuperTriangleForRange(float2 min, float2 max, float scale, out float2 p0, out float2 p1, out float2 p2) {
            float2 center = 0.5f * (min + max);
            float2 extent = max - min;
            float d = math.max(extent.x, extent.y);
            float s = math.max(1f, scale) * math.max(1e-6f, d);
            p0 = center + new float2(0f, 2f * s);
            p1 = center + new float2(-2f * s, -2f * s);
            p2 = center + new float2(2f * s, -2f * s);
        }

        static int MapSegmentVertexIndex(int localIndex, int segmentStart, int segmentCount, int superBase) {
            if (localIndex < segmentCount)
                return segmentStart + localIndex;

            return superBase + (localIndex - segmentCount);
        }

        static void FilterNeighborsByOwner(DT dt, List<int> owners, int activeCount) {
            if (dt == null || owners == null || activeCount <= 0)
                return;

            int neighborCount = dt.NeighborCount;
            if (neighborCount <= 0)
                return;

            int[] neighbors = new int[activeCount * neighborCount];
            int[] counts = new int[activeCount];

            dt.NeighborsBuffer.GetData(neighbors);
            dt.NeighborCountsBuffer.GetData(counts);

            for (int i = 0; i < activeCount; i++) {
                int owner = owners[i];
                int inCount = math.clamp(counts[i], 0, neighborCount);
                int outCount = 0;
                int rowStart = i * neighborCount;

                for (int k = 0; k < inCount; k++) {
                    int n = neighbors[rowStart + k];
                    if (n < 0 || n >= activeCount)
                        continue;
                    if (owners[n] != owner)
                        continue;

                    neighbors[rowStart + outCount] = n;
                    outCount++;
                }

                for (int k = outCount; k < neighborCount; k++)
                    neighbors[rowStart + k] = i;

                counts[i] = outCount;
            }

            dt.NeighborsBuffer.SetData(neighbors);
            dt.NeighborCountsBuffer.SetData(counts);
        }

        static bool HasMultipleOwners(List<int> owners, int activeCount) {
            if (owners == null || activeCount <= 1)
                return false;

            int first = owners[0];
            int count = math.min(activeCount, owners.Count);
            for (int i = 1; i < count; i++) {
                if (owners[i] != first)
                    return true;
            }

            return false;
        }

        void DisposeLayers() {
            if (layers == null)
                return;

            for (int i = 0; i < layers.Length; i++) {
                layers[i].dt?.Dispose();
                layers[i].dt = null;
                layers[i].ownerBodyByLocal = null;
                layers[i].globalNodeByLocal = null;
                layers[i].globalFineNodeByLocal = null;
                layers[i].activeCount = 0;
                layers[i].fineCount = 0;
                layers[i].layerKernelH = 0f;
            }
        }

        public void Dispose() {
            DisposeLayers();
            layers = null;
            maxLayer = -1;
            normCenter = 0f;
            normInvHalfExtent = 1f;
        }
    }
}
