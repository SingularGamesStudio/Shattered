using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

namespace Physics {

    public class NodeCache {
        public List<int> neighbors;
        public float2x2 L;
        public float2x2 F0 = float2x2.identity;
        public float lambda = 0;
        public float currentVolume;
    }

    public class NodeBatch {
        public List<Node> nodes;
        public List<NodeCache> caches;
        public List<DebugData> debug;
        public int Count;

        private int lastInitializedCount = 0;

        public void Initialise() {
            // Volumes: compute only for newly added nodes (indices >= lastInitializedCount)
            CacheVolumes();
            // Neighbors: must recompute for ALL nodes since new nodes can be neighbors
            CacheNeighbors();
            // Correction matrices: depend on neighbors, so recompute for ALL
            ComputeCorrectionMatrices();
            // Debug data: reset for all active nodes
            ResetDebugData();
            // F0 cache: needs initialization only for newly added nodes
            for (int i = 0; i < Count; i++) {
                var cache = caches[i];
                cache.F0 = nodes[i].F;
                cache.lambda = 0;
            }
        }

        public NodeBatch(List<Node> nodes, int maxCount) {
            this.nodes = nodes;
            Count = 0;
            caches = new List<NodeCache>(maxCount);
            debug = new List<DebugData>(maxCount);
            for (int i = 0; i < maxCount; i++) {
                caches.Add(new NodeCache());
                debug.Add(new DebugData());
            }
        }

        public void ExpandTo(int newCount) {
            if (newCount > caches.Count) {
                int toAdd = newCount - caches.Count;
                for (int i = 0; i < toAdd; i++) {
                    caches.Add(new NodeCache());
                    debug.Add(new DebugData());
                }
            }
            Count = newCount;
        }

        public void ResetDebugData() {
            for (int i = 0; i < Count; i++) {
                debug[i].Reset();
            }
        }

        public void FinalizeDebugData() {
            for (int i = 0; i < Count; i++) {
                debug[i].FinalizeAverages();
            }
        }

        public float CalculateKineticEnergy() {
            float energy = 0f;
            for (int i = 0; i < Count; i++) {
                if (nodes[i].invMass > 0f) {
                    energy += 0.5f * (1f / nodes[i].invMass) * math.lengthsq(nodes[i].vel);
                }
            }
            return energy;
        }

        public void CacheVolumes() {
            for (int i = lastInitializedCount; i < Count; i++) {
                Node node = nodes[i];
                float det = node.F.c0.x * node.F.c1.y - node.F.c0.y * node.F.c1.x;
                caches[i].currentVolume = node.restVolume * math.abs(det);
            }
        }

        public void CacheNeighbors() {
            for (int i = 0; i < Count; i++) {
                caches[i].neighbors = nodes[i].parent.hnsw.SearchKnn(nodes[i].pos, Const.NeighborCount + 1);

                if (caches[i].neighbors.Contains(i)) {
                    caches[i].neighbors.Remove(i);
                } else if (caches[i].neighbors.Count > Const.NeighborCount) {
                    caches[i].neighbors.RemoveAt(Const.NeighborCount);
                }

                caches[i].neighbors.Sort((a, b) => {
                    float2 va = nodes[a].pos - nodes[i].pos;
                    float2 vb = nodes[b].pos - nodes[i].pos;
                    return math.atan2(va.y, va.x).CompareTo(math.atan2(vb.y, vb.x));
                });
            }
        }

        public void ComputeCorrectionMatrices() {
            for (int i = 0; i < Count; i++) {
                var node = nodes[i];
                var cache = caches[i];
                if (cache?.neighbors == null) continue;

                float2x2 sum = float2x2.zero;

                for (int k = 0; k < cache.neighbors.Count; k++) {
                    int j = cache.neighbors[k];
                    if ((uint)j >= (uint)Count) continue;

                    float2 xij = nodes[j].pos - node.pos;
                    float r2 = math.lengthsq(xij);
                    if (r2 < Const.Eps * Const.Eps) continue;

                    float w = 1.0f / r2;

                    sum.c0 += w * xij * xij.x;
                    sum.c1 += w * xij * xij.y;
                }

                cache.L = DeformationUtils.PseudoInverse(sum);
            }
        }

    }
}
