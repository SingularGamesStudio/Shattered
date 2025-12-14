using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

namespace Physics {

    public class NodeCache {
        public List<int> neighbors;
        // Kernel gradient correction matrix for velocity gradient estimation (XPBI Eq. 10)
        public float2x2 L;
        public float2x2 F0 = float2x2.identity;
        public float lambda = 0;
    }

    public class NodeBatch {
        public List<Node> nodes;
        public List<NodeCache> caches;
        public List<DebugData> debug;
        public int Count;

        // Initialization state flags
        private bool neighborsInitialized = false;
        private bool correctionMatricesInitialized = false;

        public NodeBatch(List<Node> nodes) {
            this.nodes = nodes;
            Count = nodes.Count;
            caches = new List<NodeCache>(Count);
            debug = new List<DebugData>(Count);
            for (int i = 0; i < Count; i++) {
                caches.Add(new NodeCache());
                debug.Add(new DebugData());
            }
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

        public void CacheNeighbors() {
            if (neighborsInitialized) return;

            for (int i = 0; i < nodes.Count; i++) {
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

            neighborsInitialized = true;
        }

        public void ComputeCorrectionMatrices() {
            CacheNeighbors();

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

            correctionMatricesInitialized = true;
        }

    }
}