using System;
using System.Collections.Generic;
using Unity.Mathematics;

namespace Physics {

    public class NodeCache {
        public readonly List<int> neighbors;

        // XPBI Eq. (10): L_p = ( Σ_b V_b^n ∇W_b(x_p) ⊗ (x_b - x_p) )^{-1}
        // Used to correct raw kernel gradients on irregular / sparse neighborhoods (Bonet–Lok style).
        public float2x2 L;

        public float2x2 F0 = float2x2.identity;

        public float lambda = 0;
        public float currentVolume;

        // Per-node smoothing length h. Wendland C2 support radius is 2h.
        public float kernelH = 0f;

        public readonly float2[] gradC_vj;

        public readonly float[] neighborDistances;
        public readonly float[] neighborAngles;

        public NodeCache() {
            neighbors = new List<int>(Const.NeighborCount + 1);
            gradC_vj = new float2[Const.NeighborCount];
            neighborDistances = new float[Const.NeighborCount];
            neighborAngles = new float[Const.NeighborCount];
        }
    }

    public class NodeBatch {
        public List<Node> nodes;
        public List<NodeCache> caches;
        public List<DebugData> debug;
        public int Count;

        private int lastInitializedCount = 0;
        private bool fullRefreshPending = true;

        // DSU-style owner cache for hierarchical volume aggregation.
        private int[] ownerCache;
        private int[] ownerUpdateLevel;
        private int curLevel = 1;

        public NodeBatch(List<Node> nodes, int maxCount) {
            this.nodes = nodes;
            Count = 0;

            caches = new List<NodeCache>(maxCount);
            debug = new List<DebugData>(maxCount);
            for (int i = 0; i < maxCount; i++) {
                caches.Add(new NodeCache());
                debug.Add(new DebugData());
            }

            EnsureOwnerCacheCapacity(maxCount);
        }

        public void BeginStep() {
            lastInitializedCount = 0;
            fullRefreshPending = true;
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

        public void Initialise(int level) {
            long t = LoopProfiler.Stamp();
            CacheVolumes();
            LoopProfiler.Add(LoopProfiler.Section.BatchCacheVolumes, t);

            t = LoopProfiler.Stamp();
            CacheNeighbors(level);//TODO: 10-13% of tick time, not GPU-friendly
            LoopProfiler.Add(LoopProfiler.Section.BatchCacheNeighbors, t);

            t = LoopProfiler.Stamp();
            CacheKernelRadii();
            LoopProfiler.Add(LoopProfiler.Section.BatchCacheKernelRadii, t);

            t = LoopProfiler.Stamp();
            ComputeCorrectionMatrices();//TODO: 2% of tick time
            LoopProfiler.Add(LoopProfiler.Section.BatchComputeCorrectionMatrices, t);

            t = LoopProfiler.Stamp();
            ResetDebugData();
            LoopProfiler.Add(LoopProfiler.Section.BatchResetDebugData, t);

            for (int i = 0; i < Count; i++) {
                caches[i].lambda = 0;
            }

            t = LoopProfiler.Stamp();
            CacheF0();
            LoopProfiler.Add(LoopProfiler.Section.BatchCacheF0, t);

            lastInitializedCount = Count;
            fullRefreshPending = false;
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

        void EnsureOwnerCacheCapacity(int n) {
            if (ownerCache == null || ownerCache.Length < n) ownerCache = new int[n];
            if (ownerUpdateLevel == null || ownerUpdateLevel.Length < n) ownerUpdateLevel = new int[n];
        }

        int FindOwnerWithCompression(int idx, int activeCount, int totalCount) {
            if ((uint)idx < (uint)activeCount) return idx;

            int x = idx;
            int res;
            while (true) {
                if ((uint)x < (uint)activeCount) { res = x; break; }

                if (ownerUpdateLevel[x] == curLevel) { res = ownerCache[x]; break; }

                int p = nodes[x].parentIndex;
                if (p < 0 || p == x || (uint)p >= (uint)totalCount) { res = -1; break; }

                x = p;
            }

            x = idx;
            while ((uint)x >= (uint)activeCount && (uint)x < (uint)totalCount && ownerUpdateLevel[x] != curLevel) {
                ownerCache[x] = res;
                ownerUpdateLevel[x] = curLevel;

                int p = nodes[x].parentIndex;
                if (p < 0 || p == x || (uint)p >= (uint)totalCount) break;

                x = p;
            }

            return res;
        }

        public void CacheVolumes() {
            for (int i = 0; i < Count; i++) {
                caches[i].currentVolume = 0f;
            }

            int total = nodes.Count;
            EnsureOwnerCacheCapacity(total);

            curLevel++;
            if (curLevel == int.MaxValue) {
                Array.Clear(ownerUpdateLevel, 0, ownerUpdateLevel.Length);
                curLevel = 1;
            }

            for (int i = 0; i < total; i++) {
                Node node = nodes[i];
                if (node.restVolume <= Const.Eps) continue;

                float det = node.F.c0.x * node.F.c1.y - node.F.c0.y * node.F.c1.x;
                float leafVol = node.restVolume * math.abs(det);
                if (leafVol <= Const.Eps) continue;

                int owner = FindOwnerWithCompression(i, Count, total);
                if ((uint)owner < (uint)Count) {
                    caches[owner].currentVolume += leafVol;
                }
            }
        }

        public void CacheF0() {
            int start = fullRefreshPending ? 0 : lastInitializedCount;
            if (start >= Count) return;

            for (int i = start; i < Count; i++) {
                caches[i].F0 = nodes[i].F;
            }
        }

        public void CacheNeighbors(int level) {
            for (int i = 0; i < Count; i++) {
                var cache = caches[i];
                var neighbors = cache.neighbors;

                nodes[i].parent.GetNeighborsForLevel(level, i, neighbors);

                int self = neighbors.IndexOf(i);
                if (self >= 0) neighbors.RemoveAt(self);

                if (neighbors.Count > Const.NeighborCount)
                    neighbors.RemoveAt(Const.NeighborCount);

                int nCount = neighbors.Count;
                for (int k = 0; k < nCount; k++) {
                    float2 v = nodes[neighbors[k]].pos - nodes[i].pos;
                    cache.neighborAngles[k] = math.atan2(v.y, v.x);
                }

                for (int k = 1; k < nCount; k++) {
                    float ak = cache.neighborAngles[k];
                    int nk = neighbors[k];

                    int j = k - 1;
                    while (j >= 0 && cache.neighborAngles[j] > ak) {
                        cache.neighborAngles[j + 1] = cache.neighborAngles[j];
                        neighbors[j + 1] = neighbors[j];
                        j--;
                    }

                    cache.neighborAngles[j + 1] = ak;
                    neighbors[j + 1] = nk;
                }
            }
        }

        public void CacheKernelRadii() {
            for (int i = 0; i < Count; i++) {
                var cache = caches[i];
                var N = cache.neighbors;
                if (N == null || N.Count == 0) {
                    cache.kernelH = 0f;
                    continue;
                }

                int n = 0;
                for (int k = 0; k < N.Count && n < Const.NeighborCount; k++) {
                    int j = N[k];
                    if ((uint)j >= (uint)Count) continue;

                    float2 xij = nodes[j].pos - nodes[i].pos;
                    float r = math.length(xij);
                    if (r <= Const.Eps) continue;

                    cache.neighborDistances[n++] = r;
                }

                if (n == 0) {
                    cache.kernelH = 0f;
                    continue;
                }

                Array.Sort(cache.neighborDistances, 0, n);

                float median = (n & 1) == 1
                    ? cache.neighborDistances[n >> 1]
                    : 0.5f * (cache.neighborDistances[(n >> 1) - 1] + cache.neighborDistances[n >> 1]);

                cache.kernelH = median * Const.KernelHScale;
            }
        }

        public void ComputeCorrectionMatrices() {
            for (int i = 0; i < Count; i++) {
                var node = nodes[i];
                var cache = caches[i];
                var N = cache.neighbors;
                if (N == null || N.Count == 0) {
                    cache.L = float2x2.zero;
                    continue;
                }

                float h = cache.kernelH;
                if (h <= Const.Eps) {
                    cache.L = float2x2.zero;
                    continue;
                }

                float2x2 sum = float2x2.zero;

                long t = LoopProfiler.Stamp();
                for (int k = 0; k < N.Count; k++) {
                    int j = N[k];
                    if ((uint)j >= (uint)Count) continue;

                    float2 xij = nodes[j].pos - node.pos;

                    float2 gradW = SPHKernels.GradWendlandC2(xij, h);
                    if (math.lengthsq(gradW) <= Const.Eps * Const.Eps) continue;

                    float Vb = caches[j].currentVolume;
                    if (Vb <= Const.Eps) continue;

                    sum.c0 += (Vb * xij.x) * gradW;
                    sum.c1 += (Vb * xij.y) * gradW;
                }
                LoopProfiler.Add(LoopProfiler.Section.BatchCorrectionMatrixSum, t);

                t = LoopProfiler.Stamp();
                cache.L = DeformationUtils.PseudoInverse(sum);
                LoopProfiler.Add(LoopProfiler.Section.BatchCorrectionMatrixPseudoInverse, t);
            }
        }
    }
}
