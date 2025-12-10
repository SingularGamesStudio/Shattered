using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

namespace Physics {
    public interface Constraint {
        void Initialise(NodeBatch nodes);
        void Relax(NodeBatch nodes, float stiffness, float timeStep);
        void UpdatePlasticDeformation(NodeBatch nodes, float timeStep);
        string GetConstraintType();
    }

    public class ConstraintCache {
        public List<int> neighbors;

        // Kernel gradient correction matrix for velocity gradient estimation (XPBI Eq. 10)
        public float2x2 L;

        public Lambdas lambdas = new Lambdas();
        public Dictionary<string, ConstraintDebugData> debugDataPerConstraint = new Dictionary<string, ConstraintDebugData>();
    }

    public class Lambdas {
        public List<float> neighborDistance;
        public List<float> volume;

        public Lambdas() {
            neighborDistance = new List<float>();
            volume = new List<float>();
        }

        public void Reset() {
            for (int i = 0; i < neighborDistance.Count; i++) {
                neighborDistance[i] = 0f;
            }
            for (int i = 0; i < volume.Count; i++) {
                volume[i] = 0f;
            }
        }
    }

    public class ConstraintDebugData {
        public string constraintType;

        // Position update statistics
        public float maxPositionDelta = 0f;
        public float avgPositionDelta = 0f;
        public float sumPositionDelta = 0f;
        public int positionUpdateCount = 0;

        // Error and issue counters
        public int degenerateCount = 0;
        public int nanInfCount = 0;
        public int plasticFlowCount = 0;

        // Convergence metrics
        public int iterationsToConverge = 0;
        public float constraintEnergy = 0f;

        public void RecordPositionUpdate(float2 delta) {
            float magnitude = math.length(delta);
            maxPositionDelta = math.max(maxPositionDelta, magnitude);
            sumPositionDelta += magnitude;
            positionUpdateCount++;
        }

        public void FinalizeAverages() {
            if (positionUpdateCount > 0) {
                avgPositionDelta = sumPositionDelta / positionUpdateCount;
            }
        }

        public void Reset() {
            maxPositionDelta = 0f;
            avgPositionDelta = 0f;
            sumPositionDelta = 0f;
            positionUpdateCount = 0;
            degenerateCount = 0;
            nanInfCount = 0;
            plasticFlowCount = 0;
            iterationsToConverge = 0;
            constraintEnergy = 0f;
        }
    }

    public class NodeBatch {
        const int neighborCount = 6;

        public List<Node> nodes;
        public List<ConstraintCache> caches;
        public int Count;

        // Initialization state flags
        private bool neighborsInitialized = false;
        private bool correctionMatricesInitialized = false;
        private bool lambdasInitialized = false;

        public NodeBatch(List<Node> nodes) {
            this.nodes = nodes;
            Count = nodes.Count;
            caches = new List<ConstraintCache>(Count);
            for (int i = 0; i < Count; i++) {
                caches.Add(new ConstraintCache());
            }
        }

        public ConstraintDebugData GetOrCreateDebugData(int nodeIndex, string constraintType) {
            if (!caches[nodeIndex].debugDataPerConstraint.ContainsKey(constraintType)) {
                caches[nodeIndex].debugDataPerConstraint[constraintType] = new ConstraintDebugData {
                    constraintType = constraintType
                };
            }
            return caches[nodeIndex].debugDataPerConstraint[constraintType];
        }

        public void ResetDebugData(string constraintType) {
            for (int i = 0; i < Count; i++) {
                if (caches[i].debugDataPerConstraint.ContainsKey(constraintType)) {
                    caches[i].debugDataPerConstraint[constraintType].Reset();
                }
            }
        }

        public void FinalizeDebugData(string constraintType) {
            for (int i = 0; i < Count; i++) {
                if (caches[i].debugDataPerConstraint.ContainsKey(constraintType)) {
                    caches[i].debugDataPerConstraint[constraintType].FinalizeAverages();
                }
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

        // Safe to call multiple times - only runs once
        public void CacheNeighbors() {
            if (neighborsInitialized) return;

            for (int i = 0; i < nodes.Count; i++) {
                // Query k+1 neighbors (includes self)
                caches[i].neighbors = nodes[i].parent.hnsw.SearchKnn(nodes[i].pos, neighborCount + 1);

                // Remove self from neighbors
                if (caches[i].neighbors.Contains(i)) {
                    caches[i].neighbors.Remove(i);
                } else if (caches[i].neighbors.Count > neighborCount) {
                    caches[i].neighbors.RemoveAt(neighborCount);
                }

                // Sort by angle for consistent winding
                caches[i].neighbors.Sort((a, b) => {
                    float2 va = nodes[a].pos - nodes[i].pos;
                    float2 vb = nodes[b].pos - nodes[i].pos;
                    return math.atan2(va.y, va.x).CompareTo(math.atan2(vb.y, vb.x));
                });
            }

            neighborsInitialized = true;
        }
        public void ComputeCorrectionMatrices() {
            if (correctionMatricesInitialized) return;
            CacheNeighbors();

            for (int i = 0; i < Count; i++) {
                var node = nodes[i];
                var cache = caches[i];
                if (cache?.neighbors == null) continue;

                float2x2 sum = float2x2.zero;

                for (int k = 0; k < cache.neighbors.Count; k++) {
                    int j = cache.neighbors[k];
                    if (j < 0 || j >= Count) continue;

                    var neighbor = nodes[j];
                    float2 xij = neighbor.pos - node.pos;
                    float dist = math.length(xij);
                    if (dist < Const.Eps) continue;

                    float w = 1.0f / dist;
                    sum.c0 += w * xij * xij.x;
                    sum.c1 += w * xij * xij.y;
                }

                cache.L = DeformationUtils.PseudoInverse(sum);
            }

            correctionMatricesInitialized = true;
        }

        public void InitializeLambdas() {
            if (lambdasInitialized) return;
            CacheNeighbors();

            for (int i = 0; i < Count; i++) {
                var cache = caches[i];
                if (cache.neighbors == null) continue;

                int neighborCount = cache.neighbors.Count;

                cache.lambdas.neighborDistance = new List<float>(neighborCount);
                cache.lambdas.volume = new List<float>(neighborCount);

                for (int j = 0; j < neighborCount; j++) {
                    cache.lambdas.neighborDistance.Add(0f);
                    cache.lambdas.volume.Add(0f);
                }
            }

            lambdasInitialized = true;
        }

        // Reset initialization flags (useful for testing/reinitialization)
        public void ResetInitializationFlags() {
            neighborsInitialized = false;
            correctionMatricesInitialized = false;
            lambdasInitialized = false;
        }
    }
}
