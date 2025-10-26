using System.Collections.Generic;
using Unity.VisualScripting;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Animations;
namespace Physics {
    public interface Constraint {
        void Initialise(NodeBatch nodes) { }
        void Relax(NodeBatch nodes, float stiffness, float timeStep) { }
        string GetConstraintType() { return GetType().Name; }
    }

    public class ConstraintCache {
        public Lambdas lambdas = new Lambdas { };
        public List<int> neighbors;
        public List<float> neighborDistances;
        public List<float> leafVolumes;
        public float avgEdgeLen;
        public Dictionary<string, ConstraintDebugData> debugDataPerConstraint = new Dictionary<string, ConstraintDebugData>();
    }

    public class Lambdas {
        public float tension = 0;
        public List<float> neighborDistance;
        public List<float> volume;
    }

    public class ConstraintDebugData {
        public string constraintType;
        public float maxPositionDelta = 0f;
        public float avgPositionDelta = 0f;
        public float sumPositionDelta = 0f;
        public int positionUpdateCount = 0;

        public int degenerateCount = 0;
        public int nanInfCount = 0;
        public int skippedIterations = 0;

        public float preConstraintEnergy = 0f;
        public float postConstraintEnergy = 0f;

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
            skippedIterations = 0;
            preConstraintEnergy = 0f;
            postConstraintEnergy = 0f;
        }
    }

    public class NodeBatch {
        private const int NeighborsCached = 0b1;
        private const int AvgRestLenCached = 0b10;
        private const int VolumeCached = 0b100;
        const int neighborCount = 6;
        public List<Node> nodes;
        public List<ConstraintCache> caches;
        public int Count;
        private int cachedCategories = 0;

        public NodeBatch(List<Node> nodes) {
            this.nodes = nodes;
            Count = nodes.Count;
            caches = new List<ConstraintCache>(Count + 1);
            for (int i = 0; i < Count; i++) {
                caches.Add(new ConstraintCache { });
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
                    energy += 0.5f * (1f / nodes[i].invMass) * math.lengthsq(nodes[i].predPos - nodes[i].pos);
                }
            }
            return energy;
        }

        public void CacheNeighbors() {
            if ((cachedCategories & NeighborsCached) > 0) {
                return;
            }
            cachedCategories += NeighborsCached;

            for (int i = 0; i < nodes.Count; i++) {
                caches[i].neighbors = nodes[i].parent.hnsw.SearchKnn(nodes[i].pos, neighborCount + 1);
                if (caches[i].neighbors.Contains(i)) {
                    caches[i].neighbors.Remove(i);
                } else {
                    caches[i].neighbors.RemoveAt(neighborCount);
                }
                caches[i].neighbors.Sort((a, b) => Cross(nodes[a].pos - nodes[i].pos, nodes[b].pos - nodes[i].pos).CompareTo(0));

                caches[i].neighborDistances = new List<float>(caches[i].neighbors.Count + 1);

                for (int j = 0; j < caches[i].neighbors.Count; j++) {
                    caches[i].neighborDistances.Add(math.distance(nodes[i].pos, nodes[caches[i].neighbors[j]].pos));
                }
            }
        }

        public void CacheAvgRestLength() {
            if ((cachedCategories & AvgRestLenCached) > 0) {
                return;
            }
            cachedCategories += AvgRestLenCached;
            CacheNeighbors();

            for (int i = 0; i < nodes.Count; i++) {
                caches[i].avgEdgeLen = 0f;
                int cnt = 0;
                foreach (float f in caches[i].neighborDistances) {
                    if (f < 1e-6f) { continue; }
                    caches[i].avgEdgeLen += f;
                    cnt++;
                }
                caches[i].avgEdgeLen /= cnt;
            }
        }

        public void CacheVolume() {
            if ((cachedCategories & VolumeCached) > 0) {
                return;
            }
            cachedCategories += VolumeCached;
            CacheNeighbors();

            for (int i = 0; i < nodes.Count; i++) {
                caches[i].leafVolumes = new List<float>(caches[i].neighborDistances.Count + 1);
                for (int k = 0; k < caches[i].neighbors.Count; k++) {
                    int j0 = caches[i].neighbors[k];
                    int j1 = caches[i].neighbors[(k + 1) % caches[i].neighbors.Count];
                    float2 a = nodes[j0].pos - nodes[i].pos;
                    float2 b = nodes[j1].pos - nodes[i].pos;
                    caches[i].leafVolumes.Add(0.5f * Cross(a, b));
                }
            }
        }

        public void CacheLambdas() {
            for (int i = 0; i < nodes.Count; i++) {
                caches[i].lambdas.tension = 0f;
                caches[i].lambdas.neighborDistance = new List<float>(caches[i].neighborDistances.Count + 1);
                caches[i].lambdas.volume = new List<float>(caches[i].neighborDistances.Count + 1);
                for (int j = 0; j < caches[i].neighborDistances.Count; j++) {
                    caches[i].lambdas.neighborDistance.Add(0f);
                    caches[i].lambdas.volume.Add(0);
                }
            }
        }

        static float Cross(in float2 a, in float2 b) => a.x * b.y - a.y * b.x;
    }
}