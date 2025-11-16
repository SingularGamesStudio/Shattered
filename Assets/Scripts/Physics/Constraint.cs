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
        void PlasticFlow(NodeBatch nodes, float dt) { }
    }

    public class ConstraintCache {
        // Lambdas only for neighbor and volume constraints
        public Lambdas lambdas = new Lambdas { };
        public List<int> neighbors;
        public List<float> neighborDistances;
        public List<float> leafVolumes;
        public float avgEdgeLen;
        public Dictionary<string, ConstraintDebugData> debugDataPerConstraint = new Dictionary<string, ConstraintDebugData>();
    }

    public class Lambdas {
        // REMOVE tension:
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
        const int neighborCount = 6;
        public List<Node> nodes;
        public List<ConstraintCache> caches;
        public int Count;

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
                    energy += 0.5f * (1f / nodes[i].invMass) * math.lengthsq(nodes[i].predPos - nodes[i].pos);
                }
            }
            return energy;
        }

        public void CacheNeighbors() {
            for (int i = 0; i < nodes.Count; i++) {
                caches[i].neighbors = nodes[i].parent.hnsw.SearchKnn(nodes[i].pos, neighborCount + 1);
                if (caches[i].neighbors.Contains(i)) caches[i].neighbors.Remove(i);
                else if (caches[i].neighbors.Count > neighborCount) caches[i].neighbors.RemoveAt(6);
                caches[i].neighbors.Sort((a, b) => Cross(nodes[a].pos - nodes[i].pos, nodes[b].pos - nodes[i].pos).CompareTo(0));
            }
        }

        public void CacheLambdas() {
            for (int i = 0; i < nodes.Count; i++) {
                caches[i].lambdas.neighborDistance = new List<float>(caches[i].neighbors.Count);
                caches[i].lambdas.volume = new List<float>(caches[i].neighbors.Count);
                for (int j = 0; j < caches[i].neighbors.Count; j++) {
                    caches[i].lambdas.neighborDistance.Add(0f);
                    caches[i].lambdas.volume.Add(0f);
                }
            }
        }

        static float Cross(in float2 a, in float2 b) => a.x * b.y - a.y * b.x;
    }
}