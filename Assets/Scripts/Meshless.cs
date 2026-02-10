using System.Collections.Generic;
using System.Linq;
using Unity.Mathematics;
using UnityEngine;
using Physics;

public class Meshless : MonoBehaviour {
    public HNSW hnsw;
    public List<Node> nodes = new List<Node>();
    [HideInInspector]
    public int maxLayer = -1;

    [Header("Simulation parameters")]
    public float gravity = -9.81f;

    [Header("XPBI compliance")]
    public float compliance = 0f;

    public NodeBatch lastBatchDebug;

    public int[] levelEndIndex;

    public void FixNode(int nodeIdx) {
        nodes[nodeIdx].isFixed = true;
        nodes[nodeIdx].invMass = 0.0f;
        nodes[nodeIdx].vel = float2.zero;
    }

    public void Add(float2 pos) {
        Node newNode = new Node(pos, this);
        if (newNode.maxLayer > maxLayer) {
            maxLayer = newNode.maxLayer;
        }
        nodes.Add(newNode);
    }

    public void Build() {
        nodes = nodes.OrderByDescending(node => node.maxLayer).ToList();
        hnsw = new HNSW(this);

        const int volumeNeighborCount = 6;

        for (int i = 0; i < nodes.Count; i++) {
            Node node = nodes[i];

            List<int> neighbors = hnsw.SearchKnn(node.pos, volumeNeighborCount + 1);

            if (neighbors.Contains(i)) {
                neighbors.Remove(i);
            } else if (neighbors.Count > volumeNeighborCount) {
                neighbors.RemoveAt(volumeNeighborCount);
            }

            if (neighbors.Count < 2) {
                node.restVolume = 0.0f;
                continue;
            }

            int nCount = neighbors.Count;
            float2[] rel = new float2[nCount];
            float[] ang = new float[nCount];

            for (int k = 0; k < nCount; k++) {
                float2 v = nodes[neighbors[k]].pos - node.pos;
                rel[k] = v;
                ang[k] = math.atan2(v.y, v.x);
            }

            System.Array.Sort(ang, rel);

            float area = 0.0f;
            for (int k = 0; k < nCount; k++) {
                int next = (k + 1) % nCount;

                float dTheta = ang[next] - ang[k];
                if (dTheta < 0.0f) dTheta += 2.0f * math.PI;

                if (dTheta > math.PI) continue;

                float2 a = rel[k];
                float2 b = rel[next];

                float wedgeArea = 0.5f * math.abs(a.x * b.y - a.y * b.x);
                area += wedgeArea;
            }

            node.restVolume = area / 3.0f;
        }

        BuildHierarchy();
    }

    public void BuildHierarchy() {
        if (maxLayer < 0) return;

        levelEndIndex = new int[maxLayer + 1];
        int idx = 0;
        for (int level = maxLayer; level >= 0; level--) {
            for (; idx < nodes.Count && nodes[idx].maxLayer >= level; idx++) { }
            levelEndIndex[level] = idx;
        }

        for (int i = 0; i < nodes.Count; i++) {
            BuildParentRelationship(i);
        }
    }

    private void BuildParentRelationship(int nodeIdx) {
        Node node = nodes[nodeIdx];
        int parentLevel = node.maxLayer + 1;

        if (parentLevel > maxLayer) {
            node.parentIndex = -1;
            return;
        }

        var candidates = hnsw.SearchKnn(node.pos, 1, parentLevel);

        if (candidates.Count == 0) {
            node.parentIndex = -1;
            return;
        }

        node.parentIndex = candidates[0];
    }

    public int NodeCount(int level) {
        if (levelEndIndex == null || level < 0 || level > maxLayer) return 0;
        return levelEndIndex[level];
    }

    void OnEnable() => SimulationController.Instance?.Register(this);
    void OnDisable() => SimulationController.Instance?.Unregister(this);
}
