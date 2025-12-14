using System.Collections.Generic;
using System.Linq;
using Physics;
using Unity.Mathematics;
using UnityEngine;

public class Meshless : MonoBehaviour {
    public HNSW hnsw;
    public List<Node> nodes = new List<Node>();
    [HideInInspector]
    public int maxLayer = -1;

    // Simulation parameters
    const float gravity = -9.81f;

    [Header("XPBI compliance")]
    public float compliance = 0f;

    private XPBIConstraint XPBI = new XPBIConstraint();

    // Debug
    public NodeBatch lastBatchDebug;

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

            // Query k+1 neighbors (includes self)
            List<int> neighbors = hnsw.SearchKnn(node.pos, volumeNeighborCount + 1);

            // Remove self from neighbors
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
    }

    void OnEnable() => MeshlessSimulationController.Instance?.Register(this);
    void OnDisable() => MeshlessSimulationController.Instance?.Unregister(this);

    public void StepSimulation(float timeStep) {
        NodeBatch batch = new NodeBatch(nodes);

        // Initialize XPBI constitutive constraint (neighbors, correction matrices, F/Fp, lambdas)
        XPBI.Initialise(batch);

        // Apply external forces
        for (int i = 0; i < nodes.Count; i++) {
            if (nodes[i].isFixed) continue;
            nodes[i].vel.y += gravity * timeStep;
        }

        // XPBI/XPBD iterations with single constitutive constraint per particle
        for (int iter = 0; iter < Const.SolverIterations; iter++) {
            XPBI.Relax(batch, compliance, timeStep);
        }
        XPBI.CommitDeformation(batch, timeStep);

        // Integrate positions
        for (int i = 0; i < nodes.Count; i++) {
            if (nodes[i].isFixed) continue;

            if (float.IsNaN(nodes[i].vel.x) || float.IsInfinity(nodes[i].vel.x)) nodes[i].vel.x = 0f;
            if (float.IsNaN(nodes[i].vel.y) || float.IsInfinity(nodes[i].vel.y)) nodes[i].vel.y = 0f;

            nodes[i].pos += nodes[i].vel * timeStep;
            hnsw.Shift(i, nodes[i].pos);
        }

        lastBatchDebug = batch;
    }
}
