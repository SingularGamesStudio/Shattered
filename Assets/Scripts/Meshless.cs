using System.Collections.Generic;
using System.Data;
using System.Linq;
using Physics;
using Unity.Mathematics;
using Unity.VisualScripting.Antlr3.Runtime.Misc;
using UnityEditor;
using UnityEngine;

public class Meshless : MonoBehaviour {
    public void FixNode(int nodeIdx) {
        nodes[nodeIdx].isFixed = true;
        nodes[nodeIdx].invMass = 0.0f;
        nodes[nodeIdx].vel = float2.zero;
    }

    public HNSW hnsw;
    public List<Node> nodes = new List<Node>();
    [HideInInspector]
    public int maxLayer = -1;
    public float velocityCap = 10f;

    const float fixedTimeStep = 0.01f;
    const float timeScale = 3f;
    const float gravity = -9.81f;

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
    }

    private float keyHoldTime = 0.0f;
    const float holdThreshold = 0.2f; // seconds before considered held

    void Update() {
        if (Input.GetKeyDown(KeyCode.T)) {
            keyHoldTime = 0.0f; // Start timer on key down
        }

        if (Input.GetKey(KeyCode.T)) {
            keyHoldTime += Time.deltaTime;

            if (keyHoldTime >= holdThreshold) {
                StepSimulation(Time.deltaTime * timeScale);
            }
        }

        if (Input.GetKeyUp(KeyCode.T)) {
            if (keyHoldTime < holdThreshold) {
                // Considered a tap: single fixed step
                StepSimulation(fixedTimeStep);
            }

            keyHoldTime = 0.0f; // Reset timer on release
        }
    }

    public NodeBatch lastBatchDebug;

    // Manual simulation step triggered by button
    public void StepSimulation(float timeStep) {
        NodeBatch prepared = new NodeBatch(nodes);
        Physics.Constraint neighbor = new NeighborDistanceConstraint();
        neighbor.Initialise(prepared);
        Physics.Constraint volume = new VolumeConstraint();
        volume.Initialise(prepared);
        prepared.CacheLambdas();

        // 1. Apply external forces and predict positions
        foreach (var node in nodes) {
            if (node.isFixed) {
                node.predPos = node.pos;
                continue;
            }
            node.vel.y += gravity * timeStep;
            node.predPos = node.pos + node.vel * timeStep;
        }

        // 2. Constraint Solver Iterations
        for (int iter = 0; iter < Const.SolverIterations; ++iter) {
            neighbor.Relax(prepared, 0.01f, timeStep);
            volume.Relax(prepared, 0.0001f, timeStep);
        }

        // 3. PLASTIC FLOW STEP (NEW – XPBI)
        neighbor.PlasticFlow(prepared, timeStep);
        volume.PlasticFlow(prepared, timeStep);

        // 4. Cap constraint corrections
        for (int i = 0; i < nodes.Count; ++i) {
            float2 c = nodes[i].predPos - nodes[i].pos;
            float m = velocityCap * timeStep; // max correction per step from constraints
            if (math.lengthsq(c) > m * m) nodes[i].predPos = nodes[i].pos + math.normalize(c) * m;
            if (float.IsNaN(nodes[i].predPos.x) || float.IsInfinity(nodes[i].predPos.x)) {
                nodes[i].predPos.x = nodes[i].pos.x;
                Debug.LogWarning("inf");
            }
            if (float.IsNaN(nodes[i].predPos.y) || float.IsInfinity(nodes[i].predPos.y)) {
                nodes[i].predPos.y = nodes[i].pos.y;
            }
        }

        // 5. Update velocities and positions
        for (int i = 0; i < nodes.Count; ++i) {
            if (nodes[i].isFixed) {
                nodes[i].vel = float2.zero;
                nodes[i].pos = nodes[i].predPos;
                continue;
            }
            float2 dampedPosition = math.lerp(nodes[i].pos, nodes[i].predPos, 0.9f);
            nodes[i].vel = (dampedPosition - nodes[i].pos) / timeStep * 0.9f;
            hnsw.Shift(i, dampedPosition);
        }

        lastBatchDebug = prepared;
    }
}