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

    const float fixedTimeStep = 0.05f;
    const float gravity = -9.81f;
    const int constraintIters = 8;

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
                // Held long enough: continuous stepping
                float delta = Time.deltaTime * 2;
                StepSimulation(delta);
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

    // Manual simulation step triggered by button
    public void StepSimulation(float timeStep) {
        Physics.Constraint neighbor = new NeighborDistanceConstraint();
        neighbor.Initialise(nodes);

        Physics.Constraint tension = new TensionConstraint();
        tension.Initialise(nodes);

        // 1. Apply external forces and predict positions
        foreach (var node in nodes) {
            if (node.isFixed) {
                node.predPos = node.pos;
                continue;
            }
            node.vel.y += gravity * timeStep;
            //node.vel.x += (UnityEngine.Random.value - 0.5f) * 1f;
            node.predPos = node.pos + node.vel * timeStep;
        }

        // 2. Constraint Solver Iterations
        for (int iter = 0; iter < constraintIters; ++iter) {
            neighbor.Relax(nodes, 0.01f, timeStep);
            tension.Relax(nodes, 0.01f, timeStep);
        }

        UpdateNodeTensions();

        // 3. Update velocities and positions
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
    }

    public void UpdateNodeTensions(float gain = 1f, float decay = 1f, float minRatio = 0.2f, float maxRatio = 5f) {
        int count = nodes.Count;
        for (int i = 0; i < count; ++i) {
            var A = nodes[i];
            var cache = A.constraintCache;
            if (cache == null || cache.neighbors == null || cache.neighborDistances == null) continue;

            // Geometric mean of per-edge length ratios for this frame
            float sumLog = 0f;
            int samples = 0;
            for (int n = 0; n < cache.neighbors.Count; ++n) {
                int j = cache.neighbors[n];
                if (j < 0 || j >= count || j == i) continue;

                float rest = cache.neighborDistances[n];
                if (rest <= 1e-6f) continue;

                float d = math.distance(A.predPos, nodes[j].predPos);
                float ratio = math.max(d / rest, 1e-6f);
                sumLog += math.log(ratio);
                samples++;
            }
            if (samples == 0) continue;

            float rho = math.exp(sumLog / samples); // geometric mean ratio
            float rOld = math.clamp(A.contraction, 1e-6f, 1e6f);

            // Multiplicative accumulation with multiplicative decay toward 1
            float rNew = math.pow(rOld, math.saturate(decay)) * math.pow(rho, gain);

            A.contraction = math.clamp(rNew, minRatio, maxRatio);
        }
    }
}