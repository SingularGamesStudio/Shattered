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
    const int constraintIters = 1;

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
        float lagrangeMultNeighbor = 0;

        Physics.Constraint tension = new TensionConstraint();
        tension.Initialise(nodes);
        float lagrangeMultTension = 0;

        // 1. Apply external forces and predict positions
        foreach (var node in nodes) {
            if (node.isFixed) {
                node.predPos = node.pos;
                continue;
            }
            node.vel.y += gravity * timeStep;
            node.vel.x += (UnityEngine.Random.value - 0.5f) * 1f;
            node.predPos = node.pos + node.vel * timeStep;
        }

        // 2. Constraint Solver Iterations
        for (int iter = 0; iter < constraintIters; ++iter) {
            lagrangeMultNeighbor = neighbor.Relax(nodes, 0.01f, lagrangeMultNeighbor, timeStep);
            lagrangeMultTension = neighbor.Relax(nodes, 0f, lagrangeMultTension, timeStep);
        }

        float contractionSmoothing = 0.5f; // 0=no update, 1=instant set

        for (int i = 0; i < nodes.Count; ++i) {
            var neigh = nodes[i].constraintCache.neighbors;
            if (neigh.Count == 0) {
                nodes[i].contraction = nodes[i].contraction; // keep prior memory if no neighbors
                continue;
            }
            float accum = 0f;
            for (int n = 0; n < neigh.Count; ++n) {
                int j = neigh[n];
                float strain = (math.distance(nodes[i].pos, nodes[j].pos) - math.distance(nodes[i].predPos, nodes[j].predPos)) / math.distance(nodes[i].pos, nodes[j].pos);

                // Clamp to avoid outliers from near-coincident points
                accum += math.clamp(strain, -1f, 1f);
            }
            float avgStrain = accum / neigh.Count;
            nodes[i].contraction = math.clamp(math.lerp(nodes[i].contraction, avgStrain, math.clamp(contractionSmoothing, 0f, 1f)), -1f, 1f);
        }


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
}