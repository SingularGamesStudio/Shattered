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
    const float fixedTimeStep = 0.001f;
    const float timeScale = 0.1f;
    const float gravity = -9.81f;

    // Constraints
    public bool enableEdgeConstraints = true;
    public bool enableVolumeConstraints = true;

    [Header("Constraint Compliances")]
    public float edgeCompliance = 0.01f;
    public float volumeCompliance = 0.01f;

    private List<Constraint> activeConstraints = new List<Constraint>();
    private NeighborDistanceConstraint edgeConstraint;
    private VolumeConstraint volumeConstraint;

    // Debug
    public NodeBatch lastBatchDebug;

    void Start() {
        edgeConstraint = new NeighborDistanceConstraint();
        volumeConstraint = new VolumeConstraint();

        UpdateActiveConstraints();
    }

    private void UpdateActiveConstraints() {
        activeConstraints.Clear();
        if (enableEdgeConstraints) activeConstraints.Add(edgeConstraint);
        if (enableVolumeConstraints) activeConstraints.Add(volumeConstraint);
    }

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
    }

    private float keyHoldTime = 0.0f;
    const float holdThreshold = 0.2f;

    void Update() {
        if (Input.GetKeyDown(KeyCode.T)) {
            keyHoldTime = 0.0f;
        }

        if (Input.GetKey(KeyCode.T)) {
            keyHoldTime += Time.deltaTime;
            if (keyHoldTime >= holdThreshold) {
                StepSimulation(Time.deltaTime * timeScale);
            }
        }

        if (Input.GetKeyUp(KeyCode.T)) {
            if (keyHoldTime < holdThreshold) {
                StepSimulation(fixedTimeStep);
            }
            keyHoldTime = 0.0f;
        }
    }

    public void StepSimulation(float timeStep) {
        UpdateActiveConstraints();

        NodeBatch batch = new NodeBatch(nodes);

        // Initialize all active constraints (safe to call multiple times for shared cache)
        foreach (var constraint in activeConstraints) {
            constraint.Initialise(batch);
        }

        // Initialize lambdas once from main loop
        batch.InitializeLambdas();

        // Apply external forces
        for (int i = 0; i < nodes.Count; i++) {
            if (nodes[i].isFixed) continue;
            nodes[i].vel.y += gravity * timeStep;
        }

        // XPBD constraint solver iterations
        for (int iter = 0; iter < Const.SolverIterations; iter++) {
            foreach (var constraint in activeConstraints) {
                float compliance = GetCompliance(constraint);
                constraint.Relax(batch, compliance, timeStep);
            }
        }

        // Update plastic deformation after convergence (XPBI Eq. 22)
        foreach (var constraint in activeConstraints) {
            constraint.UpdatePlasticDeformation(batch, timeStep);
        }

        // Integrate positions
        for (int i = 0; i < nodes.Count; i++) {
            if (nodes[i].isFixed) continue;

            // Safety check for NaN/Inf
            if (float.IsNaN(nodes[i].vel.x) || float.IsInfinity(nodes[i].vel.x)) nodes[i].vel.x = 0f;
            if (float.IsNaN(nodes[i].vel.y) || float.IsInfinity(nodes[i].vel.y)) nodes[i].vel.y = 0f;

            nodes[i].pos += nodes[i].vel * timeStep;
            hnsw.Shift(i, nodes[i].pos);
        }

        lastBatchDebug = batch;
    }

    private float GetCompliance(Constraint constraint) {
        if (constraint is NeighborDistanceConstraint) return edgeCompliance;
        if (constraint is VolumeConstraint) return volumeCompliance;
        return 0.01f;
    }
}