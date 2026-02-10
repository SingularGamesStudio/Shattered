using System.Collections.Generic;
using Physics;
using Unity.Mathematics;
using UnityEngine;

[DefaultExecutionOrder(-1000)]
public sealed class SimulationController : MonoBehaviour {
    public static SimulationController Instance { get; private set; }

    [Header("Rate")]
    [Min(1f)] public float targetTPS = 1000f;
    public float simulationSpeed = 1f;

    [Header("Mode")]
    [Tooltip("When enabled, simulation advances only via the T key (tap = 1 tick, hold = continuous).")]
    public bool manual = true;

    [Header("Catch-up")]
    [Min(1)] public int maxTicksPerFrame = 8;

    const float holdThreshold = 0.2f;
    public bool useUnscaledTime = false;

    [Header("Debug")]
    [Tooltip("If enabled, never execute more than 1 tick per rendered frame (TPS becomes FPS-limited).")]
    public bool forceTPSToFPS = false;

    [Header("Hierarchy")]
    public bool useHierarchicalSolver = true;

    readonly List<Meshless> meshless = new List<Meshless>(64);
    readonly Dictionary<Meshless, int> frameCounters = new Dictionary<Meshless, int>();
    readonly Dictionary<Meshless, float2[]> savedPositionsCache = new Dictionary<Meshless, float2[]>();

    float accumulator;
    float keyHeldTime;

    void Awake() {
        if (Instance != null && Instance != this) { Destroy(gameObject); return; }
        Instance = this;
    }

    void OnDestroy() {
        if (Instance == this) Instance = null;
    }

    public void Register(Meshless m) {
        if (m != null && !meshless.Contains(m)) {
            meshless.Add(m);
            frameCounters[m] = 0;
            savedPositionsCache[m] = new float2[m.nodes.Count];
        }
    }

    public void Unregister(Meshless m) {
        if (m != null) {
            meshless.Remove(m);
            frameCounters.Remove(m);
            savedPositionsCache.Remove(m);
        }
    }

    void Update() {
        float frameDt = useUnscaledTime ? Time.unscaledDeltaTime : Time.deltaTime;
        if (targetTPS <= 0f || frameDt <= 0f) return;

        if (manual) ManualUpdate(frameDt);
        else AutoUpdate(frameDt);
    }

    void AutoUpdate(float frameDt) {
        float dt = 1f / targetTPS;
        accumulator += frameDt;

        int ticks = 0;
        while (accumulator >= dt) {
            Tick(dt);
            accumulator -= dt;

            if (++ticks >= maxTicksPerFrame) break;
            if (forceTPSToFPS) break;
        }
    }

    void ManualUpdate(float frameDt) {
        if (Input.GetKeyDown(KeyCode.T)) {
            keyHeldTime = 0f;
            accumulator = 0f;
        }

        if (Input.GetKey(KeyCode.T)) {
            keyHeldTime += frameDt;
            if (keyHeldTime >= holdThreshold) AutoUpdate(frameDt);
        }

        if (Input.GetKeyUp(KeyCode.T)) {
            if (keyHeldTime < holdThreshold) Tick(1f / targetTPS);
            keyHeldTime = 0f;
            accumulator = 0f;
        }
    }

    void Tick(float dt) {
        for (int i = meshless.Count - 1; i >= 0; i--) {
            var m = meshless[i];
            if (m == null || !m.isActiveAndEnabled) {
                meshless.RemoveAt(i);
                frameCounters.Remove(m);
                savedPositionsCache.Remove(m);
                continue;
            }

            StepMeshless(m, dt * simulationSpeed);
        }
    }

    private void StepMeshless(Meshless meshless, float dt) {
        ApplyExternalForces(meshless, dt);
        Solve(meshless, dt);
        IntegrateAndUpdate(meshless, dt);

        if (!frameCounters.ContainsKey(meshless)) frameCounters[meshless] = 0;
        frameCounters[meshless]++;

        if (frameCounters[meshless] % Const.HierarchyRebuildInterval == 0) {
            meshless.BuildHierarchy();
        }
    }

    private void ApplyExternalForces(Meshless meshless, float dt) {
        for (int i = 0; i < meshless.nodes.Count; i++) {
            if (meshless.nodes[i].isFixed) continue;
            meshless.nodes[i].vel.y += meshless.gravity * dt;
        }
    }

    private void Solve(Meshless meshless, float dt) {
        int maxLevel = useHierarchicalSolver && meshless.levelEndIndex != null ? meshless.maxLayer : 0;

        if (!savedPositionsCache.TryGetValue(meshless, out float2[] saved) || saved.Length < meshless.nodes.Count) {
            saved = new float2[meshless.nodes.Count];
            savedPositionsCache[meshless] = saved;
        }

        NodeBatch batch = new NodeBatch(meshless.nodes, meshless.nodes.Count);

        for (int level = maxLevel; level >= 0; level--) {
            int nodeCount = level > 0 ? meshless.NodeCount(level) : meshless.nodes.Count;

            for (int i = 0; i < nodeCount; i++) {
                saved[i] = meshless.nodes[i].pos;
            }

            batch.ExpandTo(nodeCount);
            batch.Initialise();

            int iterations = level == 0 ? Const.Iterations : Const.HPBDIterations;
            for (int iter = 0; iter < iterations; iter++) {
                XPBIConstraint.Relax(batch, meshless.compliance, dt, iter);
            }

            if (level > 0) {
                ProlongateCorrections(meshless, level, nodeCount, saved);
            } else {
                meshless.lastBatchDebug = batch;
            }
        }

        XPBIConstraint.CommitDeformation(batch, dt);
    }

    private void ProlongateCorrections(Meshless meshless, int currentLevel, int currentLevelEnd, float2[] saved) {
        int fineLevelEnd = currentLevel > 1 ? meshless.NodeCount(currentLevel - 1) : meshless.nodes.Count;

        for (int i = currentLevelEnd; i < fineLevelEnd; i++) {
            Node node = meshless.nodes[i];
            if (node.parentIndex < 0 || node.parentIndex >= currentLevelEnd) continue;

            float2 parentCorrection = meshless.nodes[node.parentIndex].pos - saved[node.parentIndex];
            node.pos += parentCorrection;
        }
    }

    private void IntegrateAndUpdate(Meshless meshless, float dt) {
        for (int i = 0; i < meshless.nodes.Count; i++) {
            if (meshless.nodes[i].isFixed) continue;

            if (float.IsNaN(meshless.nodes[i].vel.x) || float.IsInfinity(meshless.nodes[i].vel.x))
                meshless.nodes[i].vel.x = 0f;
            if (float.IsNaN(meshless.nodes[i].vel.y) || float.IsInfinity(meshless.nodes[i].vel.y))
                meshless.nodes[i].vel.y = 0f;

            meshless.nodes[i].pos += meshless.nodes[i].vel * dt;
            meshless.hnsw.Shift(i, meshless.nodes[i].pos);
        }
    }
}
