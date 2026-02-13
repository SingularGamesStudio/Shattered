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

    [Header("Profiling")]
    public bool enableLoopProfiling = false;
    [Min(1)] public int profileTicks = 2000;
    [Min(0)] public int profileWarmupTicks = 200;
    public string profileScenario = "default";
    public KeyCode profileStartKey = KeyCode.P;

    [Tooltip("Absolute path, or relative. Relative resolves to project root in Editor, persistentDataPath in Player.")]
    public string profileOutputDirectory = "Profiles";

    readonly List<Meshless> meshless = new List<Meshless>(64);
    readonly Dictionary<Meshless, int> frameCounters = new Dictionary<Meshless, int>();
    readonly Dictionary<Meshless, float2[]> savedVelocitiesCache = new Dictionary<Meshless, float2[]>();
    readonly Dictionary<Meshless, NodeBatch> batchCache = new Dictionary<Meshless, NodeBatch>();

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
            savedVelocitiesCache[m] = new float2[m.nodes.Count];
            batchCache[m] = new NodeBatch(m.nodes, m.nodes.Count);
        }
    }

    public void Unregister(Meshless m) {
        if (m != null) {
            meshless.Remove(m);
            frameCounters.Remove(m);
            savedVelocitiesCache.Remove(m);
            batchCache.Remove(m);
        }
    }

    void Update() {
        LoopProfiler.SetEnabledInController(enableLoopProfiling);

        if (enableLoopProfiling && Input.GetKeyDown(profileStartKey)) {
            LoopProfiler.StartCapture(profileTicks, profileWarmupTicks, profileScenario, profileOutputDirectory);
        }

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
        int tickMeshlessCount = 0;
        int tickNodesTotal = 0;
        int tickMaxNodes = 0;
        int tickMaxLevel = 0;

        if (LoopProfiler.IsActive) {
            tickMeshlessCount = meshless.Count;
            for (int i = 0; i < meshless.Count; i++) {
                var m = meshless[i];
                if (m == null || !m.isActiveAndEnabled) continue;

                int n = m.nodes.Count;
                tickNodesTotal += n;
                if (n > tickMaxNodes) tickMaxNodes = n;

                int level = useHierarchicalSolver && m.levelEndIndex != null ? m.maxLayer : 0;
                if (level > tickMaxLevel) tickMaxLevel = level;
            }
        }

        LoopProfiler.BeginTick(dt, tickMeshlessCount, tickNodesTotal, tickMaxNodes, tickMaxLevel, useHierarchicalSolver);

        long tickStart = LoopProfiler.Stamp();

        for (int i = meshless.Count - 1; i >= 0; i--) {
            var m = meshless[i];
            if (m == null || !m.isActiveAndEnabled) {
                meshless.RemoveAt(i);
                if (m != null) {
                    frameCounters.Remove(m);
                    savedVelocitiesCache.Remove(m);
                    batchCache.Remove(m);
                }
                continue;
            }

            StepMeshless(m, dt * simulationSpeed);
        }

        LoopProfiler.Add(LoopProfiler.Section.TickTotal, tickStart);
        LoopProfiler.EndTick();
    }

    private void StepMeshless(Meshless meshless, float dt) {
        long t = LoopProfiler.Stamp();
        ApplyExternalForces(meshless, dt);
        LoopProfiler.Add(LoopProfiler.Section.ExternalForces, t);

        t = LoopProfiler.Stamp();
        Solve(meshless, dt);
        LoopProfiler.Add(LoopProfiler.Section.SolveTotal, t);

        t = LoopProfiler.Stamp();
        IntegrateAndUpdate(meshless, dt);
        LoopProfiler.Add(LoopProfiler.Section.IntegrateAndUpdate, t);

        if (!frameCounters.ContainsKey(meshless)) frameCounters[meshless] = 0;
        frameCounters[meshless]++;

        if (frameCounters[meshless] % Const.HierarchyRebuildInterval == 0) {
            t = LoopProfiler.Stamp();
            meshless.BuildHierarchy();
            LoopProfiler.Add(LoopProfiler.Section.HierarchyRebuild, t);
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

        if (!savedVelocitiesCache.TryGetValue(meshless, out float2[] savedVel) || savedVel.Length < meshless.nodes.Count) {
            savedVel = new float2[meshless.nodes.Count];
            savedVelocitiesCache[meshless] = savedVel;
        }

        if (!batchCache.TryGetValue(meshless, out NodeBatch batch) || batch == null || batch.nodes != meshless.nodes) {
            batch = new NodeBatch(meshless.nodes, meshless.nodes.Count);
            batchCache[meshless] = batch;
        }

        batch.BeginStep();

        for (int level = maxLevel; level >= 0; level--) {
            int nodeCount = level > 0 ? meshless.NodeCount(level) : meshless.nodes.Count;

            long t = LoopProfiler.Stamp();
            for (int i = 0; i < nodeCount; i++) {
                savedVel[i] = meshless.nodes[i].vel;
            }
            LoopProfiler.Add(LoopProfiler.Section.SolveSavedVelCopy, t);

            batch.ExpandTo(nodeCount);

            t = LoopProfiler.Stamp();
            batch.Initialise();
            LoopProfiler.Add(LoopProfiler.Section.SolveBatchInit, t);

            int iterations = level == 0 ? Const.Iterations : Const.HPBDIterations;
            t = LoopProfiler.Stamp();
            for (int iter = 0; iter < iterations; iter++) {
                XPBIConstraint.Relax(batch, meshless.compliance, dt, iter);
            }
            LoopProfiler.Add(LoopProfiler.Section.SolveRelax, t);

            if (level > 0) {
                t = LoopProfiler.Stamp();
                ProlongateVelocityCorrections(meshless, level, nodeCount, savedVel);
                LoopProfiler.Add(LoopProfiler.Section.SolveProlongate, t);
            } else {
                t = LoopProfiler.Stamp();
                batch.FinalizeDebugData();
                meshless.lastBatchDebug = batch;
                LoopProfiler.Add(LoopProfiler.Section.SolveFinalizeDebug, t);
            }
        }

        long commitStart = LoopProfiler.Stamp();
        XPBIConstraint.CommitDeformation(batch, dt);
        LoopProfiler.Add(LoopProfiler.Section.SolveCommitDeformation, commitStart);
    }

    private void ProlongateVelocityCorrections(Meshless meshless, int currentLevel, int currentLevelEnd, float2[] savedVel) {
        int fineLevelEnd = currentLevel > 1 ? meshless.NodeCount(currentLevel - 1) : meshless.nodes.Count;

        for (int i = currentLevelEnd; i < fineLevelEnd; i++) {
            Node node = meshless.nodes[i];
            if (node.parentIndex < 0 || node.parentIndex >= currentLevelEnd) continue;

            float2 parentDeltaV = meshless.nodes[node.parentIndex].vel - savedVel[node.parentIndex];
            node.vel += parentDeltaV;
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
            meshless.hnsw.Shift(i, meshless.nodes[i].pos);//TODO: 5-7% of total tick time, not GPU-friendly
        }
    }
}
