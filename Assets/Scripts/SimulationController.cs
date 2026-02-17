using System.Collections.Generic;
using GPU.Solver;
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

    [Header("Hierarchy")]
    public bool useHierarchicalSolver = true;

    [Tooltip("Compute shader with kernels from XPBISolver.compute.")]
    public ComputeShader gpuXpbiSolverShader;

    readonly List<Meshless> meshless = new List<Meshless>(64);
    readonly Dictionary<Meshless, int> frameCounters = new Dictionary<Meshless, int>();
    readonly Dictionary<Meshless, float2[]> savedVelocitiesCache = new Dictionary<Meshless, float2[]>();

    readonly Dictionary<Meshless, XPBISolver> gpuSolverCache = new Dictionary<Meshless, XPBISolver>();

    float accumulator;
    float keyHeldTime;

    void Awake() {
        if (Instance != null && Instance != this) { Destroy(gameObject); return; }
        Instance = this;
    }

    void OnDestroy() {
        if (Instance == this) Instance = null;

        foreach (var kv in gpuSolverCache)
            kv.Value?.Dispose();
        gpuSolverCache.Clear();
    }

    public void Register(Meshless m) {
        if (m != null && !meshless.Contains(m)) {
            meshless.Add(m);
            frameCounters[m] = 0;
            savedVelocitiesCache[m] = new float2[m.nodes.Count];
        }
    }

    public void Unregister(Meshless m) {
        if (m != null) {
            meshless.Remove(m);
            frameCounters.Remove(m);
            savedVelocitiesCache.Remove(m);

            if (gpuSolverCache.TryGetValue(m, out XPBISolver solver)) {
                solver.Dispose();
                gpuSolverCache.Remove(m);
            }
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
                if (m != null) {
                    frameCounters.Remove(m);
                    savedVelocitiesCache.Remove(m);

                    if (gpuSolverCache.TryGetValue(m, out XPBISolver solver)) {
                        solver.Dispose();
                        gpuSolverCache.Remove(m);
                    }
                }
                continue;
            }
            StepMeshlessGpu(m, dt * simulationSpeed);
        }
    }

    void StepMeshlessGpu(Meshless meshless, float dt) {
        if (!gpuSolverCache.TryGetValue(meshless, out XPBISolver solver) || solver == null) {
            solver = new XPBISolver(gpuXpbiSolverShader);
            gpuSolverCache[meshless] = solver;
        }

        solver.UploadFromMeshless(meshless);
        solver.SolveHierarchical(meshless, dt, useHierarchicalSolver);
        solver.DownloadToMeshless(meshless);

        IntegrateAndUpdate(meshless, dt, dtReadback: false);

        if (!frameCounters.ContainsKey(meshless)) frameCounters[meshless] = 0;
        frameCounters[meshless]++;

        if (frameCounters[meshless] % Const.HierarchyRebuildInterval == 0) {
            meshless.BuildHierarchyWithDtReadback();
        }
    }

    private void IntegrateAndUpdate(Meshless meshless, float dt, bool dtReadback) {
        for (int i = 0; i < meshless.nodes.Count; i++) {
            if (meshless.nodes[i].isFixed) continue;

            if (float.IsNaN(meshless.nodes[i].vel.x) || float.IsInfinity(meshless.nodes[i].vel.x))
                meshless.nodes[i].vel.x = 0f;
            if (float.IsNaN(meshless.nodes[i].vel.y) || float.IsInfinity(meshless.nodes[i].vel.y))
                meshless.nodes[i].vel.y = 0f;

            meshless.nodes[i].pos += meshless.nodes[i].vel * dt;
        }

        meshless.UpdateDelaunayAfterIntegration(dtReadback);
    }
}
