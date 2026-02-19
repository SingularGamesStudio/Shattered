using System.Collections.Generic;
using GPU.Solver;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

[DefaultExecutionOrder(-1000)]
public sealed class SimulationController : MonoBehaviour {
    public static SimulationController Instance { get; private set; }

    [Header("Rate")]
    [Min(1f)] public float targetTPS = 1000f;
    [Min(1f)] public float targetFPS = 60f;
    public float simulationSpeed = 1f;

    [Header("Mode")]
    [Tooltip("When enabled, simulation advances only via the T key (tap = 1 tick, hold = continuous).")]
    public bool manual = true;

    [Header("UI")]
    public bool showTpsOverlay = true;
    public Vector2 tpsOverlayPos = new Vector2(10f, 10f);

    const float holdThreshold = 0.2f;
    public bool useUnscaledTime = false;

    [Header("Hierarchy")]
    public bool useHierarchicalSolver = true;

    [Tooltip("Compute shader with kernels from XPBISolver.compute.")]
    public ComputeShader gpuXpbiSolverShader;

    [Header("Async GPU (experimental)")]
    [Tooltip("When enabled, submits XPBI ticks as async compute batches, and swaps DT position buffers only after a GPU fence passes.")]
    public bool asyncGpu = true;

    [Tooltip("Desired async compute queue; if the platform doesn't support async compute, Unity executes on the graphics queue.")]
    public ComputeQueueType asyncQueue = ComputeQueueType.Background;

    [Min(1)] public int maxTicksPerBatch = 32;

    [Tooltip("Updates DT positions buffers for rendering after an async batch. Topology maintenance is disabled in async mode by default.")]
    public bool asyncUpdateDtPositions = true;

    [Tooltip("If enabled, still runs DT.Maintain() inside ticks. This will contend with rendering because DT topology buffers are shared.")]
    public bool asyncRunDtMaintain = false;

    [Header("CPU snapshots (AI)")]
    [Min(1)] public int snapshotEveryTicks = 10;

    readonly List<Meshless> meshless = new List<Meshless>(64);
    readonly Dictionary<Meshless, XPBISolver> gpuSolverCache = new Dictionary<Meshless, XPBISolver>();
    readonly Dictionary<Meshless, int> tickCounters = new Dictionary<Meshless, int>();

    struct ReadbackSlot {
        public AsyncGPUReadbackRequest request;
        public bool pending;
        public int count;
    }

    readonly Dictionary<Meshless, ReadbackSlot> readbackSlots = new Dictionary<Meshless, ReadbackSlot>();
    readonly Dictionary<Meshless, float2[]> latestSnapshot = new Dictionary<Meshless, float2[]>();
    readonly Dictionary<Meshless, int> latestSnapshotTick = new Dictionary<Meshless, int>();

    struct AsyncBatchState {
        public bool pending;
        public GraphicsFence fence;
        public int tickIdAfterBatch;
        public int dtSwapMaxLevel;
    }

    readonly Dictionary<Meshless, AsyncBatchState> asyncStates = new Dictionary<Meshless, AsyncBatchState>();

    float accumulator;
    float keyHeldTime;

    int lastFrameTicks;
    float tpsSmoothed;
    float fpsSmoothed;


    float renderAlpha;
    public float RenderAlpha => renderAlpha;

    void Awake() {
        if (Instance != null && Instance != this) { Destroy(gameObject); return; }
        Instance = this;
    }

    void OnDestroy() {
        if (Instance == this) Instance = null;

        foreach (var kv in gpuSolverCache)
            kv.Value?.Dispose();
        gpuSolverCache.Clear();

        tickCounters.Clear();
        readbackSlots.Clear();
        latestSnapshot.Clear();
        latestSnapshotTick.Clear();
        asyncStates.Clear();
    }

    public void Register(Meshless m) {
        if (m != null && !meshless.Contains(m)) {
            meshless.Add(m);
            tickCounters[m] = 0;
            readbackSlots[m] = default;
            asyncStates[m] = default;
        }
    }

    public void Unregister(Meshless m) {
        if (m != null) {
            meshless.Remove(m);
            tickCounters.Remove(m);
            readbackSlots.Remove(m);
            latestSnapshot.Remove(m);
            latestSnapshotTick.Remove(m);
            asyncStates.Remove(m);

            if (gpuSolverCache.TryGetValue(m, out XPBISolver solver)) {
                solver.Dispose();
                gpuSolverCache.Remove(m);
            }
        }
    }

    public bool TryGetLatestPositionsSnapshot(Meshless m, out float2[] positions, out int count, out int tickId) {
        if (m != null && latestSnapshot.TryGetValue(m, out positions)) {
            count = positions.Length;
            tickId = latestSnapshotTick.TryGetValue(m, out int t) ? t : 0;
            return true;
        }

        positions = null;
        count = 0;
        tickId = 0;
        return false;
    }

    void Update() {
        if (targetTPS <= 0f)
            return;

        float frameDt = useUnscaledTime ? Time.unscaledDeltaTime : Time.deltaTime;
        if (frameDt < 0f) frameDt = 0f;

        float tickDt = 1f / targetTPS;

        lastFrameTicks = 0;

        if (asyncGpu) {
            ProcessAsyncBatchCompletions();

            if (manual) ManualUpdateAsync(frameDt, tickDt);
            else AutoUpdateAsync(frameDt, tickDt);

            renderAlpha = tickDt > 0f ? Mathf.Clamp01(accumulator / tickDt) : 0f;
        } else {
            float clampedFps = Mathf.Max(1f, targetFPS);
            float budgetSeconds = (1f / clampedFps);
            float frameEndTime = Time.realtimeSinceStartup + budgetSeconds;

            if (manual) ManualUpdate(frameDt, tickDt, frameEndTime);
            else AutoUpdate(frameDt, tickDt, frameEndTime);

            renderAlpha = tickDt > 0f ? Mathf.Clamp01(accumulator / tickDt) : 0f;

            ProcessReadbacks();
        }

        UpdateTpsDisplay(frameDt);
    }

    void AutoUpdateAsync(float frameDt, float tickDt) {
        accumulator += frameDt;

        int ticksToRun = 0;
        while (accumulator >= tickDt && ticksToRun < maxTicksPerBatch) {
            accumulator -= tickDt;
            ticksToRun++;
        }

        if (ticksToRun <= 0) return;

        float dt = tickDt * simulationSpeed;

        for (int i = meshless.Count - 1; i >= 0; i--) {
            var m = meshless[i];
            if (m == null || !m.isActiveAndEnabled) {
                meshless.RemoveAt(i);
                if (m != null) Unregister(m);
                continue;
            }

            SubmitMeshlessAsyncBatch(m, dt, ticksToRun);
        }

        lastFrameTicks += ticksToRun;
    }

    void ManualUpdateAsync(float frameDt, float tickDt) {
        if (Input.GetKeyDown(KeyCode.T)) {
            keyHeldTime = 0f;
            accumulator = 0f;
        }

        if (Input.GetKey(KeyCode.T)) {
            keyHeldTime += frameDt;

            if (keyHeldTime >= holdThreshold) {
                accumulator += frameDt;

                int ticksToRun = 0;
                while (accumulator >= tickDt && ticksToRun < maxTicksPerBatch) {
                    accumulator -= tickDt;
                    ticksToRun++;
                }

                if (ticksToRun > 0) {
                    float dt = tickDt * simulationSpeed;

                    for (int i = meshless.Count - 1; i >= 0; i--) {
                        var m = meshless[i];
                        if (m == null || !m.isActiveAndEnabled) {
                            meshless.RemoveAt(i);
                            if (m != null) Unregister(m);
                            continue;
                        }

                        SubmitMeshlessAsyncBatch(m, dt, ticksToRun);
                    }

                    lastFrameTicks += ticksToRun;
                }
            }
        }

        if (Input.GetKeyUp(KeyCode.T)) {
            if (keyHeldTime < holdThreshold) {
                float dt = tickDt * simulationSpeed;

                for (int i = meshless.Count - 1; i >= 0; i--) {
                    var m = meshless[i];
                    if (m == null || !m.isActiveAndEnabled) {
                        meshless.RemoveAt(i);
                        if (m != null) Unregister(m);
                        continue;
                    }

                    SubmitMeshlessAsyncBatch(m, dt, 1);
                }

                lastFrameTicks += 1;
            }

            keyHeldTime = 0f;
            accumulator = 0f;
        }
    }

    void AutoUpdate(float frameDt, float tickDt, float frameEndTime) {
        accumulator += frameDt;

        int ticks = 0;
        float now = Time.realtimeSinceStartup;

        while (accumulator >= tickDt && now <= frameEndTime) {
            Tick(tickDt);
            accumulator -= tickDt;
            ticks++;
            now = Time.realtimeSinceStartup;
        }

        if (accumulator >= tickDt && now > frameEndTime)
            accumulator = 0f;

        lastFrameTicks += ticks;
    }

    void ManualUpdate(float frameDt, float tickDt, float frameEndTime) {
        if (Input.GetKeyDown(KeyCode.T)) {
            keyHeldTime = 0f;
            accumulator = 0f;
        }

        if (Input.GetKey(KeyCode.T)) {
            keyHeldTime += frameDt;

            if (keyHeldTime >= holdThreshold) {
                accumulator += frameDt;

                int ticks = 0;
                float now = Time.realtimeSinceStartup;

                while (accumulator >= tickDt && now <= frameEndTime) {
                    Tick(tickDt);
                    accumulator -= tickDt;
                    ticks++;
                    now = Time.realtimeSinceStartup;
                }

                lastFrameTicks += ticks;
            }
        }

        if (Input.GetKeyUp(KeyCode.T)) {
            if (keyHeldTime < holdThreshold) {
                Tick(tickDt);
                lastFrameTicks += 1;
            }
            keyHeldTime = 0f;
            accumulator = 0f;
        }
    }

    void Tick(float tickDt) {
        float dt = tickDt * simulationSpeed;

        for (int i = meshless.Count - 1; i >= 0; i--) {
            var m = meshless[i];
            if (m == null || !m.isActiveAndEnabled) {
                meshless.RemoveAt(i);
                if (m != null) Unregister(m);
                continue;
            }

            StepMeshlessGpuTruth(m, dt);
        }
    }

    void SubmitMeshlessAsyncBatch(Meshless m, float dtPerTick, int ticksToRun) {
        if (m == null || ticksToRun <= 0)
            return;

        if (!gpuSolverCache.TryGetValue(m, out XPBISolver solver) || solver == null) {
            solver = new XPBISolver(gpuXpbiSolverShader);
            gpuSolverCache[m] = solver;
            solver.InitializeFromMeshless(m);
        }

        if (!tickCounters.ContainsKey(m)) tickCounters[m] = 0;
        int lastTickId = tickCounters[m];
        int tickIdAfterBatch = lastTickId + ticksToRun;

        if (!asyncStates.TryGetValue(m, out AsyncBatchState st))
            st = default;

        if (st.pending && !st.fence.passed)
            return;

        if (st.pending && st.fence.passed) {
            solver.CompleteAsyncBatch(m, st.dtSwapMaxLevel);
            st.pending = false;
            asyncStates[m] = st;
        }

        bool rebuildParents = ((tickIdAfterBatch % Const.HierarchyRebuildInterval) == 0);

        int dtSwapMaxLevel = asyncUpdateDtPositions ? (asyncRunDtMaintain ? m.maxLayer : 0) : -1;

        st.fence = solver.SubmitGpuTruthAsyncBatch(
            m,
            dtPerTick,
            ticksToRun,
            useHierarchicalSolver,
            m.dtFixIterationsPerTick,
            m.dtLegalizeIterationsPerTick,
            rebuildParents,
            asyncUpdateDtPositions,
            dtSwapMaxLevel,
            asyncRunDtMaintain,
            asyncQueue
        );

        st.pending = true;
        st.tickIdAfterBatch = tickIdAfterBatch;
        st.dtSwapMaxLevel = dtSwapMaxLevel;
        asyncStates[m] = st;

        tickCounters[m] = tickIdAfterBatch;

        // In async mode, snapshots are disabled by default because they create extra GPU traffic and can defeat the "smooth render" goal.
        // If you need them, re-enable and ensure you request them only after fences pass.
    }

    void ProcessAsyncBatchCompletions() {
        if (asyncStates.Count == 0) return;

        var keys = ListPool<Meshless>.Get();
        foreach (var kv in asyncStates) keys.Add(kv.Key);

        for (int i = 0; i < keys.Count; i++) {
            var m = keys[i];
            if (m == null) continue;

            if (!asyncStates.TryGetValue(m, out AsyncBatchState st))
                continue;

            if (!st.pending)
                continue;

            if (!st.fence.passed)
                continue;

            if (gpuSolverCache.TryGetValue(m, out XPBISolver solver) && solver != null)
                solver.CompleteAsyncBatch(m, st.dtSwapMaxLevel);

            st.pending = false;
            asyncStates[m] = st;
        }

        ListPool<Meshless>.Release(keys);
    }

    void StepMeshlessGpuTruth(Meshless m, float dt) {
        if (!gpuSolverCache.TryGetValue(m, out XPBISolver solver) || solver == null) {
            solver = new XPBISolver(gpuXpbiSolverShader);
            gpuSolverCache[m] = solver;
            solver.InitializeFromMeshless(m);
        }

        if (!tickCounters.ContainsKey(m)) tickCounters[m] = 0;
        int tickId = ++tickCounters[m];

        bool rebuildParents = (tickId % Const.HierarchyRebuildInterval) == 0;

        solver.StepGpuTruth(
            m,
            dt,
            useHierarchicalSolver,
            m.dtFixIterationsPerTick,
            m.dtLegalizeIterationsPerTick,
            rebuildParents);

        if (snapshotEveryTicks > 0 && (tickId % snapshotEveryTicks) == 0)
            TryScheduleSnapshotReadback(m, solver, m.nodes.Count, tickId);
    }

    void TryScheduleSnapshotReadback(Meshless m, XPBISolver solver, int count, int tickId) {
        if (solver == null || solver.PositionBuffer == null) return;
        if (count <= 0) return;

        if (!readbackSlots.TryGetValue(m, out ReadbackSlot slot))
            slot = default;

        if (slot.pending)
            return;

        int bytes = count * 8; // float2
        slot.request = AsyncGPUReadback.Request(solver.PositionBuffer, bytes, 0);
        slot.pending = true;
        slot.count = count;

        readbackSlots[m] = slot;
        latestSnapshotTick[m] = tickId;
    }

    void ProcessReadbacks() {
        if (readbackSlots.Count == 0) return;

        var keys = ListPool<Meshless>.Get();
        foreach (var kv in readbackSlots) keys.Add(kv.Key);

        for (int i = 0; i < keys.Count; i++) {
            var m = keys[i];
            if (m == null) continue;

            if (!readbackSlots.TryGetValue(m, out ReadbackSlot slot))
                continue;

            if (!slot.pending)
                continue;

            if (!slot.request.done)
                continue;

            slot.pending = false;
            readbackSlots[m] = slot;

            if (slot.request.hasError)
                continue;

            var data = slot.request.GetData<float2>();
            if (!latestSnapshot.TryGetValue(m, out float2[] dst) || dst == null || dst.Length != slot.count)
                dst = latestSnapshot[m] = new float2[slot.count];

            data.CopyTo(dst);
        }

        ListPool<Meshless>.Release(keys);
    }

    void UpdateTpsDisplay(float frameDt) {
        float k = 1f - Mathf.Exp(-frameDt * 8f);

        float fpsInst = frameDt > 0f ? (1f / frameDt) : 0f;
        fpsSmoothed = Mathf.Lerp(fpsSmoothed, fpsInst, k);

        if (manual && lastFrameTicks == 0)
            return;

        float inst = frameDt > 0f ? (lastFrameTicks / frameDt) : 0f;
        tpsSmoothed = Mathf.Lerp(tpsSmoothed, inst, k);
    }

    void OnGUI() {
        if (!showTpsOverlay) return;

        bool paused = manual && !Input.GetKey(KeyCode.T) && lastFrameTicks == 0;

        float tickDt = targetTPS > 0f ? (1f / targetTPS) : 0f;
        string text =
            $"FPS: {fpsSmoothed:0}\n" +
            $"TPS: {tpsSmoothed:0}\n" +
            $"Target: {targetTPS:0}\n" +
            $"Ticks/frame: {lastFrameTicks}\n" +
            $"Tick dt: {tickDt:0.000000}\n" +
            $"Sim speed: {simulationSpeed:0.###}";

        GUI.Label(new Rect(tpsOverlayPos.x, tpsOverlayPos.y, 260f, 105f), text);
    }


    static class ListPool<T> {
        static readonly Stack<List<T>> pool = new Stack<List<T>>(16);

        public static List<T> Get() {
            return pool.Count > 0 ? pool.Pop() : new List<T>(64);
        }

        public static void Release(List<T> list) {
            list.Clear();
            pool.Push(list);
        }
    }
}
