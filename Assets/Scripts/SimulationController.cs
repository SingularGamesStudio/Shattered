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
    public ComputeShader ColoringShader;
    public ComputeShader delaunayShader;

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

    // Triple‑buffer state per Meshless
    struct AsyncTripleState {
        public int renderSlot;          // slot currently used for rendering (0‑2)
        public int writeSlot;            // slot currently being written to (if any)
        public int freeSlot;             // slot guaranteed not in use (if >=0)
        public GraphicsFence[] renderFences; // per slot, signals when rendering finished using that slot
        public bool writePending;        // true if a compute batch is in progress
        public GraphicsFence computeFence; // fence for the current batch
        public int completedWriteSlot;   // slot whose compute finished but waiting to become render
        public bool computeCompleted;    // true if compute finished but promotion pending
    }
    readonly Dictionary<Meshless, AsyncTripleState> asyncStates = new Dictionary<Meshless, AsyncTripleState>();

    // Persistent command buffer to record render fences
    CommandBuffer renderFenceCmd;
    bool cmdAttached;

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

    void OnEnable() {
        if (Camera.main != null) {
            renderFenceCmd = new CommandBuffer();
            renderFenceCmd.name = "RecordRenderFences";
            Camera.main.AddCommandBuffer(CameraEvent.AfterEverything, renderFenceCmd);
            cmdAttached = true;
        }
    }

    void OnDisable() {
        if (cmdAttached && Camera.main != null && renderFenceCmd != null) {
            Camera.main.RemoveCommandBuffer(CameraEvent.AfterEverything, renderFenceCmd);
            renderFenceCmd.Release();
            renderFenceCmd = null;
            cmdAttached = false;
        }
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

            // Initialise triple‑buffer state
            asyncStates[m] = new AsyncTripleState {
                renderSlot = 0,
                writeSlot = -1,
                freeSlot = 1,
                renderFences = new GraphicsFence[3],
                writePending = false,
                computeCompleted = false,
                completedWriteSlot = -1
            };
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

        // Process completed async batches and try to promote finished slots
        ProcessAsyncBatchCompletions();

        if (manual) ManualUpdateAsync(frameDt, tickDt);
        else AutoUpdateAsync(frameDt, tickDt);

        renderAlpha = tickDt > 0f ? Mathf.Clamp01(accumulator / tickDt) : 0f;

        UpdateTpsDisplay(frameDt);
    }

    void LateUpdate() {
        if (!cmdAttached || renderFenceCmd == null || Camera.main == null)
            return;

        renderFenceCmd.Clear();

        // Record a fence for the current render slot of each active Meshless
        foreach (var m in meshless) {
            if (m == null) continue;
            if (!asyncStates.TryGetValue(m, out var state)) continue;

            int slot = state.renderSlot;
            GraphicsFence fence = renderFenceCmd.CreateGraphicsFence(
                GraphicsFenceType.AsyncQueueSynchronisation,
                SynchronisationStageFlags.PixelProcessing);

            // Update state immediately – the fence handle is valid now.
            state.renderFences[slot] = fence;
            asyncStates[m] = state;
        }

        // No manual execution – the camera will execute this command buffer automatically.
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

    void SubmitMeshlessAsyncBatch(Meshless m, float dtPerTick, int ticksToRun) {
        if (m == null || ticksToRun <= 0)
            return;

        if (!gpuSolverCache.TryGetValue(m, out XPBISolver solver) || solver == null) {
            solver = new XPBISolver(gpuXpbiSolverShader, ColoringShader);
            gpuSolverCache[m] = solver;
        }

        if (!tickCounters.ContainsKey(m)) tickCounters[m] = 0;
        int lastTickId = tickCounters[m];
        int tickIdAfterBatch = lastTickId + ticksToRun;

        if (!asyncStates.TryGetValue(m, out var state))
            return;

        // If we already have a pending write, cannot start another.
        if (state.writePending)
            return;

        // If there's a completed slot waiting to become render, we cannot start a new batch because we have no free slot.
        if (state.computeCompleted)
            return;

        // The free slot should be >=0. Check its render fence.
        int freeSlot = state.freeSlot;
        if (freeSlot < 0 || freeSlot > 2)
            return;

        GraphicsFence freeRenderFence = state.renderFences[freeSlot];
        if (!freeRenderFence.Equals(default(GraphicsFence)) && !freeRenderFence.passed) {
            // The free slot is still being read by the renderer – cannot write to it yet.
            return;
        }

        // Determine whether we need to rebuild adjacency after the batch.
        bool rebuildParents = ((tickIdAfterBatch % Const.HierarchyRebuildInterval) == 0);
        // dtSwapMaxLevel: -1 means no swap, 0 means swap positions only, >0 swaps topology as well.
        int dtSwapMaxLevel = asyncUpdateDtPositions ? (asyncRunDtMaintain ? m.maxLayer : 0) : -1;

        // Submit the batch, writing into the free slot.
        GraphicsFence computeFence = solver.SubmitSolve(
            m,
            dtPerTick,
            ticksToRun,
            useHierarchicalSolver,
            asyncQueue,
            freeSlot
        );

        // Update state
        state.writePending = true;
        state.writeSlot = freeSlot;
        state.computeFence = computeFence;
        state.freeSlot = -1; // temporarily no free slot
        asyncStates[m] = state;

        tickCounters[m] = tickIdAfterBatch;
    }

    void ProcessAsyncBatchCompletions() {
        if (asyncStates.Count == 0) return;

        var keys = ListPool<Meshless>.Get();
        foreach (var kv in asyncStates) keys.Add(kv.Key);

        for (int i = 0; i < keys.Count; i++) {
            var m = keys[i];
            if (m == null) continue;

            if (!asyncStates.TryGetValue(m, out var state))
                continue;

            // First, check if we have a compute batch that just finished
            if (state.writePending && state.computeFence.passed) {
                // Compute on writeSlot is done.
                // Mark it as completed, waiting to become render.
                state.computeCompleted = true;
                state.completedWriteSlot = state.writeSlot;
                state.writePending = false;
                // writeSlot is now pending promotion; keep writeSlot as is for now.
                // freeSlot remains -1 because we haven't freed anything yet.
                asyncStates[m] = state;
            }

            // If we have a completed slot waiting, try to promote it to render.
            if (state.computeCompleted) {
                if (state.computeCompleted) {
                    int candidateSlot = state.completedWriteSlot;
                    GraphicsFence renderFence = state.renderFences[state.renderSlot];
                    if (renderFence.Equals(default(GraphicsFence)) || renderFence.passed) {
                        // Old render slot is free (either never used or rendering finished)
                        int oldRender = state.renderSlot;
                        state.renderSlot = candidateSlot;
                        state.freeSlot = oldRender;
                        state.computeCompleted = false;
                        state.completedWriteSlot = -1;
                        asyncStates[m] = state;
                    }
                    // else: cannot promote yet, wait until next frame.
                }
            }
        }

        ListPool<Meshless>.Release(keys);
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