using System;
using System.Collections.Generic;
using GPU.Delaunay;
using GPU.Neighbors;
using GPU.Solver;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

[DefaultExecutionOrder(-1000)]
public sealed class SimulationController : MonoBehaviour {
    public enum NeighborSearchMode {
        DelaunayTriangulation = 0,
        UniformGridPaper = 1,
    }

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
    public bool ConvergenceDebugEnabled = true;
    Vector2 tpsOverlayPos = new Vector2(10f, 10f);

    const float holdThreshold = 0.2f;

    [Header("Hierarchy")]
    public bool useHierarchicalSolver = true;

    [Header("Neighbor Search")]
    public NeighborSearchMode neighborSearchMode = NeighborSearchMode.DelaunayTriangulation;

    [Tooltip("Compute shader with kernels from XPBISolver.compute.")]
    public ComputeShader gpuXpbiSolverShader;
    public ComputeShader ColoringShader;
    public ComputeShader delaunayShader;
    public ComputeShader uniformGridNeighborShader;

    [Header("GPU")]
    public ComputeQueueType asyncQueue = ComputeQueueType.Background;
    [Min(1)] public int maxTicksPerBatch = 32;

    [Header("CPU Readback")]
    public bool enableContinuousCpuReadback = true;
    [Min(0.001f)] public float cpuReadbackInterval = 0.02f;

    readonly List<Meshless> meshless = new List<Meshless>(64);
    readonly List<Meshless> activeMeshlessBatch = new List<Meshless>(64);
    readonly List<int> activeMeshlessBaseOffsets = new List<int>(64);
    XPBISolver globalSolver;
    GlobalDTHierarchy globalDTHierarchy;
    UniformGridNeighborSearch uniformGridNeighborSearch;
    bool globalHierarchyDirty = true;

    readonly List<XPBISolver.ForceEvent> gatheredForceEvents = new List<XPBISolver.ForceEvent>(256);
    XPBISolver.ForceEvent[] forceEventUpload = Array.Empty<XPBISolver.ForceEvent>();

    bool globalCpuReadbackPending;
    float globalCpuReadbackLastRequestTime = -999f;
    int cachedGlobalBatchFrame = -1;
    bool cachedGlobalBatchValid;
    float lastAsyncSubmitIssueLogTime = -999f;

    readonly List<Meshless> readbackMeshesSnapshot = new List<Meshless>(64);
    readonly List<int> readbackBaseOffsetsSnapshot = new List<int>(64);
    int[] readbackMappingSnapshot = Array.Empty<int>();
    int readbackMappingCount;
    int[] readbackGlobalToLocal = Array.Empty<int>();
    float2 readbackNormCenter;
    float readbackNormInvHalfExtent;

    // Triple-buffer state (global for all meshless systems)
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
    AsyncTripleState asyncState;

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

        globalSolver?.Dispose();
        globalSolver = null;

        globalDTHierarchy?.Dispose();
        globalDTHierarchy = null;

        uniformGridNeighborSearch?.Dispose();
        uniformGridNeighborSearch = null;

        meshless.Clear();
        activeMeshlessBatch.Clear();
        activeMeshlessBaseOffsets.Clear();
    }

    public void Register(Meshless m) {
        if (m != null && !meshless.Contains(m)) {
            meshless.Add(m);
            globalHierarchyDirty = true;
            cachedGlobalBatchFrame = -1;

            if (globalSolver == null)
                globalSolver = new XPBISolver(gpuXpbiSolverShader, ColoringShader);
            if (globalDTHierarchy == null)
                globalDTHierarchy = new GlobalDTHierarchy(delaunayShader);
            if (uniformGridNeighborSearch == null && uniformGridNeighborShader != null)
                uniformGridNeighborSearch = new UniformGridNeighborSearch(uniformGridNeighborShader, Const.NeighborCount);

            if (asyncState.renderFences == null || asyncState.renderFences.Length != 3) {
                asyncState = new AsyncTripleState {
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
    }

    public void Unregister(Meshless m) {
        if (m != null) {
            meshless.Remove(m);
            globalHierarchyDirty = true;
            cachedGlobalBatchFrame = -1;
        }
    }

    public bool TryGetStableReadSlot(out int slot) {
        slot = 0;
        int renderSlot = asyncState.renderSlot;
        if (renderSlot < 0 || renderSlot > 2)
            return false;

        slot = renderSlot;
        return true;
    }

    public bool TryGetGlobalRenderBatch(out GlobalDTHierarchy hierarchy, out IReadOnlyList<Meshless> meshes, out IReadOnlyList<int> baseOffsets) {
        hierarchy = null;
        meshes = null;
        baseOffsets = null;

        int frame = Time.frameCount;
        if (cachedGlobalBatchFrame != frame) {
            BuildActiveMeshlessBatch();
            if (activeMeshlessBatch.Count == 0) {
                cachedGlobalBatchValid = false;
                cachedGlobalBatchFrame = frame;
                return false;
            }

            EnsureGlobalDTHierarchyBuilt();
            cachedGlobalBatchValid = globalDTHierarchy != null && globalDTHierarchy.MaxLayer >= 0;
            cachedGlobalBatchFrame = frame;
        }

        if (!cachedGlobalBatchValid)
            return false;

        hierarchy = globalDTHierarchy;
        meshes = activeMeshlessBatch;
        baseOffsets = activeMeshlessBaseOffsets;
        return true;
    }

    void Update() {
        if (targetTPS <= 0f)
            return;

        float tickDt = 1f / targetTPS;

        lastFrameTicks = 0;

        // Process completed async batches and try to promote finished slots
        ProcessAsyncBatchCompletions();

        if (manual) ManualUpdateAsync(Time.deltaTime, tickDt);
        else AutoUpdateAsync(Time.deltaTime, tickDt);

        UpdateContinuousCpuReadback();

        renderAlpha = tickDt > 0f ? Mathf.Clamp01(accumulator / tickDt) : 0f;

        UpdateTpsDisplay(Time.deltaTime);
    }

    void LateUpdate() {
        if (!cmdAttached || renderFenceCmd == null || Camera.main == null)
            return;

        renderFenceCmd.Clear();

        int slot = asyncState.renderSlot;
        if (slot < 0 || slot > 2)
            return;

        GraphicsFence fence = renderFenceCmd.CreateGraphicsFence(
            GraphicsFenceType.AsyncQueueSynchronisation,
            SynchronisationStageFlags.PixelProcessing);
        asyncState.renderFences[slot] = fence;
    }

    void AutoUpdateAsync(float frameDt, float tickDt) {
        accumulator += frameDt;

        int ticksToRun = Mathf.Min(maxTicksPerBatch, (int)(accumulator / tickDt));

        if (ticksToRun <= 0) return;

        float dt = tickDt * simulationSpeed;

        if (SubmitGlobalAsyncBatch(dt, ticksToRun)) {
            accumulator -= tickDt * ticksToRun;
            if (accumulator < 0f)
                accumulator = 0f;
            lastFrameTicks += ticksToRun;
        }
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

                int ticksToRun = Mathf.Min(maxTicksPerBatch, (int)(accumulator / tickDt));

                if (ticksToRun > 0) {
                    float dt = tickDt * simulationSpeed;
                    if (SubmitGlobalAsyncBatch(dt, ticksToRun)) {
                        accumulator -= tickDt * ticksToRun;
                        if (accumulator < 0f)
                            accumulator = 0f;
                        lastFrameTicks += ticksToRun;
                    }
                }
            }
        }

        if (Input.GetKeyUp(KeyCode.T)) {
            if (keyHeldTime < holdThreshold) {
                float dt = tickDt * simulationSpeed;
                if (SubmitGlobalAsyncBatch(dt, 1))
                    lastFrameTicks += 1;
            }

            keyHeldTime = 0f;
            accumulator = 0f;
        }
    }

    bool SubmitGlobalAsyncBatch(float dtPerTick, int ticksToRun) {
        if (ticksToRun <= 0)
            return false;

        if (globalSolver == null)
            globalSolver = new XPBISolver(gpuXpbiSolverShader, ColoringShader);

        if (asyncState.renderFences == null || asyncState.renderFences.Length != 3)
            return false;

        if (asyncState.writePending || asyncState.computeCompleted) {
            LogAsyncSubmitIssue("Cannot submit async batch: previous batch still pending.");
            return false;
        }

        if (!TrySelectWritableSlot(ref asyncState, out int freeSlot)) {
            LogAsyncSubmitIssue("Cannot submit async batch: no writable slot available.");
            return false;
        }

        BuildActiveMeshlessBatch();
        if (activeMeshlessBatch.Count == 0)
            return false;

        bool useUniformGridSearch = neighborSearchMode == NeighborSearchMode.UniformGridPaper;
        if (useUniformGridSearch && uniformGridNeighborSearch == null) {
            if (uniformGridNeighborShader == null)
                return false;

            uniformGridNeighborSearch = new UniformGridNeighborSearch(uniformGridNeighborShader, Const.NeighborCount);
        }

        GatherForceEventsForGlobal(activeMeshlessBatch, activeMeshlessBaseOffsets, dtPerTick, ticksToRun);
        if (gatheredForceEvents.Count > 0) {
            EnsureForceUploadCapacity(gatheredForceEvents.Count);
            gatheredForceEvents.CopyTo(forceEventUpload, 0);
            globalSolver.SetGameplayForces(forceEventUpload, gatheredForceEvents.Count);
        } else {
            globalSolver.ClearGameplayForces();
        }

        EnsureGlobalDTHierarchyBuilt();

        float2 neighborBoundsMin = default;
        float2 neighborBoundsMax = default;
        if (useUniformGridSearch) {
            float boundsPadding = 0f;
            if (globalDTHierarchy != null &&
                globalDTHierarchy.TryGetLayerExecutionContext(0, out _, out _, out float layerKernelH))
                boundsPadding = Mathf.Max(1e-5f, Const.WendlandSupport * layerKernelH);

            if (!TryComputeBatchBounds(activeMeshlessBatch, out neighborBoundsMin, out neighborBoundsMax, boundsPadding))
                return false;
        }

        bool useHierarchicalThisBatch = useHierarchicalSolver && !useUniformGridSearch;

        int readSlot = asyncState.renderSlot;
        GraphicsFence computeFence = globalSolver.SubmitSolve(
            activeMeshlessBatch,
            dtPerTick,
            ticksToRun,
            useHierarchicalThisBatch,
            ConvergenceDebugEnabled,
            asyncQueue,
            readSlot,
            freeSlot,
            globalDTHierarchy,
            useUniformGridSearch ? uniformGridNeighborSearch : null,
            neighborBoundsMin,
            neighborBoundsMax
        );

        asyncState.writePending = true;
        asyncState.writeSlot = freeSlot;
        asyncState.computeFence = computeFence;
        asyncState.freeSlot = -1;
        return true;
    }

    static bool IsFencePassedOrUnset(GraphicsFence fence) {
        return fence.Equals(default(GraphicsFence)) || fence.passed;
    }

    static bool IsSlotWritable(in AsyncTripleState state, int slot) {
        if (slot < 0 || slot > 2)
            return false;
        if (slot == state.renderSlot || slot == state.writeSlot || slot == state.completedWriteSlot)
            return false;

        return IsFencePassedOrUnset(state.renderFences[slot]);
    }

    static int NextRingSlot(int slot) {
        if (slot < 0 || slot > 2)
            return -1;

        return (slot + 1) % 3;
    }

    bool TrySelectWritableSlot(ref AsyncTripleState state, out int slot) {
        slot = -1;

        int candidate = NextRingSlot(state.renderSlot);
        if (!IsSlotWritable(state, candidate))
            return false;

        slot = candidate;
        state.freeSlot = candidate;
        return true;
    }

    void BuildActiveMeshlessBatch() {
        activeMeshlessBatch.Clear();
        activeMeshlessBaseOffsets.Clear();

        int baseOffset = 0;
        for (int i = meshless.Count - 1; i >= 0; i--) {
            var m = meshless[i];
            if (m == null || !m.isActiveAndEnabled || m.nodes == null || m.nodes.Count <= 0) {
                meshless.RemoveAt(i);
                globalHierarchyDirty = true;
                cachedGlobalBatchFrame = -1;
                continue;
            }
        }

        for (int i = 0; i < meshless.Count; i++) {
            var m = meshless[i];
            activeMeshlessBatch.Add(m);
            activeMeshlessBaseOffsets.Add(baseOffset);
            baseOffset += m.nodes.Count;
        }
    }

    void EnsureGlobalDTHierarchyBuilt() {
        if (globalDTHierarchy == null)
            globalDTHierarchy = new GlobalDTHierarchy(delaunayShader);
        if (!globalHierarchyDirty)
            return;

        int maxLayerOverride = neighborSearchMode == NeighborSearchMode.UniformGridPaper ? 0 : -1;
        globalDTHierarchy.Rebuild(activeMeshlessBatch, activeMeshlessBaseOffsets, true, maxLayerOverride);
        globalHierarchyDirty = false;
    }

    bool TryComputeBatchBounds(List<Meshless> activeMeshes, out float2 boundsMin, out float2 boundsMax, float padding = 0f) {
        boundsMin = new float2(float.PositiveInfinity, float.PositiveInfinity);
        boundsMax = new float2(float.NegativeInfinity, float.NegativeInfinity);

        bool hasPoint = false;
        for (int meshIdx = 0; meshIdx < activeMeshes.Count; meshIdx++) {
            Meshless m = activeMeshes[meshIdx];
            if (m == null || m.nodes == null || m.nodes.Count == 0)
                continue;

            for (int i = 0; i < m.nodes.Count; i++) {
                float2 p = m.nodes[i].pos;
                boundsMin = math.min(boundsMin, p);
                boundsMax = math.max(boundsMax, p);
                hasPoint = true;
            }
        }

        if (!hasPoint)
            return false;

        float2 ext = boundsMax - boundsMin;
        if (ext.x <= 1e-5f) {
            boundsMin.x -= 0.5f;
            boundsMax.x += 0.5f;
        }
        if (ext.y <= 1e-5f) {
            boundsMin.y -= 0.5f;
            boundsMax.y += 0.5f;
        }

        if (padding > 0f) {
            boundsMin -= new float2(padding, padding);
            boundsMax += new float2(padding, padding);
        }

        return true;
    }

    void LogAsyncSubmitIssue(string message) {
        if (Time.unscaledTime < lastAsyncSubmitIssueLogTime + 1f)
            return;

        lastAsyncSubmitIssueLogTime = Time.unscaledTime;
        Debug.LogWarning(message);
    }

    void GatherForceEventsForGlobal(List<Meshless> activeMeshes, List<int> baseOffsets, float dtPerTick, int ticksToRun) {
        gatheredForceEvents.Clear();

        var controllers = MeshlessForceControllerRegistry.Controllers;
        for (int meshIdx = 0; meshIdx < activeMeshes.Count; meshIdx++) {
            Meshless target = activeMeshes[meshIdx];
            int baseOffset = baseOffsets[meshIdx];

            for (int i = 0; i < controllers.Count; i++) {
                var controller = controllers[i];
                if (controller == null || !controller.IsActive)
                    continue;

                int startIndex = gatheredForceEvents.Count;
                controller.GatherForceEvents(target, dtPerTick, ticksToRun, gatheredForceEvents);

                for (int e = startIndex; e < gatheredForceEvents.Count; e++) {
                    XPBISolver.ForceEvent evt = gatheredForceEvents[e];
                    evt.node += (uint)baseOffset;
                    gatheredForceEvents[e] = evt;
                }
            }
        }
    }

    void EnsureForceUploadCapacity(int count) {
        if (forceEventUpload.Length >= count)
            return;

        int newCap = forceEventUpload.Length > 0 ? forceEventUpload.Length : 64;
        while (newCap < count)
            newCap *= 2;

        forceEventUpload = new XPBISolver.ForceEvent[newCap];
    }

    void ProcessAsyncBatchCompletions() {
        if (asyncState.renderFences == null)
            return;

        if (asyncState.writePending && asyncState.computeFence.passed) {
            asyncState.computeCompleted = true;
            asyncState.completedWriteSlot = asyncState.writeSlot;
            asyncState.writePending = false;
            asyncState.writeSlot = -1;
        }

        if (asyncState.computeCompleted) {
            int candidateSlot = asyncState.completedWriteSlot;
            if (candidateSlot < 0 || candidateSlot > 2)
                return;

            GraphicsFence renderFence = asyncState.renderFences[asyncState.renderSlot];
            if (renderFence.Equals(default(GraphicsFence)) || renderFence.passed) {
                int oldRender = asyncState.renderSlot;
                asyncState.renderSlot = candidateSlot;
                asyncState.freeSlot = oldRender;
                asyncState.computeCompleted = false;
                asyncState.completedWriteSlot = -1;
            }
        }
    }

    void UpdateContinuousCpuReadback() {
        if (!enableContinuousCpuReadback)
            return;

        if (globalCpuReadbackPending)
            return;

        if (Time.time < globalCpuReadbackLastRequestTime + cpuReadbackInterval)
            return;

        if (!TryGetGlobalRenderBatch(out GlobalDTHierarchy hierarchy, out IReadOnlyList<Meshless> meshes, out IReadOnlyList<int> baseOffsets))
            return;

        if (!TryGetStableReadSlot(out int slot))
            return;

        if (!hierarchy.TryGetLayerDt(0, out DT dt) || dt == null)
            return;

        if (!hierarchy.TryGetLayerMappings(0, out _, out int[] globalNodeByLocal, out _, out int activeCount, out _))
            return;
        if (activeCount <= 0)
            return;

        var positions = dt.GetPositionsBuffer(slot);
        if (positions == null)
            return;

        globalCpuReadbackPending = true;
        globalCpuReadbackLastRequestTime = Time.time;

        readbackMeshesSnapshot.Clear();
        readbackBaseOffsetsSnapshot.Clear();
        for (int i = 0; i < meshes.Count; i++)
            readbackMeshesSnapshot.Add(meshes[i]);
        for (int i = 0; i < baseOffsets.Count; i++)
            readbackBaseOffsetsSnapshot.Add(baseOffsets[i]);

        if (readbackMappingSnapshot.Length < globalNodeByLocal.Length)
            readbackMappingSnapshot = new int[globalNodeByLocal.Length];
        Array.Copy(globalNodeByLocal, 0, readbackMappingSnapshot, 0, globalNodeByLocal.Length);
        readbackMappingCount = globalNodeByLocal.Length;

        readbackNormCenter = hierarchy.NormCenter;
        readbackNormInvHalfExtent = hierarchy.NormInvHalfExtent;

        AsyncGPUReadback.Request(positions, OnGlobalCpuReadbackComplete);
    }

    void OnGlobalCpuReadbackComplete(AsyncGPUReadbackRequest request) {
        globalCpuReadbackPending = false;

        if (request.hasError)
            return;

        if (readbackMeshesSnapshot == null || readbackBaseOffsetsSnapshot == null || readbackMappingSnapshot == null)
            return;

        int total = 0;
        for (int i = 0; i < readbackMeshesSnapshot.Count; i++) {
            Meshless m = readbackMeshesSnapshot[i];
            if (m == null || m.nodes == null)
                continue;
            total += m.nodes.Count;
        }
        if (total <= 0)
            return;

        var data = request.GetData<float2>();
        if (data.Length <= 0)
            return;

        int safeTotal = Mathf.Max(1, total);
        if (readbackGlobalToLocal.Length < safeTotal)
            readbackGlobalToLocal = new int[safeTotal];
        for (int i = 0; i < safeTotal; i++)
            readbackGlobalToLocal[i] = -1;

        int mapCount = Mathf.Min(readbackMappingCount, data.Length);
        for (int li = 0; li < mapCount; li++) {
            int gi = readbackMappingSnapshot[li];
            if (gi >= 0 && gi < safeTotal)
                readbackGlobalToLocal[gi] = li;
        }

        float inv = readbackNormInvHalfExtent > 0f ? (1f / readbackNormInvHalfExtent) : 0f;
        if (!(inv > 0f))
            return;

        for (int meshIdx = 0; meshIdx < readbackMeshesSnapshot.Count; meshIdx++) {
            Meshless m = readbackMeshesSnapshot[meshIdx];
            if (m == null || m.nodes == null || !m.isActiveAndEnabled)
                continue;

            int baseOffset = readbackBaseOffsetsSnapshot[meshIdx];
            int count = m.nodes.Count;
            for (int i = 0; i < count; i++) {
                int gi = baseOffset + i;
                if (gi < 0 || gi >= safeTotal)
                    continue;
                int li = readbackGlobalToLocal[gi];
                if (li < 0 || li >= data.Length)
                    continue;

                float2 worldPos = data[li] * inv + readbackNormCenter;
                Node node = m.nodes[i];
                node.pos = worldPos;
                m.nodes[i] = node;
            }
        }
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

}