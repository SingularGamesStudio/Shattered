using System;
using System.Runtime.InteropServices;
using GPU.Delaunay;
using GPU.Solver;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

[DefaultExecutionOrder(1200)]
public sealed class CollisionDebugRenderer : MonoBehaviour {
    public enum ReadbackMode {
        AsyncGpuReadback = 0,
        SyncGetData = 1,
    }

    [Serializable]
    struct ReadbackSnapshot {
        public ComputeBuffer DtPositions;
        public ComputeBuffer GlobalPositions;
        public ComputeBuffer BoundaryChunkCount;
        public ComputeBuffer BoundaryChunks;
        public ComputeBuffer CollisionEventCount;
        public ComputeBuffer CollisionEvents;
        public int ActiveCount;
        public int TotalCount;
        public int BoundaryChunkCapacity;
        public int CollisionEventCapacity;
        public float2 NormCenter;
        public float NormInvHalfExtent;
    }

    sealed class PendingAsyncReadback {
        public int RequestId;
        public int ActiveCount;
        public int TotalCount;
        public float2 NormCenter;
        public float NormInvHalfExtent;

        public bool Failed;

        public bool DtReady;
        public bool GlobalReady;
        public bool BoundaryCountReady;
        public bool BoundaryDataReady;
        public bool CollisionCountReady;
        public bool CollisionDataReady;

        public int BoundaryCount;
        public int CollisionCount;

        public float2[] DtPositions = Array.Empty<float2>();
        public float2[] GlobalPositions = Array.Empty<float2>();
        public BoundaryChunkGpu[] Boundaries = Array.Empty<BoundaryChunkGpu>();
        public CollisionEventGpu[] Collisions = Array.Empty<CollisionEventGpu>();
    }

    [StructLayout(LayoutKind.Sequential)]
    struct BoundaryChunkGpu {
        public uint aLi;
        public uint bLi;
        public int owner;
        public uint pad;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct CollisionEventGpu {
        public uint aGi;
        public uint bGi;
        public uint qaGi;
        public uint qbGi;
        public uint oaGi;
        public uint obGi;
        public float4 nPen;
        public float2 segST;
        public float2 pad;
    }

    [Header("Readback")]
    public bool enableOverlay = true;
    public ReadbackMode readbackMode = ReadbackMode.SyncGetData;
    [Min(0.005f)] public float readbackInterval = 0.05f;

    [Header("Boundary")]
    public bool drawBoundaries = true;
    public Color boundaryColor = new Color(0.2f, 1f, 0.35f, 1f);
    [Min(0)] public int maxBoundarySegments = 6000;

    [Header("Collision Arrows")]
    public bool drawCollisionArrows = true;
    public Color collisionArrowColor = new Color(1f, 0.35f, 0.15f, 1f);
    [Min(0)] public int maxCollisionArrows = 2000;
    [Min(0.01f)] public float arrowScale = 1f;
    [Min(0f)] public float minArrowLength = 0.015f;
    [Min(0.002f)] public float arrowHeadLength = 0.02f;
    [Range(0.1f, 1f)] public float arrowHeadWidthScale = 0.55f;

    [Header("Overlay")]
    [Min(1f)] public float lineThickness = 2f;

    [Header("Camera")]
    [Tooltip("Optional explicit camera used for projection. Leave null to auto-resolve.")]
    public Camera targetCamera;
    [Tooltip("Project debug XY points onto a plane in front of the camera for stable screen mapping.")]
    public bool projectOnCameraFrontPlane = true;
    [Min(0.01f)] public float cameraFrontPlaneDistance = 1f;
    public float zDepth = 0f;

    float nextReadbackTime;

    int lastActiveCount;
    int lastTotalCount;
    int lastBoundaryCount;
    int lastCollisionCount;
    float2 lastNormCenter;
    float lastNormInvHalfExtent;

    float2[] dtPositions = Array.Empty<float2>();
    float2[] globalPositions = Array.Empty<float2>();
    BoundaryChunkGpu[] boundaryChunks = Array.Empty<BoundaryChunkGpu>();
    CollisionEventGpu[] collisionEvents = Array.Empty<CollisionEventGpu>();

    readonly uint[] oneUintScratch = new uint[1];

    int asyncRequestId;
    PendingAsyncReadback pending;

    bool lastDtLooksNormalized = true;
    bool lastGlobalLooksNormalized;

    void OnDisable() {
        pending = null;
    }

    void Update() {
        if (!enableOverlay)
            return;

        if (Time.time < nextReadbackTime)
            return;

        nextReadbackTime = Time.time + Mathf.Max(0.005f, readbackInterval);

        if (!TryBuildSnapshot(out ReadbackSnapshot snapshot))
            return;

        ReadbackMode mode = readbackMode;
        if (mode == ReadbackMode.AsyncGpuReadback && !SystemInfo.supportsAsyncGPUReadback)
            mode = ReadbackMode.SyncGetData;

        if (mode == ReadbackMode.SyncGetData) {
            RunSyncReadback(snapshot);
            return;
        }

        if (pending != null)
            return;

        StartAsyncReadback(snapshot);
    }

    void OnGUI() {
        if (!enableOverlay)
            return;

        if (Event.current != null && Event.current.type != EventType.Repaint)
            return;

        Camera cam = ResolveDebugCamera();
        if (cam == null)
            return;

        DrawOverlayInGui(cam);
    }

    bool TryBuildSnapshot(out ReadbackSnapshot snapshot) {
        snapshot = default;

        SimulationController sim = SimulationController.Instance;
        if (sim == null)
            return false;

        XPBISolver solver = sim.GlobalSolver;
        if (solver == null || solver.collisionEvent == null)
            return false;

        if (!sim.TryGetGlobalRenderBatch(out DTHierarchy hierarchy, out var meshes, out _))
            return false;

        if (!sim.TryGetStableReadSlot(out int stableSlot))
            return false;

        if (!hierarchy.TryGetLayerDt(0, out DT dt) || dt == null)
            return false;

        if (!hierarchy.TryGetLayerMappings(0, out _, out _, out _, out int activeCount, out _))
            return false;
        if (activeCount <= 0)
            return false;

        ComputeBuffer dtPositionsBuffer = dt.GetPositionsBuffer(stableSlot);
        if (dtPositionsBuffer == null)
            return false;

        ComputeBuffer globalPositionsBuffer = solver.pos;
        if (globalPositionsBuffer == null)
            return false;

        ComputeBuffer boundaryCountBuffer = solver.collisionEvent.BoundaryChunkCountBuffer;
        ComputeBuffer boundaryBuffer = solver.collisionEvent.BoundaryChunksBuffer;
        ComputeBuffer collisionCountBuffer = solver.collisionEvent.CollisionEventCountBuffer;
        ComputeBuffer collisionBuffer = solver.collisionEvent.CollisionEventsBuffer;

        if (boundaryCountBuffer == null || boundaryBuffer == null || collisionCountBuffer == null || collisionBuffer == null)
            return false;

        int totalCount = 0;
        if (meshes != null) {
            for (int i = 0; i < meshes.Count; i++) {
                Meshless m = meshes[i];
                if (m == null || m.nodes == null)
                    continue;
                totalCount += m.nodes.Count;
            }
        }

        if (totalCount <= 0)
            return false;

        snapshot = new ReadbackSnapshot {
            DtPositions = dtPositionsBuffer,
            GlobalPositions = globalPositionsBuffer,
            BoundaryChunkCount = boundaryCountBuffer,
            BoundaryChunks = boundaryBuffer,
            CollisionEventCount = collisionCountBuffer,
            CollisionEvents = collisionBuffer,
            ActiveCount = Mathf.Min(activeCount, dtPositionsBuffer.count),
            TotalCount = Mathf.Min(totalCount, globalPositionsBuffer.count),
            BoundaryChunkCapacity = boundaryBuffer.count,
            CollisionEventCapacity = collisionBuffer.count,
            NormCenter = hierarchy.NormCenter,
            NormInvHalfExtent = hierarchy.NormInvHalfExtent,
        };

        return snapshot.ActiveCount > 0 && snapshot.TotalCount > 0;
    }

    void RunSyncReadback(in ReadbackSnapshot snapshot) {
        oneUintScratch[0] = 0;
        snapshot.BoundaryChunkCount.GetData(oneUintScratch, 0, 0, 1);
        int boundaryCount = Mathf.Clamp((int)oneUintScratch[0], 0, snapshot.BoundaryChunkCapacity);

        oneUintScratch[0] = 0;
        snapshot.CollisionEventCount.GetData(oneUintScratch, 0, 0, 1);
        int collisionCount = Mathf.Clamp((int)oneUintScratch[0], 0, snapshot.CollisionEventCapacity);

        EnsureCapacity(ref dtPositions, snapshot.ActiveCount);
        snapshot.DtPositions.GetData(dtPositions, 0, 0, snapshot.ActiveCount);

        EnsureCapacity(ref globalPositions, snapshot.TotalCount);
        snapshot.GlobalPositions.GetData(globalPositions, 0, 0, snapshot.TotalCount);

        EnsureCapacity(ref boundaryChunks, boundaryCount);
        if (boundaryCount > 0)
            snapshot.BoundaryChunks.GetData(boundaryChunks, 0, 0, boundaryCount);

        EnsureCapacity(ref collisionEvents, collisionCount);
        if (collisionCount > 0)
            snapshot.CollisionEvents.GetData(collisionEvents, 0, 0, collisionCount);

        lastActiveCount = snapshot.ActiveCount;
        lastTotalCount = snapshot.TotalCount;
        lastBoundaryCount = boundaryCount;
        lastCollisionCount = collisionCount;
        lastNormCenter = snapshot.NormCenter;
        lastNormInvHalfExtent = snapshot.NormInvHalfExtent;
    }

    void StartAsyncReadback(in ReadbackSnapshot snapshot) {
        PendingAsyncReadback state = new PendingAsyncReadback {
            RequestId = ++asyncRequestId,
            ActiveCount = snapshot.ActiveCount,
            TotalCount = snapshot.TotalCount,
            NormCenter = snapshot.NormCenter,
            NormInvHalfExtent = snapshot.NormInvHalfExtent,
        };

        pending = state;

        int dtBytes = snapshot.ActiveCount * sizeof(float) * 2;
        AsyncGPUReadback.Request(snapshot.DtPositions, dtBytes, 0, req => OnAsyncDtPositions(req, state.RequestId));

        int globalBytes = snapshot.TotalCount * sizeof(float) * 2;
        AsyncGPUReadback.Request(snapshot.GlobalPositions, globalBytes, 0, req => OnAsyncGlobalPositions(req, state.RequestId));

        ComputeBuffer boundaryChunksBuffer = snapshot.BoundaryChunks;
        int boundaryChunkCapacity = snapshot.BoundaryChunkCapacity;
        ComputeBuffer collisionEventsBuffer = snapshot.CollisionEvents;
        int collisionEventCapacity = snapshot.CollisionEventCapacity;

        AsyncGPUReadback.Request(snapshot.BoundaryChunkCount, req => OnAsyncBoundaryCount(req, state.RequestId, boundaryChunksBuffer, boundaryChunkCapacity));
        AsyncGPUReadback.Request(snapshot.CollisionEventCount, req => OnAsyncCollisionCount(req, state.RequestId, collisionEventsBuffer, collisionEventCapacity));
    }

    void OnAsyncDtPositions(AsyncGPUReadbackRequest request, int requestId) {
        if (!TryGetPending(requestId, out PendingAsyncReadback state))
            return;

        if (request.hasError) {
            state.Failed = true;
            TryFinalizeAsync(state);
            return;
        }

        var data = request.GetData<float2>();
        state.DtPositions = new float2[data.Length];
        data.CopyTo(state.DtPositions);
        state.DtReady = true;

        TryFinalizeAsync(state);
    }

    void OnAsyncGlobalPositions(AsyncGPUReadbackRequest request, int requestId) {
        if (!TryGetPending(requestId, out PendingAsyncReadback state))
            return;

        if (request.hasError) {
            state.Failed = true;
            TryFinalizeAsync(state);
            return;
        }

        var data = request.GetData<float2>();
        state.GlobalPositions = new float2[data.Length];
        data.CopyTo(state.GlobalPositions);
        state.GlobalReady = true;

        TryFinalizeAsync(state);
    }

    void OnAsyncBoundaryCount(AsyncGPUReadbackRequest request, int requestId, ComputeBuffer boundaryBuffer, int boundaryCapacity) {
        if (!TryGetPending(requestId, out PendingAsyncReadback state))
            return;

        if (request.hasError) {
            state.Failed = true;
            TryFinalizeAsync(state);
            return;
        }

        var data = request.GetData<uint>();
        int count = (data.Length > 0) ? Mathf.Clamp((int)data[0], 0, boundaryCapacity) : 0;

        state.BoundaryCount = count;
        state.BoundaryCountReady = true;

        if (count <= 0) {
            state.Boundaries = Array.Empty<BoundaryChunkGpu>();
            state.BoundaryDataReady = true;
            TryFinalizeAsync(state);
            return;
        }

        int bytes = count * (sizeof(uint) + sizeof(uint) + sizeof(int) + sizeof(uint));
        AsyncGPUReadback.Request(boundaryBuffer, bytes, 0, req => OnAsyncBoundaryData(req, requestId));
    }

    void OnAsyncBoundaryData(AsyncGPUReadbackRequest request, int requestId) {
        if (!TryGetPending(requestId, out PendingAsyncReadback state))
            return;

        if (request.hasError) {
            state.Failed = true;
            TryFinalizeAsync(state);
            return;
        }

        var data = request.GetData<BoundaryChunkGpu>();
        state.Boundaries = new BoundaryChunkGpu[data.Length];
        data.CopyTo(state.Boundaries);
        state.BoundaryDataReady = true;

        TryFinalizeAsync(state);
    }

    void OnAsyncCollisionCount(AsyncGPUReadbackRequest request, int requestId, ComputeBuffer collisionBuffer, int collisionCapacity) {
        if (!TryGetPending(requestId, out PendingAsyncReadback state))
            return;

        if (request.hasError) {
            state.Failed = true;
            TryFinalizeAsync(state);
            return;
        }

        var data = request.GetData<uint>();
        int count = (data.Length > 0) ? Mathf.Clamp((int)data[0], 0, collisionCapacity) : 0;

        state.CollisionCount = count;
        state.CollisionCountReady = true;

        if (count <= 0) {
            state.Collisions = Array.Empty<CollisionEventGpu>();
            state.CollisionDataReady = true;
            TryFinalizeAsync(state);
            return;
        }

        int bytes = count * Marshal.SizeOf<CollisionEventGpu>();
        AsyncGPUReadback.Request(collisionBuffer, bytes, 0, req => OnAsyncCollisionData(req, requestId));
    }

    void OnAsyncCollisionData(AsyncGPUReadbackRequest request, int requestId) {
        if (!TryGetPending(requestId, out PendingAsyncReadback state))
            return;

        if (request.hasError) {
            state.Failed = true;
            TryFinalizeAsync(state);
            return;
        }

        var data = request.GetData<CollisionEventGpu>();
        state.Collisions = new CollisionEventGpu[data.Length];
        data.CopyTo(state.Collisions);
        state.CollisionDataReady = true;

        TryFinalizeAsync(state);
    }

    bool TryGetPending(int requestId, out PendingAsyncReadback state) {
        state = pending;
        if (state == null)
            return false;
        if (state.RequestId != requestId)
            return false;
        return true;
    }

    void TryFinalizeAsync(PendingAsyncReadback state) {
        if (pending == null || pending.RequestId != state.RequestId)
            return;

        if (state.Failed) {
            pending = null;
            return;
        }

        if (!state.DtReady || !state.GlobalReady || !state.BoundaryCountReady || !state.BoundaryDataReady || !state.CollisionCountReady || !state.CollisionDataReady)
            return;

        dtPositions = state.DtPositions;
        globalPositions = state.GlobalPositions;
        boundaryChunks = state.Boundaries;
        collisionEvents = state.Collisions;

        lastActiveCount = Mathf.Min(state.ActiveCount, dtPositions.Length);
        lastTotalCount = Mathf.Min(state.TotalCount, globalPositions.Length);
        lastBoundaryCount = Mathf.Min(state.BoundaryCount, boundaryChunks.Length);
        lastCollisionCount = Mathf.Min(state.CollisionCount, collisionEvents.Length);
        lastNormCenter = state.NormCenter;
        lastNormInvHalfExtent = state.NormInvHalfExtent;

        pending = null;
    }

    void DrawOverlayInGui(Camera cam) {
        float invHalfExtent = lastNormInvHalfExtent > 0f ? (1f / lastNormInvHalfExtent) : 0f;
        lastDtLooksNormalized = LooksNormalized(dtPositions, lastActiveCount);
        lastGlobalLooksNormalized = LooksNormalized(globalPositions, lastTotalCount);

        float thickness = Mathf.Max(1f, lineThickness);

        if (drawBoundaries && lastBoundaryCount > 0 && lastActiveCount > 0 && dtPositions != null) {
            int drawCount = maxBoundarySegments > 0 ? Mathf.Min(lastBoundaryCount, maxBoundarySegments) : lastBoundaryCount;
            for (int i = 0; i < drawCount; i++) {
                BoundaryChunkGpu edge = boundaryChunks[i];
                int a = (int)edge.aLi;
                int b = (int)edge.bLi;
                if (a < 0 || b < 0 || a >= lastActiveCount || b >= lastActiveCount)
                    continue;

                float2 wa = ToWorld(dtPositions[a], lastDtLooksNormalized, invHalfExtent, lastNormCenter);
                float2 wb = ToWorld(dtPositions[b], lastDtLooksNormalized, invHalfExtent, lastNormCenter);
                if (!TryWorldToGuiPoint(cam, wa, out Vector2 ga) || !TryWorldToGuiPoint(cam, wb, out Vector2 gb))
                    continue;

                DrawGuiLine(ga, gb, boundaryColor, thickness);
            }
        }

        if (drawCollisionArrows && lastCollisionCount > 0 && lastTotalCount > 0 && globalPositions != null) {
            int drawCount = maxCollisionArrows > 0 ? Mathf.Min(lastCollisionCount, maxCollisionArrows) : lastCollisionCount;
            for (int i = 0; i < drawCount; i++) {
                CollisionEventGpu evt = collisionEvents[i];
                int a = (int)evt.aGi;
                int b = (int)evt.bGi;
                if (a < 0 || b < 0 || a >= lastTotalCount || b >= lastTotalCount)
                    continue;

                float2 pa = ToWorld(globalPositions[a], lastGlobalLooksNormalized, invHalfExtent, lastNormCenter);
                float2 pb = ToWorld(globalPositions[b], lastGlobalLooksNormalized, invHalfExtent, lastNormCenter);
                float2 mid = 0.5f * (pa + pb);

                float2 n = evt.nPen.xy;
                float nLen = math.length(n);
                if (!(nLen > 1e-6f))
                    continue;
                n /= nLen;

                float shaftLength = math.max(minArrowLength, evt.nPen.z * arrowScale);
                float2 tip = mid + n * shaftLength;

                float headLen = math.max(0.001f, arrowHeadLength);
                float headWidth = headLen * math.clamp(arrowHeadWidthScale, 0.1f, 1f);
                float2 side = new float2(-n.y, n.x);
                float2 back = tip - n * headLen;
                float2 left = back + side * headWidth;
                float2 right = back - side * headWidth;

                if (!TryWorldToGuiPoint(cam, mid, out Vector2 gMid) ||
                    !TryWorldToGuiPoint(cam, tip, out Vector2 gTip) ||
                    !TryWorldToGuiPoint(cam, left, out Vector2 gLeft) ||
                    !TryWorldToGuiPoint(cam, right, out Vector2 gRight))
                    continue;

                DrawGuiLine(gMid, gTip, collisionArrowColor, thickness);
                DrawGuiLine(gTip, gLeft, collisionArrowColor, thickness);
                DrawGuiLine(gTip, gRight, collisionArrowColor, thickness);
            }
        }
    }

    Camera ResolveDebugCamera() {
        if (targetCamera != null && targetCamera.isActiveAndEnabled)
            return targetCamera;

        Camera own = GetComponent<Camera>();
        if (own != null && own.isActiveAndEnabled)
            return own;

        global::Renderer mainRenderer = FindFirstObjectByType<global::Renderer>();
        if (mainRenderer != null) {
            Camera rendererCamera = mainRenderer.GetComponent<Camera>();
            if (rendererCamera != null && rendererCamera.isActiveAndEnabled)
                return rendererCamera;
        }

        if (Camera.main != null && Camera.main.isActiveAndEnabled)
            return Camera.main;

        if (Camera.current != null && Camera.current.isActiveAndEnabled)
            return Camera.current;

        Camera[] cams = Camera.allCameras;
        for (int i = 0; i < cams.Length; i++) {
            Camera c = cams[i];
            if (c != null && c.isActiveAndEnabled)
                return c;
        }

        return null;
    }

    bool TryWorldToGuiPoint(Camera cam, float2 world, out Vector2 guiPoint) {
        Vector3 screen = cam.WorldToScreenPoint(new Vector3(world.x, world.y, ResolveProjectionZ(cam)));
        guiPoint = new Vector2(screen.x, Screen.height - screen.y);
        return true;
    }

    float ResolveProjectionZ(Camera cam) {
        if (!projectOnCameraFrontPlane || cam == null)
            return zDepth;

        float dist = Mathf.Max(0.01f, cameraFrontPlaneDistance);
        return cam.transform.position.z + cam.transform.forward.z * dist;
    }

    static void DrawGuiLine(Vector2 a, Vector2 b, Color color, float thickness) {
        float length = Vector2.Distance(a, b);
        if (!(length > 0.01f))
            return;

        Matrix4x4 prev = GUI.matrix;
        Color prevColor = GUI.color;

        float angle = Mathf.Atan2(b.y - a.y, b.x - a.x) * Mathf.Rad2Deg;
        GUI.color = color;
        GUIUtility.RotateAroundPivot(angle, a);
        GUI.DrawTexture(new Rect(a.x, a.y - thickness * 0.5f, length, thickness), Texture2D.whiteTexture);

        GUI.matrix = prev;
        GUI.color = prevColor;
    }

    static bool LooksNormalized(float2[] data, int count) {
        if (data == null || count <= 0)
            return false;

        int sampleCount = Mathf.Min(count, 64);
        if (sampleCount <= 0)
            return false;

        float maxAbs = 0f;
        int step = Mathf.Max(1, count / sampleCount);
        int visited = 0;
        for (int i = 0; i < count && visited < sampleCount; i += step) {
            float2 p = data[i];
            maxAbs = Mathf.Max(maxAbs, Mathf.Abs(p.x), Mathf.Abs(p.y));
            visited++;
        }

        return maxAbs <= 2.5f;
    }

    static float2 ToWorld(float2 point, bool normalized, float invHalfExtent, float2 center) {
        if (!normalized)
            return point;

        if (!(invHalfExtent > 0f))
            return point;

        return point * invHalfExtent + center;
    }

    static void EnsureCapacity<T>(ref T[] array, int count) {
        if (count <= 0) {
            if (array == null)
                array = Array.Empty<T>();
            return;
        }

        if (array == null || array.Length < count)
            array = new T[count];
    }
}
