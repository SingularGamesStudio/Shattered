using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using GPU.Delaunay;
using GPU.Solver;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

[DefaultExecutionOrder(1200)]
public sealed class CollisionDebugRenderer : MonoBehaviour {
    const int CollisionDebugStatCount = 64;
    const uint InvalidU32 = 0xFFFFFFFFu;
    const int StatBoundaryFlags = 0;
    const int StatBoundaryEdges = 1;
    const int StatOwnerEdgeOverflow = 2;
    const int StatMaxEdgeBinLoad = 3;
    const int StatMaxVertBinLoad = 4;
    const int StatEdgeBinOverflows = 5;
    const int StatVertBinOverflows = 6;
    const int StatVertexCandidates = 7;
    const int StatVertexFeatureHits = 8;
    const int StatVertexWithinSupport = 9;
    const int StatVertexContactsWritten = 10;
    const int StatEdgePairCandidates = 11;
    const int StatEdgeWithinSupport = 12;
    const int StatEdgeContactsEmitted = 13;
    const int StatOwnerAabbRejects = 14;
    const int StatVertexStageOverflow = 15;
    const int StatBuildHalfedgeThreads = 16;
    const int StatBuildRejectNonBoundaryFlag = 17;
    const int StatBuildRejectInvalidFace = 18;
    const int StatBuildRejectNotInternalBoundary = 19;
    const int StatBuildRejectInvalidGlobalVertex = 20;
    const int StatBuildRejectInvalidOwner = 21;
    const int StatBuildOwnerEdgeRefWrites = 22;
    const int StatBuildOwnerEdgeRefOverflow = 23;
    const int StatVertexWorkItems = 24;
    const int StatVertexRejectPairOob = 25;
    const int StatVertexRejectSameOwner = 26;
    const int StatVertexRejectOwnerOverflow = 27;
    const int StatVertexRejectNoOwnerEdgeRef = 28;
    const int StatVertexRejectInvalidVgi = 29;
    const int StatVertexRejectNoBoundaryHit = 30;
    const int StatVertexRejectSupport = 31;
    const int StatVertexRejectDegenerateNormal = 32;
    const int StatVertexCompactRejectInvalid = 33;
    const int StatVertexCompactValid = 34;
    const int StatVertexRejectAabb = 35;
    const int StatEdgeWorkItems = 36;
    const int StatEdgeRejectPairOob = 37;
    const int StatEdgeRejectSwap = 38;
    const int StatEdgeRejectSameOwner = 39;
    const int StatEdgeRejectOwnerOverflow = 40;
    const int StatEdgeRejectNoOwnerEdgeRef = 41;
    const int StatEdgeRejectInvalidEdgeA = 42;
    const int StatEdgeBinCellsVisited = 43;
    const int StatEdgeBinEdgeRefsScanned = 44;
    const int StatEdgeRejectOwnerMismatchB = 45;
    const int StatEdgeRejectInvalidEdgeB = 46;
    const int StatEdgeRejectEdgeBbox = 47;
    const int StatEdgeRejectNoIntersection = 48;
    const int StatEdgePointIntersections = 49;
    const int StatEdgeRejectNonPointHit = 50;
    const int StatEdgeRejectEndpointIntersection = 51;
    const int StatEdgeRejectNonCanonicalBin = 52;
    const int StatEdgeRejectDegenerateNormal = 53;
    const int StatEdgeRejectNoPenetration = 54;
    const int StatEdgeRejectAabb = 55;
    const int StatBuildRejectInvalidLocalVertex = 56;
    const int StatBuildRejectDegenerateEndpoints = 57;

    public enum ReadbackMode {
        AsyncGpuReadback = 0,
        SyncGetData = 1,
    }

    [Serializable]
    struct ReadbackSnapshot {
        public ComputeBuffer DtPositions;
        public ComputeBuffer GlobalPositions;
        public ComputeBuffer InvMass;
        public ComputeBuffer BoundaryChunkCount;
        public ComputeBuffer BoundaryEdgeOwner;
        public ComputeBuffer BoundaryEdgeV0;
        public ComputeBuffer BoundaryEdgeV1;
        public ComputeBuffer BoundaryEdgeNormal;
        public ComputeBuffer BoundaryEdgeP0;
        public ComputeBuffer BoundaryEdgeP1;
        public ComputeBuffer CollisionEventCount;
        public ComputeBuffer CollisionEvents;
        public ComputeBuffer CoarseContactCount;
        public ComputeBuffer CoarseContacts;
        public int ActiveCount;
        public int TotalCount;
        public int BoundaryChunkCapacity;
        public int CollisionEventCapacity;
        public int CoarseContactCapacity;
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
        public bool InvMassReady;
        public bool BoundaryCountReady;
        public bool BoundaryOwnerReady;
        public bool BoundaryV0Ready;
        public bool BoundaryV1Ready;
        public bool BoundaryNormalReady;
        public bool BoundaryP0Ready;
        public bool BoundaryP1Ready;
        public bool CollisionCountReady;
        public bool CollisionDataReady;
        public bool CoarseCollisionCountReady;
        public bool CoarseCollisionDataReady;

        public int BoundaryCount;
        public int BoundaryRawCount;
        public int BoundaryReadCount;
        public int BoundaryOwnerValidCount;
        public int BoundarySegmentValidCount;
        public int CollisionCount;
        public int CoarseCollisionCount;

        public float2[] DtPositions = Array.Empty<float2>();
        public float2[] GlobalPositions = Array.Empty<float2>();
        public float[] InvMass = Array.Empty<float>();
        public BoundaryChunkGpu[] Boundaries = Array.Empty<BoundaryChunkGpu>();
        public CollisionEventGpu[] Collisions = Array.Empty<CollisionEventGpu>();
        public CoarseContactGpu[] CoarseCollisions = Array.Empty<CoarseContactGpu>();
    }

    [StructLayout(LayoutKind.Sequential)]
    struct BoundaryChunkGpu {
        public int va;
        public int vb;
        public int ownerID;
        public float2 outNormal;
        public float2 p0;
        public float2 p1;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct CollisionEventGpu {
        public uint ownerA;
        public uint ownerB;
        public uint nodeGi0;
        public uint nodeGi1;
        public uint nodeGi2;
        public uint nodeGi3;
        public float beta0;
        public float beta1;
        public float beta2;
        public float beta3;
        public float2 normal;
        public float penetration;
    }

    [StructLayout(LayoutKind.Sequential)]
    struct CoarseContactGpu {
        public uint ownerA;
        public uint ownerB;
        public uint coarseGiA;
        public uint coarseGiB;
        public float2 normal;
        public float penetration;
        public float weight;
        public float2 point;
    }

    [Header("Readback")]
    public bool enableOverlay = true;
    public ReadbackMode readbackMode = ReadbackMode.SyncGetData;
    [Min(0.005f)] public float readbackInterval = 0.05f;

    [Header("Boundary")]
    public bool drawBoundaries = true;
    public Color boundaryColor = new Color(0.2f, 1f, 0.35f, 1f);
    [Min(0)] public int maxBoundarySegments = 6000;
    [Min(0)] public int fallbackBoundaryScanLimit = 0;
    public bool drawBoundaryNormals = true;
    public Color boundaryNormalColor = new Color(0.12f, 0.85f, 1f, 1f);
    [Min(0.001f)] public float boundaryNormalLength = 0.02f;

    [Header("Collision Arrows")]
    public bool drawCollisionArrows = true;
    public Color collisionArrowColor = new Color(1f, 0.35f, 0.15f, 1f);
    public bool drawCoarseCollisionArrows = true;
    public Color coarseCollisionArrowColor = new Color(0.1f, 0.75f, 1f, 1f);
    public bool colorArrowsByOwnerPair = true;
    [Min(0)] public int maxCollisionArrows = 7000;
    [Min(0.01f)] public float arrowScale = 1f;
    [Min(0f)] public float minArrowLength = 0.015f;
    [Min(0.002f)] public float arrowHeadLength = 0.02f;
    [Range(0.1f, 1f)] public float arrowHeadWidthScale = 0.55f;
    [Min(0f)] public float overlapFanoutWorld = 0.002f;
    public bool highlightEdgeIntersections = true;
    public Color intersectionColor = new Color(1f, 0.95f, 0.2f, 1f);
    [Min(1f)] public float intersectionMarkerPixels = 8f;

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
    int lastCoarseCollisionCount;
    float2 lastNormCenter;
    float lastNormInvHalfExtent;

    float2[] dtPositions = Array.Empty<float2>();
    float2[] globalPositions = Array.Empty<float2>();
    float[] invMass = Array.Empty<float>();
    BoundaryChunkGpu[] boundaryChunks = Array.Empty<BoundaryChunkGpu>();
    CollisionEventGpu[] collisionEvents = Array.Empty<CollisionEventGpu>();
    CoarseContactGpu[] coarseCollisionEvents = Array.Empty<CoarseContactGpu>();

    readonly uint[] oneUintScratch = new uint[1];

    int asyncRequestId;
    PendingAsyncReadback pending;

    bool lastDtLooksNormalized = true;
    bool lastGlobalLooksNormalized;
    int lastArrowAttempted;
    int lastArrowDrawn;
    int lastArrowSkippedDegenerate;
    int lastArrowSkippedProjection;
    int lastArrowClipped;
    int lastArrowOverlapInstances;
    int lastIntersectionMarkers;
    readonly Dictionary<int, int> arrowOverlapCounts = new Dictionary<int, int>(2048);

    int GetBoundaryReadCount(int requestedCount, int capacity) {
        int boundedCapacity = Mathf.Max(0, capacity);
        if (boundedCapacity == 0)
            return 0;

        int count = Mathf.Clamp(requestedCount, 0, boundedCapacity);
        if (count > 0)
            return count;

        int fallback = fallbackBoundaryScanLimit > 0
            ? Mathf.Clamp(fallbackBoundaryScanLimit, 1, boundedCapacity)
            : boundedCapacity;
        return fallback;
    }

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
        ComputeBuffer invMassBuffer = solver.invMass;
        if (globalPositionsBuffer == null || invMassBuffer == null)
            return false;

        ComputeBuffer boundaryCountBuffer = solver.collisionEvent.BoundaryChunkCountBuffer;
        ComputeBuffer boundaryOwnerBuffer = solver.collisionEvent.BoundaryEdgeOwnerBuffer;
        ComputeBuffer boundaryV0Buffer = solver.collisionEvent.BoundaryEdgeV0Buffer;
        ComputeBuffer boundaryV1Buffer = solver.collisionEvent.BoundaryEdgeV1Buffer;
        ComputeBuffer boundaryNormalBuffer = solver.collisionEvent.BoundaryEdgeNormalBuffer;
        ComputeBuffer boundaryP0Buffer = solver.collisionEvent.BoundaryEdgeP0Buffer;
        ComputeBuffer boundaryP1Buffer = solver.collisionEvent.BoundaryEdgeP1Buffer;
        ComputeBuffer collisionCountBuffer = solver.collisionEvent.CollisionEventCountBuffer;
        ComputeBuffer collisionBuffer = solver.collisionEvent.CollisionEventsBuffer;
        ComputeBuffer coarseCollisionCountBuffer = solver.collisionEvent.CoarseContactCountBuffer;
        ComputeBuffer coarseCollisionBuffer = solver.collisionEvent.CoarseContactsBuffer;
        if (boundaryCountBuffer == null || boundaryOwnerBuffer == null || boundaryV0Buffer == null || boundaryV1Buffer == null || boundaryNormalBuffer == null || boundaryP0Buffer == null || boundaryP1Buffer == null || collisionCountBuffer == null || collisionBuffer == null || coarseCollisionCountBuffer == null || coarseCollisionBuffer == null)
            return false;

        int totalCount = math.min(globalPositionsBuffer.count, invMassBuffer.count);

        if (totalCount <= 0)
            return false;

        snapshot = new ReadbackSnapshot {
            DtPositions = dtPositionsBuffer,
            GlobalPositions = globalPositionsBuffer,
            InvMass = invMassBuffer,
            BoundaryChunkCount = boundaryCountBuffer,
            BoundaryEdgeOwner = boundaryOwnerBuffer,
            BoundaryEdgeV0 = boundaryV0Buffer,
            BoundaryEdgeV1 = boundaryV1Buffer,
            BoundaryEdgeNormal = boundaryNormalBuffer,
            BoundaryEdgeP0 = boundaryP0Buffer,
            BoundaryEdgeP1 = boundaryP1Buffer,
            CollisionEventCount = collisionCountBuffer,
            CollisionEvents = collisionBuffer,
            CoarseContactCount = coarseCollisionCountBuffer,
            CoarseContacts = coarseCollisionBuffer,
            ActiveCount = Mathf.Min(activeCount, dtPositionsBuffer.count),
            TotalCount = Mathf.Min(totalCount, globalPositionsBuffer.count),
            BoundaryChunkCapacity = boundaryOwnerBuffer.count,
            CollisionEventCapacity = collisionBuffer.count,
            CoarseContactCapacity = coarseCollisionBuffer.count,
            NormCenter = hierarchy.NormCenter,
            NormInvHalfExtent = hierarchy.NormInvHalfExtent,
        };

        return snapshot.ActiveCount > 0 && snapshot.TotalCount > 0;
    }

    void RunSyncReadback(in ReadbackSnapshot snapshot) {
        oneUintScratch[0] = 0;
        snapshot.BoundaryChunkCount.GetData(oneUintScratch, 0, 0, 1);
        int boundaryRawCount = (int)oneUintScratch[0];
        int boundaryCount = GetBoundaryReadCount(boundaryRawCount, snapshot.BoundaryChunkCapacity);
        oneUintScratch[0] = 0;
        snapshot.CollisionEventCount.GetData(oneUintScratch, 0, 0, 1);
        int collisionCount = Mathf.Clamp((int)oneUintScratch[0], 0, snapshot.CollisionEventCapacity);
        oneUintScratch[0] = 0;
        snapshot.CoarseContactCount.GetData(oneUintScratch, 0, 0, 1);
        int coarseCollisionCount = Mathf.Clamp((int)oneUintScratch[0], 0, snapshot.CoarseContactCapacity);

        EnsureCapacity(ref dtPositions, snapshot.ActiveCount);
        snapshot.DtPositions.GetData(dtPositions, 0, 0, snapshot.ActiveCount);

        EnsureCapacity(ref globalPositions, snapshot.TotalCount);
        snapshot.GlobalPositions.GetData(globalPositions, 0, 0, snapshot.TotalCount);

        EnsureCapacity(ref invMass, snapshot.TotalCount);
        snapshot.InvMass.GetData(invMass, 0, 0, snapshot.TotalCount);

        EnsureCapacity(ref boundaryChunks, boundaryCount);
        if (boundaryCount > 0) {
            uint[] owners = new uint[boundaryCount];
            uint[] v0 = new uint[boundaryCount];
            uint[] v1 = new uint[boundaryCount];
            float2[] normals = new float2[boundaryCount];
            float2[] p0 = new float2[boundaryCount];
            float2[] p1 = new float2[boundaryCount];

            snapshot.BoundaryEdgeOwner.GetData(owners, 0, 0, boundaryCount);
            snapshot.BoundaryEdgeV0.GetData(v0, 0, 0, boundaryCount);
            snapshot.BoundaryEdgeV1.GetData(v1, 0, 0, boundaryCount);
            snapshot.BoundaryEdgeNormal.GetData(normals, 0, 0, boundaryCount);
            snapshot.BoundaryEdgeP0.GetData(p0, 0, 0, boundaryCount);
            snapshot.BoundaryEdgeP1.GetData(p1, 0, 0, boundaryCount);

            int emitted = 0;
            for (int i = 0; i < boundaryCount; i++) {
                if (owners[i] == uint.MaxValue)
                    continue;
                if ((int)owners[i] < 0)
                    continue;

                float2 seg = p1[i] - p0[i];
                if (math.lengthsq(seg) <= 1e-12f)
                    continue;

                boundaryChunks[emitted] = new BoundaryChunkGpu {
                    va = (int)v0[i],
                    vb = (int)v1[i],
                    ownerID = (int)owners[i],
                    outNormal = normals[i],
                    p0 = p0[i],
                    p1 = p1[i],
                };
                emitted++;
            }

            boundaryCount = emitted;
        }

        EnsureCapacity(ref collisionEvents, collisionCount);
        if (collisionCount > 0)
            snapshot.CollisionEvents.GetData(collisionEvents, 0, 0, collisionCount);

        EnsureCapacity(ref coarseCollisionEvents, coarseCollisionCount);
        if (coarseCollisionCount > 0)
            snapshot.CoarseContacts.GetData(coarseCollisionEvents, 0, 0, coarseCollisionCount);

        lastActiveCount = snapshot.ActiveCount;
        lastTotalCount = snapshot.TotalCount;
        lastBoundaryCount = boundaryCount;
        lastCollisionCount = collisionCount;
        lastCoarseCollisionCount = coarseCollisionCount;
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

        int invMassBytes = snapshot.TotalCount * sizeof(float);
        AsyncGPUReadback.Request(snapshot.InvMass, invMassBytes, 0, req => OnAsyncInvMass(req, state.RequestId));

        ComputeBuffer boundaryOwnerBuffer = snapshot.BoundaryEdgeOwner;
        ComputeBuffer boundaryV0Buffer = snapshot.BoundaryEdgeV0;
        ComputeBuffer boundaryV1Buffer = snapshot.BoundaryEdgeV1;
        ComputeBuffer boundaryNormalBuffer = snapshot.BoundaryEdgeNormal;
        ComputeBuffer boundaryP0Buffer = snapshot.BoundaryEdgeP0;
        ComputeBuffer boundaryP1Buffer = snapshot.BoundaryEdgeP1;
        int boundaryChunkCapacity = snapshot.BoundaryChunkCapacity;
        ComputeBuffer collisionEventsBuffer = snapshot.CollisionEvents;
        int collisionEventCapacity = snapshot.CollisionEventCapacity;
        ComputeBuffer coarseCollisionBuffer = snapshot.CoarseContacts;
        int coarseCollisionCapacity = snapshot.CoarseContactCapacity;

        AsyncGPUReadback.Request(snapshot.BoundaryChunkCount, req => OnAsyncBoundaryCount(req, state.RequestId, boundaryOwnerBuffer, boundaryV0Buffer, boundaryV1Buffer, boundaryNormalBuffer, boundaryP0Buffer, boundaryP1Buffer, boundaryChunkCapacity));
        AsyncGPUReadback.Request(snapshot.CollisionEventCount, req => OnAsyncCollisionCount(req, state.RequestId, collisionEventsBuffer, collisionEventCapacity));
        AsyncGPUReadback.Request(snapshot.CoarseContactCount, req => OnAsyncCoarseCollisionCount(req, state.RequestId, coarseCollisionBuffer, coarseCollisionCapacity));
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

    void OnAsyncInvMass(AsyncGPUReadbackRequest request, int requestId) {
        if (!TryGetPending(requestId, out PendingAsyncReadback state))
            return;

        if (request.hasError) {
            state.Failed = true;
            TryFinalizeAsync(state);
            return;
        }

        var data = request.GetData<float>();
        state.InvMass = new float[data.Length];
        data.CopyTo(state.InvMass);
        state.InvMassReady = true;

        TryFinalizeAsync(state);
    }

    void OnAsyncBoundaryCount(
        AsyncGPUReadbackRequest request,
        int requestId,
        ComputeBuffer boundaryOwnerBuffer,
        ComputeBuffer boundaryV0Buffer,
        ComputeBuffer boundaryV1Buffer,
        ComputeBuffer boundaryNormalBuffer,
        ComputeBuffer boundaryP0Buffer,
        ComputeBuffer boundaryP1Buffer,
        int boundaryCapacity
    ) {
        if (!TryGetPending(requestId, out PendingAsyncReadback state))
            return;

        if (request.hasError) {
            state.Failed = true;
            TryFinalizeAsync(state);
            return;
        }

        var data = request.GetData<uint>();
        int rawCount = (data.Length > 0) ? (int)data[0] : 0;
        int count = GetBoundaryReadCount(rawCount, boundaryCapacity);

        state.BoundaryCount = count;
        state.BoundaryRawCount = rawCount;
        state.BoundaryReadCount = count;
        state.BoundaryCountReady = true;

        if (count <= 0) {
            state.Boundaries = Array.Empty<BoundaryChunkGpu>();
            state.BoundaryOwnerReady = true;
            state.BoundaryV0Ready = true;
            state.BoundaryV1Ready = true;
            state.BoundaryNormalReady = true;
            state.BoundaryP0Ready = true;
            state.BoundaryP1Ready = true;
            TryFinalizeAsync(state);
            return;
        }

        int bytesUint = count * sizeof(uint);
        int bytesFloat2 = count * sizeof(float) * 2;
        AsyncGPUReadback.Request(boundaryOwnerBuffer, bytesUint, 0, req => OnAsyncBoundaryOwnerData(req, requestId));
        AsyncGPUReadback.Request(boundaryV0Buffer, bytesUint, 0, req => OnAsyncBoundaryV0Data(req, requestId));
        AsyncGPUReadback.Request(boundaryV1Buffer, bytesUint, 0, req => OnAsyncBoundaryV1Data(req, requestId));
        AsyncGPUReadback.Request(boundaryNormalBuffer, bytesFloat2, 0, req => OnAsyncBoundaryNormalData(req, requestId));
        AsyncGPUReadback.Request(boundaryP0Buffer, bytesFloat2, 0, req => OnAsyncBoundaryP0Data(req, requestId));
        AsyncGPUReadback.Request(boundaryP1Buffer, bytesFloat2, 0, req => OnAsyncBoundaryP1Data(req, requestId));
    }

    void OnAsyncBoundaryOwnerData(AsyncGPUReadbackRequest request, int requestId) {
        if (!TryGetPending(requestId, out PendingAsyncReadback state))
            return;

        if (request.hasError) {
            state.Failed = true;
            TryFinalizeAsync(state);
            return;
        }

        var data = request.GetData<uint>();
        EnsureCapacity(ref state.Boundaries, state.BoundaryCount);
        for (int i = 0; i < state.BoundaryCount && i < data.Length; i++)
            state.Boundaries[i].ownerID = (int)data[i];
        state.BoundaryOwnerReady = true;

        TryFinalizeAsync(state);
    }

    void OnAsyncBoundaryV0Data(AsyncGPUReadbackRequest request, int requestId) {
        if (!TryGetPending(requestId, out PendingAsyncReadback state))
            return;

        if (request.hasError) {
            state.Failed = true;
            TryFinalizeAsync(state);
            return;
        }

        var data = request.GetData<uint>();
        EnsureCapacity(ref state.Boundaries, state.BoundaryCount);
        for (int i = 0; i < state.BoundaryCount && i < data.Length; i++)
            state.Boundaries[i].va = (int)data[i];
        state.BoundaryV0Ready = true;

        TryFinalizeAsync(state);
    }

    void OnAsyncBoundaryV1Data(AsyncGPUReadbackRequest request, int requestId) {
        if (!TryGetPending(requestId, out PendingAsyncReadback state))
            return;

        if (request.hasError) {
            state.Failed = true;
            TryFinalizeAsync(state);
            return;
        }

        var data = request.GetData<uint>();
        EnsureCapacity(ref state.Boundaries, state.BoundaryCount);
        for (int i = 0; i < state.BoundaryCount && i < data.Length; i++)
            state.Boundaries[i].vb = (int)data[i];
        state.BoundaryV1Ready = true;

        TryFinalizeAsync(state);
    }

    void OnAsyncBoundaryNormalData(AsyncGPUReadbackRequest request, int requestId) {
        if (!TryGetPending(requestId, out PendingAsyncReadback state))
            return;

        if (request.hasError) {
            state.Failed = true;
            TryFinalizeAsync(state);
            return;
        }

        var data = request.GetData<float2>();
        EnsureCapacity(ref state.Boundaries, state.BoundaryCount);
        for (int i = 0; i < state.BoundaryCount && i < data.Length; i++)
            state.Boundaries[i].outNormal = data[i];
        state.BoundaryNormalReady = true;

        TryFinalizeAsync(state);
    }

    void OnAsyncBoundaryP0Data(AsyncGPUReadbackRequest request, int requestId) {
        if (!TryGetPending(requestId, out PendingAsyncReadback state))
            return;

        if (request.hasError) {
            state.Failed = true;
            TryFinalizeAsync(state);
            return;
        }

        var data = request.GetData<float2>();
        EnsureCapacity(ref state.Boundaries, state.BoundaryCount);
        for (int i = 0; i < state.BoundaryCount && i < data.Length; i++)
            state.Boundaries[i].p0 = data[i];
        state.BoundaryP0Ready = true;

        TryFinalizeAsync(state);
    }

    void OnAsyncBoundaryP1Data(AsyncGPUReadbackRequest request, int requestId) {
        if (!TryGetPending(requestId, out PendingAsyncReadback state))
            return;

        if (request.hasError) {
            state.Failed = true;
            TryFinalizeAsync(state);
            return;
        }

        var data = request.GetData<float2>();
        EnsureCapacity(ref state.Boundaries, state.BoundaryCount);
        for (int i = 0; i < state.BoundaryCount && i < data.Length; i++)
            state.Boundaries[i].p1 = data[i];
        state.BoundaryP1Ready = true;

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

    void OnAsyncCoarseCollisionCount(AsyncGPUReadbackRequest request, int requestId, ComputeBuffer coarseCollisionBuffer, int coarseCollisionCapacity) {
        if (!TryGetPending(requestId, out PendingAsyncReadback state))
            return;

        if (request.hasError) {
            state.Failed = true;
            TryFinalizeAsync(state);
            return;
        }

        var data = request.GetData<uint>();
        int count = (data.Length > 0) ? Mathf.Clamp((int)data[0], 0, coarseCollisionCapacity) : 0;

        state.CoarseCollisionCount = count;
        state.CoarseCollisionCountReady = true;

        if (count <= 0) {
            state.CoarseCollisions = Array.Empty<CoarseContactGpu>();
            state.CoarseCollisionDataReady = true;
            TryFinalizeAsync(state);
            return;
        }

        int bytes = count * Marshal.SizeOf<CoarseContactGpu>();
        AsyncGPUReadback.Request(coarseCollisionBuffer, bytes, 0, req => OnAsyncCoarseCollisionData(req, requestId));
    }

    void OnAsyncCoarseCollisionData(AsyncGPUReadbackRequest request, int requestId) {
        if (!TryGetPending(requestId, out PendingAsyncReadback state))
            return;

        if (request.hasError) {
            state.Failed = true;
            TryFinalizeAsync(state);
            return;
        }

        var data = request.GetData<CoarseContactGpu>();
        state.CoarseCollisions = new CoarseContactGpu[data.Length];
        data.CopyTo(state.CoarseCollisions);
        state.CoarseCollisionDataReady = true;

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

        if (!state.DtReady || !state.GlobalReady || !state.InvMassReady || !state.BoundaryCountReady || !state.BoundaryOwnerReady || !state.BoundaryV0Ready || !state.BoundaryV1Ready || !state.BoundaryNormalReady || !state.BoundaryP0Ready || !state.BoundaryP1Ready || !state.CollisionCountReady || !state.CollisionDataReady || !state.CoarseCollisionCountReady || !state.CoarseCollisionDataReady)
            return;

        int emittedBoundaryCount = 0;
        int ownerValidCount = 0;
        int segmentValidCount = 0;
        for (int i = 0; i < state.BoundaryCount && i < state.Boundaries.Length; i++) {
            if (state.Boundaries[i].ownerID < 0)
                continue;
            ownerValidCount++;

            float2 seg = state.Boundaries[i].p1 - state.Boundaries[i].p0;
            if (math.lengthsq(seg) <= 1e-12f)
                continue;
            segmentValidCount++;

            state.Boundaries[emittedBoundaryCount++] = state.Boundaries[i];
        }
        state.BoundaryOwnerValidCount = ownerValidCount;
        state.BoundarySegmentValidCount = segmentValidCount;

        dtPositions = state.DtPositions;
        globalPositions = state.GlobalPositions;
        invMass = state.InvMass;
        boundaryChunks = state.Boundaries;
        collisionEvents = state.Collisions;
        coarseCollisionEvents = state.CoarseCollisions;

        lastActiveCount = Mathf.Min(state.ActiveCount, dtPositions.Length);
        lastTotalCount = Mathf.Min(state.TotalCount, globalPositions.Length);
        lastBoundaryCount = Mathf.Min(emittedBoundaryCount, boundaryChunks.Length);
        lastCollisionCount = Mathf.Min(state.CollisionCount, collisionEvents.Length);
        lastCoarseCollisionCount = Mathf.Min(state.CoarseCollisionCount, coarseCollisionEvents.Length);
        lastNormCenter = state.NormCenter;
        lastNormInvHalfExtent = state.NormInvHalfExtent;

        pending = null;
    }

    void DrawOverlayInGui(Camera cam) {
        float invHalfExtent = lastNormInvHalfExtent > 0f ? (1f / lastNormInvHalfExtent) : 0f;
        // Space contract in this pipeline:
        // - DT positions are normalized by hierarchy sync.
        // - Solver/global positions and emitted collision points are world-space.
        // Using value-range heuristics here can misclassify small world coordinates as normalized.
        lastDtLooksNormalized = true;
        lastGlobalLooksNormalized = false;
        lastArrowAttempted = 0;
        lastArrowDrawn = 0;
        lastArrowSkippedDegenerate = 0;
        lastArrowSkippedProjection = 0;
        lastArrowClipped = 0;
        lastArrowOverlapInstances = 0;
        lastIntersectionMarkers = 0;
        arrowOverlapCounts.Clear();

        float thickness = Mathf.Max(1f, lineThickness);

        if (drawBoundaries && lastBoundaryCount > 0 && lastTotalCount > 0 && globalPositions != null) {
            int drawCount = maxBoundarySegments > 0 ? Mathf.Min(lastBoundaryCount, maxBoundarySegments) : lastBoundaryCount;
            for (int i = 0; i < drawCount; i++) {
                BoundaryChunkGpu edge = boundaryChunks[i];
                float2 wa = ToWorld(edge.p0, lastGlobalLooksNormalized, invHalfExtent, lastNormCenter);
                float2 wb = ToWorld(edge.p1, lastGlobalLooksNormalized, invHalfExtent, lastNormCenter);

                float edgeLen2 = math.lengthsq(wb - wa);
                if (edgeLen2 <= 1e-12f)
                    continue;

                if (!TryWorldToGuiPoint(cam, wa, out Vector2 ga) || !TryWorldToGuiPoint(cam, wb, out Vector2 gb))
                    continue;

                DrawGuiLine(ga, gb, boundaryColor, thickness);

                if (drawBoundaryNormals) {
                    float2 mid = 0.5f * (wa + wb);
                    float2 n = edge.outNormal;
                    float nLen = math.length(n);
                    if (nLen > 1e-6f) {
                        n /= nLen;
                        float2 tip = mid + n * boundaryNormalLength;
                        if (TryWorldToGuiPoint(cam, mid, out Vector2 gMid) && TryWorldToGuiPoint(cam, tip, out Vector2 gTip))
                            DrawGuiLine(gMid, gTip, boundaryNormalColor, Mathf.Max(1f, thickness * 0.8f));
                    }
                }
            }
        }

        if (drawCollisionArrows && lastCollisionCount > 0 && lastTotalCount > 0 && globalPositions != null) {
            int drawCount = maxCollisionArrows > 0 ? Mathf.Min(lastCollisionCount, maxCollisionArrows) : lastCollisionCount;
            lastArrowAttempted = drawCount;
            lastArrowClipped = Mathf.Max(0, lastCollisionCount - drawCount);
            for (int i = 0; i < drawCount; i++) {
                int sourceIndex = (drawCount < lastCollisionCount)
                    ? Mathf.Clamp((int)((long)i * (long)lastCollisionCount / (long)drawCount), 0, lastCollisionCount - 1)
                    : i;

                CollisionEventGpu evt = collisionEvents[sourceIndex];
                if (!TryResolveFineContactOrigin(evt, invHalfExtent, out float2 c)) {
                    lastArrowSkippedDegenerate++;
                    continue;
                }

                float2 n = evt.normal;
                float nLen = math.length(n);
                if (!(nLen > 1e-6f)) {
                    lastArrowSkippedDegenerate++;
                    continue;
                }
                n /= nLen;

                int overlapKey = BuildArrowOverlapKey(c, n);
                if (!arrowOverlapCounts.TryGetValue(overlapKey, out int overlapCount))
                    overlapCount = 0;
                arrowOverlapCounts[overlapKey] = overlapCount + 1;
                if (overlapCount > 0) {
                    lastArrowOverlapInstances++;
                    float2 side = new float2(-n.y, n.x);
                    c += side * (overlapCount * overlapFanoutWorld);
                }

                float shaftLength = math.max(minArrowLength, evt.penetration * arrowScale);
                if (cam != null && cam.orthographic) {
                    float worldPerPixel = (2f * cam.orthographicSize) / math.max(1f, Screen.height);
                    float minVisibleLen = 6f * worldPerPixel;
                    shaftLength = math.max(shaftLength, minVisibleLen);
                }

                Color arrowColor = GetCollisionArrowColor(evt);
                if (DrawCollisionArrow(cam, c, n, shaftLength, thickness, arrowColor))
                    lastArrowDrawn++;
                else
                    lastArrowSkippedProjection++;

                if (highlightEdgeIntersections && IsLikelyEdgeEdgeFineContact(evt)) {
                    if (TryWorldToGuiPoint(cam, c, out Vector2 gC)) {
                        DrawCrossMarker(gC, intersectionMarkerPixels, intersectionColor, thickness);
                        lastIntersectionMarkers++;
                    }
                }
            }
        }

        if (drawCoarseCollisionArrows && lastCoarseCollisionCount > 0) {
            int drawCount = maxCollisionArrows > 0 ? Mathf.Min(lastCoarseCollisionCount, maxCollisionArrows) : lastCoarseCollisionCount;
            for (int i = 0; i < drawCount; i++) {
                int sourceIndex = (drawCount < lastCoarseCollisionCount)
                    ? Mathf.Clamp((int)((long)i * (long)lastCoarseCollisionCount / (long)drawCount), 0, lastCoarseCollisionCount - 1)
                    : i;

                CoarseContactGpu evt = coarseCollisionEvents[sourceIndex];
                float2 n = evt.normal;
                float nLen = math.length(n);
                if (!(nLen > 1e-6f))
                    continue;
                n /= nLen;

                if (!TryResolveCoarseContactOrigin(evt, invHalfExtent, out float2 c))
                    continue;

                float shaftLength = math.max(minArrowLength, evt.penetration * arrowScale);
                if (cam != null && cam.orthographic) {
                    float worldPerPixel = (2f * cam.orthographicSize) / math.max(1f, Screen.height);
                    float minVisibleLen = 6f * worldPerPixel;
                    shaftLength = math.max(shaftLength, minVisibleLen);
                }

                DrawCollisionArrow(cam, c, n, shaftLength, thickness, coarseCollisionArrowColor);
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
        if (!float.IsFinite(screen.x) || !float.IsFinite(screen.y) || !float.IsFinite(screen.z))
            return false;
        return screen.z >= 0f;
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

    static void DrawCrossMarker(Vector2 center, float sizePixels, Color color, float thickness) {
        float h = Mathf.Max(1f, sizePixels) * 0.5f;
        DrawGuiLine(new Vector2(center.x - h, center.y - h), new Vector2(center.x + h, center.y + h), color, thickness);
        DrawGuiLine(new Vector2(center.x - h, center.y + h), new Vector2(center.x + h, center.y - h), color, thickness);
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

    bool TryGetWorldPoint(int index, float invHalfExtent, out float2 world) {
        world = default;
        if (index < 0 || index >= lastTotalCount || globalPositions == null || index >= globalPositions.Length)
            return false;

        world = ToWorld(globalPositions[index], lastGlobalLooksNormalized, invHalfExtent, lastNormCenter);
        return true;
    }

    bool TryResolveAnyWorldPoint(int index, float invHalfExtent, out float2 world) {
        if (TryGetWorldPoint(index, invHalfExtent, out world))
            return true;

        if (index < 0 || index >= lastActiveCount || dtPositions == null || index >= dtPositions.Length) {
            world = default;
            return false;
        }

        world = ToWorld(dtPositions[index], lastDtLooksNormalized, invHalfExtent, lastNormCenter);
        return true;
    }

    bool IsFinite(float2 v) =>
    math.all(math.isfinite(v));

bool IsValidGi(uint gi) =>
    gi != InvalidU32;

bool TryResolveContactSidePoint(
    uint[] gi,
    float[] beta,
    float invHalfExtent,
    bool wantPositive,
    out float2 point)
{
    point = default;

    float w = 0f;
    float2 acc = 0f;

    for (int i = 0; i < 4; i++) {
        uint g = gi[i];
        if (!IsValidGi(g))
            continue;

        float b = beta[i];
        if (wantPositive) {
            if (b <= 1e-12f)
                continue;
        } else {
            if (b >= -1e-12f)
                continue;
            b = -b;
        }

        if (!TryGetWorldPoint((int)g, invHalfExtent, out float2 p))
            continue;
        if (!IsFinite(p))
            continue;

        acc += p * b;
        w += b;
    }

    if (w <= 1e-12f)
        return false;

    point = acc / w;
    return IsFinite(point);
}

bool TryResolveFineContactOrigin(CollisionEventGpu evt, float invHalfExtent, out float2 origin) {
    origin = default;

    float2 n = evt.normal;
    float n2 = math.lengthsq(n);
    bool hasNormal = n2 > 1e-20f;
    if (hasNormal)
        n *= math.rsqrt(n2);
    else
        n = 0f;

    uint[] gi = { evt.nodeGi0, evt.nodeGi1, evt.nodeGi2, evt.nodeGi3 };
    float[] beta = { evt.beta0, evt.beta1, evt.beta2, evt.beta3 };

    bool hasA = TryResolveContactSidePoint(gi, beta, invHalfExtent, wantPositive: false, out float2 pA);
    bool hasB = TryResolveContactSidePoint(gi, beta, invHalfExtent, wantPositive: true, out float2 pB);

    if (hasA && hasB) {
        origin = 0.5f * (pA + pB);
        return IsFinite(origin);
    }

    float pen = math.max(evt.penetration, 0f);

    if (hasA) {
        origin = hasNormal ? (pA + 0.5f * pen * n) : pA;
        return IsFinite(origin);
    }

    if (hasB) {
        origin = hasNormal ? (pB - 0.5f * pen * n) : pB;
        return IsFinite(origin);
    }

    return false;
}

bool TryResolveCoarseContactOrigin(CoarseContactGpu evt, float invHalfExtent, out float2 origin) {
    origin = default;

    float2 n = evt.normal;
    float n2 = math.lengthsq(n);
    bool hasNormal = n2 > 1e-20f;
    if (hasNormal)
        n *= math.rsqrt(n2);
    else
        n = 0f;

    if (IsFinite(evt.point)) {
        origin = evt.point;
        return true;
    }

    bool hasA = false;
    bool hasB = false;
    float2 a = 0f;
    float2 b = 0f;

    if (evt.coarseGiA != InvalidU32)
        hasA = TryGetWorldPoint((int)evt.coarseGiA, invHalfExtent, out a) && IsFinite(a);

    if (evt.coarseGiB != InvalidU32)
        hasB = TryGetWorldPoint((int)evt.coarseGiB, invHalfExtent, out b) && IsFinite(b);

    if (hasA && hasB) {
        origin = 0.5f * (a + b);
        return IsFinite(origin);
    }

    float pen = math.max(evt.penetration, 0f);

    if (hasA) {
        origin = hasNormal ? (a + 0.5f * pen * n) : a;
        return IsFinite(origin);
    }

    if (hasB) {
        origin = hasNormal ? (b - 0.5f * pen * n) : b;
        return IsFinite(origin);
    }

    return false;
}

    static bool IsLikelyEdgeEdgeFineContact(CollisionEventGpu evt) {
        int validNodes = 0;
        if (evt.nodeGi0 != InvalidU32) validNodes++;
        if (evt.nodeGi1 != InvalidU32) validNodes++;
        if (evt.nodeGi2 != InvalidU32) validNodes++;
        if (evt.nodeGi3 != InvalidU32) validNodes++;
        return validNodes >= 4;
    }

    bool DrawCollisionArrow(Camera cam, float2 origin, float2 direction, float shaftLength, float thickness, Color arrowColor) {
        float dirLen = math.length(direction);
        if (!(dirLen > 1e-6f))
            return false;

        float2 n = direction / dirLen;
        float2 tip = origin + n * shaftLength;

        float headLen = math.max(0.001f, arrowHeadLength);
        float headWidth = headLen * math.clamp(arrowHeadWidthScale, 0.1f, 1f);
        float2 side = new float2(-n.y, n.x);
        float2 back = tip - n * headLen;
        float2 left = back + side * headWidth;
        float2 right = back - side * headWidth;

        if (!TryWorldToGuiPoint(cam, origin, out Vector2 gOrigin) ||
            !TryWorldToGuiPoint(cam, tip, out Vector2 gTip) ||
            !TryWorldToGuiPoint(cam, left, out Vector2 gLeft) ||
            !TryWorldToGuiPoint(cam, right, out Vector2 gRight))
            return false;

        DrawGuiLine(gOrigin, gTip, arrowColor, thickness);
        DrawGuiLine(gTip, gLeft, arrowColor, thickness);
        DrawGuiLine(gTip, gRight, arrowColor, thickness);
        return true;
    }

    Color GetCollisionArrowColor(CollisionEventGpu evt) {
        if (!colorArrowsByOwnerPair)
            return collisionArrowColor;

        uint a = evt.ownerA;
        uint b = evt.ownerB;
        if (b < a) {
            uint t = a;
            a = b;
            b = t;
        }

        unchecked {
            uint h = 2166136261u;
            h = (h ^ a) * 16777619u;
            h = (h ^ b) * 16777619u;
            float hue = (h & 0x00FFFFFFu) / 16777215f;
            return Color.HSVToRGB(hue, 0.82f, 1f);
        }
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

    static int BuildArrowOverlapKey(float2 origin, float2 direction) {
        int ox = Mathf.RoundToInt(origin.x * 200f);
        int oy = Mathf.RoundToInt(origin.y * 200f);
        int dx = Mathf.RoundToInt(direction.x * 100f);
        int dy = Mathf.RoundToInt(direction.y * 100f);

        unchecked {
            int h = 17;
            h = h * 31 + ox;
            h = h * 31 + oy;
            h = h * 31 + dx;
            h = h * 31 + dy;
            return h;
        }
    }
}
