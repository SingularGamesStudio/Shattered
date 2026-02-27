using Unity.Mathematics;
using UnityEngine;
using GPU.Delaunay;
using GPU.Neighbors;
using System;
using System.Collections.Generic;

namespace GPU.Solver {
    public sealed partial class XPBISolver {
        private ComputeBuffer pos;
        private ComputeBuffer vel;
        private ComputeBuffer materialIds;
        private ComputeBuffer invMass;
        private ComputeBuffer restVolume;
        private ComputeBuffer parentIndex;
        private ComputeBuffer F;
        private ComputeBuffer Fp;
        private ComputeBuffer currentVolumeBits;
        private ComputeBuffer currentTotalMassBits;
        private ComputeBuffer fixedChildPosBits;
        private ComputeBuffer fixedChildCount;
        private ComputeBuffer L;
        private ComputeBuffer F0;
        private ComputeBuffer lambda;
        private ComputeBuffer collisionLambda;
        private ComputeBuffer savedVelPrefix;
        private ComputeBuffer velDeltaBits;
        private ComputeBuffer velPrev;
        private ComputeBuffer lambdaPrev;
        private ComputeBuffer jrVelDeltaBits;
        private ComputeBuffer jrLambdaDelta;
        private ComputeBuffer coloringColor;
        private ComputeBuffer coloringProposed;
        private ComputeBuffer coloringPrio;
        private ComputeBuffer forceEvents;
        private ComputeBuffer coarseFixed;
        private ComputeBuffer restrictedDeltaVBits;
        private ComputeBuffer restrictedDeltaVCount;
        private ComputeBuffer restrictedDeltaVAvg;
        private ComputeBuffer convergenceDebug;
        private readonly Dictionary<int, ComputeBuffer> globalLayerNodeMapBuffers = new Dictionary<int, ComputeBuffer>(16);
        private readonly Dictionary<int, ComputeBuffer> globalLayerGlobalToLocalBuffers = new Dictionary<int, ComputeBuffer>(16);
        private readonly Dictionary<int, ComputeBuffer> globalLayerOwnerByLocalBuffers = new Dictionary<int, ComputeBuffer>(16);
        private readonly Dictionary<int, int[]> globalLayerGlobalToLocalCpu = new Dictionary<int, int[]>(16);
        private readonly Dictionary<int, int[]> globalLayerNodeMapRefs = new Dictionary<int, int[]>(16);
        private readonly Dictionary<int, int[]> globalLayerGlobalToLocalRefs = new Dictionary<int, int[]>(16);
        private readonly Dictionary<int, int[]> globalLayerOwnerByLocalRefs = new Dictionary<int, int[]>(16);
        private readonly Dictionary<int, int> globalLayerNodeMapRefCounts = new Dictionary<int, int>(16);
        private readonly Dictionary<int, int> globalLayerGlobalToLocalRefCounts = new Dictionary<int, int>(16);
        private readonly Dictionary<int, int> globalLayerOwnerByLocalRefCounts = new Dictionary<int, int>(16);
        private readonly Dictionary<int, int> globalLayerNodeMapHashes = new Dictionary<int, int>(16);
        private readonly Dictionary<int, int> globalLayerGlobalToLocalHashes = new Dictionary<int, int>(16);
        private readonly Dictionary<int, int> globalLayerOwnerByLocalHashes = new Dictionary<int, int>(16);
        private readonly Dictionary<int, int> globalLayerGlobalToLocalTotals = new Dictionary<int, int>(16);
        private ComputeBuffer defaultDtGlobalNodeMap;
        private ComputeBuffer defaultDtGlobalToLocalMap;
        private ComputeBuffer defaultDtOwnerByLocal;


        // CPU mirror arrays.
        private float2[] posCpu;
        private float2[] velCpu;
        private int[] materialIdsCpu;
        private float[] invMassCpu;
        private float[] restVolumeCpu;
        private int[] parentIndexCpu;
        private float4[] FCpu;
        private float4[] FpCpu;
        private ForceEvent[] forceEventsCpu;
        private uint[] convergenceDebugCpu;

        // Capacity and event counts.
        private int capacity;
        private int forceEventsCapacity;
        private int forceEventsCount;
        private int convergenceDebugRequiredUInts;
        private int convergenceDebugMaxIter;
        private int convergenceDebugLayers;

        private void InitializeFromMeshless(System.Collections.Generic.List<MeshRange> ranges, int totalCount) {
            EnsureCapacity(totalCount);

            for (int rangeIdx = 0; rangeIdx < ranges.Count; rangeIdx++) {
                MeshRange range = ranges[rangeIdx];
                Meshless m = range.meshless;
                int baseIndex = range.baseIndex;

                for (int i = 0; i < range.totalCount; i++) {
                    int gi = baseIndex + i;
                    var node = m.nodes[i];
                    posCpu[gi] = node.pos;
                    velCpu[gi] = float2.zero;
                    materialIdsCpu[gi] = node.materialId;
                    invMassCpu[gi] = node.invMass;
                    restVolumeCpu[gi] = node.restVolume;
                    parentIndexCpu[gi] = -1;
                    FCpu[gi] = new float4(1f, 0f, 0f, 1f);
                    FpCpu[gi] = new float4(1f, 0f, 0f, 1f);
                }
            }

            pos.SetData(posCpu, 0, 0, totalCount);
            vel.SetData(velCpu, 0, 0, totalCount);
            materialIds.SetData(materialIdsCpu, 0, 0, totalCount);
            invMass.SetData(invMassCpu, 0, 0, totalCount);
            restVolume.SetData(restVolumeCpu, 0, 0, totalCount);
            parentIndex.SetData(parentIndexCpu, 0, 0, totalCount);
            F.SetData(FCpu, 0, 0, totalCount);
            Fp.SetData(FpCpu, 0, 0, totalCount);

            initializedCount = totalCount;
            layoutInitialized = true;
        }

        public void EnsureCapacity(int n) {
            if (n <= capacity) return;

            int newCap = math.max(256, capacity);
            while (newCap < n) newCap *= 2;
            capacity = newCap;

            ReleaseBuffers();

            pos = new ComputeBuffer(capacity, sizeof(float) * 2, ComputeBufferType.Structured);
            vel = new ComputeBuffer(capacity, sizeof(float) * 2, ComputeBufferType.Structured);
            materialIds = new ComputeBuffer(capacity, sizeof(int), ComputeBufferType.Structured);
            invMass = new ComputeBuffer(capacity, sizeof(float), ComputeBufferType.Structured);
            restVolume = new ComputeBuffer(capacity, sizeof(float), ComputeBufferType.Structured);
            parentIndex = new ComputeBuffer(capacity, sizeof(int), ComputeBufferType.Structured);
            F = new ComputeBuffer(capacity, sizeof(float) * 4, ComputeBufferType.Structured);
            Fp = new ComputeBuffer(capacity, sizeof(float) * 4, ComputeBufferType.Structured);

            currentVolumeBits = new ComputeBuffer(capacity, sizeof(uint), ComputeBufferType.Structured);
            currentTotalMassBits = new ComputeBuffer(capacity, sizeof(uint), ComputeBufferType.Structured);
            fixedChildPosBits = new ComputeBuffer(capacity * 2, sizeof(uint), ComputeBufferType.Structured);
            fixedChildCount = new ComputeBuffer(capacity, sizeof(uint), ComputeBufferType.Structured);
            L = new ComputeBuffer(capacity, sizeof(float) * 4, ComputeBufferType.Structured);
            F0 = new ComputeBuffer(capacity, sizeof(float) * 4, ComputeBufferType.Structured);
            lambda = new ComputeBuffer(capacity, sizeof(float), ComputeBufferType.Structured);
            collisionLambda = new ComputeBuffer(capacity * Const.NeighborCount, sizeof(float), ComputeBufferType.Structured);

            savedVelPrefix = new ComputeBuffer(capacity, sizeof(float) * 2, ComputeBufferType.Structured);
            velDeltaBits = new ComputeBuffer(capacity * 2, sizeof(uint), ComputeBufferType.Structured);
            velPrev = new ComputeBuffer(capacity, sizeof(float) * 2, ComputeBufferType.Structured);
            lambdaPrev = new ComputeBuffer(capacity, sizeof(float), ComputeBufferType.Structured);
            jrVelDeltaBits = new ComputeBuffer(capacity * 2, sizeof(uint), ComputeBufferType.Structured);
            jrLambdaDelta = new ComputeBuffer(capacity, sizeof(float), ComputeBufferType.Structured);

            coloringColor = new ComputeBuffer(capacity, sizeof(int), ComputeBufferType.Structured);
            coloringProposed = new ComputeBuffer(capacity, sizeof(int), ComputeBufferType.Structured);
            coloringPrio = new ComputeBuffer(capacity, sizeof(uint), ComputeBufferType.Structured);

            coarseFixed = new ComputeBuffer(capacity, sizeof(uint), ComputeBufferType.Structured);
            restrictedDeltaVBits = new ComputeBuffer(capacity * 2, sizeof(uint), ComputeBufferType.Structured);
            restrictedDeltaVCount = new ComputeBuffer(capacity, sizeof(uint), ComputeBufferType.Structured);
            restrictedDeltaVAvg = new ComputeBuffer(capacity, sizeof(float) * 2, ComputeBufferType.Structured);

            posCpu = new float2[capacity];
            velCpu = new float2[capacity];
            materialIdsCpu = new int[capacity];
            invMassCpu = new float[capacity];
            restVolumeCpu = new float[capacity];
            parentIndexCpu = new int[capacity];
            FCpu = new float4[capacity];
            FpCpu = new float4[capacity];

            EnsureForceEventCapacity(64);

            defaultDtGlobalNodeMap = new ComputeBuffer(1, sizeof(int), ComputeBufferType.Structured);
            defaultDtGlobalNodeMap.SetData(new[] { 0 });
            defaultDtGlobalToLocalMap = new ComputeBuffer(1, sizeof(int), ComputeBufferType.Structured);
            defaultDtGlobalToLocalMap.SetData(new[] { -1 });
            defaultDtOwnerByLocal = new ComputeBuffer(1, sizeof(int), ComputeBufferType.Structured);
            defaultDtOwnerByLocal.SetData(new[] { -1 });

            initializedCount = -1;
        }

        private void EnsureConvergenceDebugCapacity(int layers, int maxIter) {
            int requiredUInts = layers * maxIter * ConvergenceDebugIterBufSize;

            if (convergenceDebug != null &&
                convergenceDebug.IsValid() &&
                convergenceDebug.count == requiredUInts &&
                convergenceDebugMaxIter == maxIter &&
                convergenceDebugLayers == layers &&
                convergenceDebugCpu != null &&
                convergenceDebugCpu.Length == requiredUInts)
                return;

            convergenceDebug?.Dispose();

            convergenceDebug = new ComputeBuffer(requiredUInts, sizeof(uint), ComputeBufferType.Structured);
            convergenceDebugCpu = new uint[requiredUInts];
            convergenceDebugRequiredUInts = requiredUInts;
            convergenceDebugMaxIter = maxIter;
            convergenceDebugLayers = layers;
            return;
        }

        void EnsureForceEventCapacity(int n) {
            if (n <= forceEventsCapacity && forceEvents != null && forceEventsCpu != null) return;

            int newCap = math.max(64, forceEventsCapacity);
            while (newCap < n) newCap *= 2;
            forceEventsCapacity = newCap;

            forceEvents?.Dispose();
            forceEvents = new ComputeBuffer(forceEventsCapacity, sizeof(int) + sizeof(float) * 2, ComputeBufferType.Structured);

            forceEventsCpu = new ForceEvent[forceEventsCapacity];
        }

        private void SetCommonShaderParams(float dt, float gravity, float compliance, int total, int baseIndex) {
            asyncCb.SetComputeFloatParam(shader, "_Dt", dt);
            asyncCb.SetComputeFloatParam(shader, "_Gravity", gravity);
            asyncCb.SetComputeFloatParam(shader, "_Compliance", compliance);
            asyncCb.SetComputeFloatParam(shader, "_MaxSpeed", Const.MaxVelocity);
            asyncCb.SetComputeFloatParam(shader, "_MaxStep", Const.MaxDisplacementPerTick);
            asyncCb.SetComputeIntParam(shader, "_TotalCount", total);
            asyncCb.SetComputeIntParam(shader, "_Base", baseIndex);
            asyncCb.SetComputeFloatParam(shader, "_ProlongationScale", Const.ProlongationScale);
            asyncCb.SetComputeFloatParam(shader, "_PostProlongSmoothing", Const.PostProlongSmoothing);
            asyncCb.SetComputeFloatParam(shader, "_WendlandSupport", Const.WendlandSupport);
            asyncCb.SetComputeFloatParam(shader, "_CollisionSupportScale", Const.CollisionSupportScale);
            asyncCb.SetComputeFloatParam(shader, "_CollisionCompliance", Const.CollisionCompliance);
            asyncCb.SetComputeFloatParam(shader, "_CollisionFriction", Const.CollisionFriction);
            asyncCb.SetComputeFloatParam(shader, "_CollisionRestitution", Const.CollisionRestitution);
            asyncCb.SetComputeFloatParam(shader, "_CollisionRestitutionThreshold", Const.CollisionRestitutionThreshold);
            asyncCb.SetComputeIntParam(shader, "_CollisionEnable", Const.EnableCollisionConstraints ? 1 : 0);
            asyncCb.SetComputeIntParam(shader, "_UseAffineProlongation", Const.UseAffineProlongation ? 1 : 0);
        }

        void PrepareParentRebuildBuffers(INeighborSearch neighborSearch, int baseIndex, int activeCount, int fineCount, bool useDtGlobalNodeMap = false, int dtLocalBase = 0, ComputeBuffer dtGlobalNodeMap = null, ComputeBuffer dtGlobalToLayerLocalMap = null) {
            asyncCb.SetComputeIntParam(shader, "_Base", baseIndex);
            asyncCb.SetComputeIntParam(shader, "_DtNeighborCount", neighborSearch.NeighborCount);
            asyncCb.SetComputeIntParam(shader, "_ParentRangeStart", baseIndex + activeCount);
            asyncCb.SetComputeIntParam(shader, "_ParentRangeEnd", baseIndex + fineCount);
            asyncCb.SetComputeIntParam(shader, "_ParentCoarseCount", activeCount);

            asyncCb.SetComputeBufferParam(shader, kRebuildParentsAtLayer, "_Pos", pos);
            asyncCb.SetComputeBufferParam(shader, kRebuildParentsAtLayer, "_ParentIndex", parentIndex);
            asyncCb.SetComputeBufferParam(shader, kRebuildParentsAtLayer, "_DtNeighbors", neighborSearch.NeighborsBuffer);
            asyncCb.SetComputeBufferParam(shader, kRebuildParentsAtLayer, "_DtNeighborCounts", neighborSearch.NeighborCountsBuffer);
            BindDtGlobalMappingParams(kRebuildParentsAtLayer, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
        }

        void PrepareHierarchicalStatsBuffers(int activeCount, int fineCount) {
            asyncCb.SetComputeIntParam(shader, "_ActiveCount", activeCount);
            asyncCb.SetComputeIntParam(shader, "_FineCount", fineCount);

            asyncCb.SetComputeBufferParam(shader, kClearHierarchicalStats, "_CurrentVolumeBits", currentVolumeBits);
            asyncCb.SetComputeBufferParam(shader, kClearHierarchicalStats, "_CurrentTotalMassBits", currentTotalMassBits);
            asyncCb.SetComputeBufferParam(shader, kClearHierarchicalStats, "_FixedChildPosBits", fixedChildPosBits);
            asyncCb.SetComputeBufferParam(shader, kClearHierarchicalStats, "_FixedChildCount", fixedChildCount);
            asyncCb.SetComputeBufferParam(shader, kClearHierarchicalStats, "_CoarseFixed", coarseFixed);

            asyncCb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_InvMass", invMass);
            asyncCb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_RestVolume", restVolume);
            asyncCb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_Pos", pos);
            asyncCb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_F", F);
            asyncCb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_ParentIndex", parentIndex);
            asyncCb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_CurrentVolumeBits", currentVolumeBits);
            asyncCb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_CurrentTotalMassBits", currentTotalMassBits);
            asyncCb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_FixedChildPosBits", fixedChildPosBits);
            asyncCb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_FixedChildCount", fixedChildCount);

            asyncCb.SetComputeBufferParam(shader, kFinalizeHierarchicalStats, "_FixedChildCount", fixedChildCount);
            asyncCb.SetComputeBufferParam(shader, kFinalizeHierarchicalStats, "_CoarseFixed", coarseFixed);
        }

        void PrepareIntegratePosParams() {
            asyncCb.SetComputeBufferParam(shader, kClampVelocities, "_Vel", vel);
            asyncCb.SetComputeBufferParam(shader, kClampVelocities, "_InvMass", invMass);
            asyncCb.SetComputeBufferParam(shader, kIntegratePositions, "_Pos", pos);
            asyncCb.SetComputeBufferParam(shader, kIntegratePositions, "_Vel", vel);
            asyncCb.SetComputeBufferParam(shader, kIntegratePositions, "_InvMass", invMass);
        }

        private static int ComputeMappingHash(int[] mapping, int count) {
            unchecked {
                int hash = 17;
                hash = (hash * 31) ^ count;
                for (int i = 0; i < count; i++)
                    hash = (hash * 31) ^ mapping[i];
                return hash;
            }
        }

        private ComputeBuffer EnsureGlobalLayerNodeMapBuffer(int layer, int[] globalNodeByLocal, int activeCount) {
            if (globalNodeByLocal == null)
                throw new ArgumentNullException(nameof(globalNodeByLocal));

            if (activeCount < 0 || activeCount > globalNodeByLocal.Length)
                throw new ArgumentOutOfRangeException(nameof(activeCount));

            if (!globalLayerNodeMapBuffers.TryGetValue(layer, out ComputeBuffer mapBuffer) || mapBuffer == null || !mapBuffer.IsValid() || mapBuffer.count != activeCount) {
                mapBuffer?.Dispose();
                mapBuffer = new ComputeBuffer(math.max(1, activeCount), sizeof(int), ComputeBufferType.Structured);
                globalLayerNodeMapBuffers[layer] = mapBuffer;
                globalLayerNodeMapHashes[layer] = int.MinValue;
            }

            if (globalLayerNodeMapRefs.TryGetValue(layer, out int[] previousRef) &&
                ReferenceEquals(previousRef, globalNodeByLocal) &&
                globalLayerNodeMapRefCounts.TryGetValue(layer, out int previousCount) &&
                previousCount == activeCount)
                return mapBuffer;

            int mappingHash = ComputeMappingHash(globalNodeByLocal, activeCount);
            bool shouldUpload = !globalLayerNodeMapHashes.TryGetValue(layer, out int previousHash) || previousHash != mappingHash;

            if (shouldUpload && activeCount > 0)
                mapBuffer.SetData(globalNodeByLocal, 0, 0, activeCount);

            if (shouldUpload)
                globalLayerNodeMapHashes[layer] = mappingHash;

            globalLayerNodeMapRefs[layer] = globalNodeByLocal;
            globalLayerNodeMapRefCounts[layer] = activeCount;

            return mapBuffer;
        }

        private ComputeBuffer EnsureGlobalLayerGlobalToLocalBufferCached(int layer, int[] globalNodeByLocal, int activeCount, int totalNodeCount) {
            if (globalNodeByLocal == null)
                throw new ArgumentNullException(nameof(globalNodeByLocal));

            int safeTotal = math.max(1, totalNodeCount);

            if (!globalLayerGlobalToLocalBuffers.TryGetValue(layer, out ComputeBuffer mapBuffer) || mapBuffer == null || !mapBuffer.IsValid() || mapBuffer.count != safeTotal) {
                mapBuffer?.Dispose();
                mapBuffer = new ComputeBuffer(safeTotal, sizeof(int), ComputeBufferType.Structured);
                globalLayerGlobalToLocalBuffers[layer] = mapBuffer;
                globalLayerGlobalToLocalHashes[layer] = int.MinValue;
            }

            if (globalLayerGlobalToLocalRefs.TryGetValue(layer, out int[] previousRef) &&
                ReferenceEquals(previousRef, globalNodeByLocal) &&
                globalLayerGlobalToLocalRefCounts.TryGetValue(layer, out int previousCount) &&
                previousCount == activeCount &&
                globalLayerGlobalToLocalTotals.TryGetValue(layer, out int previousTotal) &&
                previousTotal == safeTotal)
                return mapBuffer;

            if (!globalLayerGlobalToLocalCpu.TryGetValue(layer, out int[] globalToLocal) || globalToLocal == null || globalToLocal.Length != safeTotal) {
                globalToLocal = new int[safeTotal];
                globalLayerGlobalToLocalCpu[layer] = globalToLocal;
            }

            int mappingHash = ComputeMappingHash(globalNodeByLocal, activeCount);
            bool totalChanged = !globalLayerGlobalToLocalTotals.TryGetValue(layer, out int prevTotal) || prevTotal != safeTotal;
            bool mappingChanged = !globalLayerGlobalToLocalHashes.TryGetValue(layer, out int prevHash) || prevHash != mappingHash;
            if (!totalChanged && !mappingChanged)
                return mapBuffer;

            for (int i = 0; i < globalToLocal.Length; i++)
                globalToLocal[i] = -1;

            for (int li = 0; li < activeCount; li++) {
                int gi = globalNodeByLocal[li];
                if (gi >= 0 && gi < globalToLocal.Length)
                    globalToLocal[gi] = li;
            }

            mapBuffer.SetData(globalToLocal);
            globalLayerGlobalToLocalHashes[layer] = mappingHash;
            globalLayerGlobalToLocalTotals[layer] = safeTotal;
            globalLayerGlobalToLocalRefs[layer] = globalNodeByLocal;
            globalLayerGlobalToLocalRefCounts[layer] = activeCount;
            return mapBuffer;
        }

        private void BindDtGlobalMappingParams(int kernel, bool useDtGlobalNodeMap, int dtLocalBase, ComputeBuffer dtGlobalNodeMap, ComputeBuffer dtGlobalToLayerLocalMap) {
            asyncCb.SetComputeIntParam(shader, "_UseDtGlobalNodeMap", useDtGlobalNodeMap ? 1 : 0);
            asyncCb.SetComputeIntParam(shader, "_DtLocalBase", dtLocalBase);

            asyncCb.SetComputeBufferParam(shader, kernel, "_DtGlobalNodeMap", dtGlobalNodeMap ?? defaultDtGlobalNodeMap);
            asyncCb.SetComputeBufferParam(shader, kernel, "_DtGlobalToLayerLocalMap", dtGlobalToLayerLocalMap ?? defaultDtGlobalToLocalMap);
        }

        private ComputeBuffer EnsureGlobalLayerOwnerByLocalBuffer(int layer, int[] ownerBodyByLocal, int activeCount) {
            if (ownerBodyByLocal == null)
                throw new ArgumentNullException(nameof(ownerBodyByLocal));

            if (activeCount < 0 || activeCount > ownerBodyByLocal.Length)
                throw new ArgumentOutOfRangeException(nameof(activeCount));

            if (!globalLayerOwnerByLocalBuffers.TryGetValue(layer, out ComputeBuffer ownerBuffer) || ownerBuffer == null || !ownerBuffer.IsValid() || ownerBuffer.count != math.max(1, activeCount)) {
                ownerBuffer?.Dispose();
                ownerBuffer = new ComputeBuffer(math.max(1, activeCount), sizeof(int), ComputeBufferType.Structured);
                globalLayerOwnerByLocalBuffers[layer] = ownerBuffer;
                globalLayerOwnerByLocalHashes[layer] = int.MinValue;
            }

            if (globalLayerOwnerByLocalRefs.TryGetValue(layer, out int[] previousRef) &&
                ReferenceEquals(previousRef, ownerBodyByLocal) &&
                globalLayerOwnerByLocalRefCounts.TryGetValue(layer, out int previousCount) &&
                previousCount == activeCount)
                return ownerBuffer;

            int ownerHash = ComputeMappingHash(ownerBodyByLocal, activeCount);
            bool shouldUpload = !globalLayerOwnerByLocalHashes.TryGetValue(layer, out int previousHash) || previousHash != ownerHash;
            if (shouldUpload && activeCount > 0)
                ownerBuffer.SetData(ownerBodyByLocal, 0, 0, activeCount);

            if (shouldUpload)
                globalLayerOwnerByLocalHashes[layer] = ownerHash;

            globalLayerOwnerByLocalRefs[layer] = ownerBodyByLocal;
            globalLayerOwnerByLocalRefCounts[layer] = activeCount;
            return ownerBuffer;
        }

        private void PrepareUpdateDtPosParamsMapped(DT dtLayer, ComputeBuffer dtGlobalNodeMap, ComputeBuffer dtGlobalToLayerLocalMap, int activeCount, float2 dtNormCenter, float dtNormInvHalfExtent, int pingWrite) {
            asyncCb.SetComputeIntParam(shader, "_ActiveCount", activeCount);
            asyncCb.SetComputeIntParam(shader, "_DtNeighborCount", dtLayer.NeighborCount);
            asyncCb.SetComputeBufferParam(shader, kUpdateDtPositionsMapped, "_Pos", pos);
            asyncCb.SetComputeBufferParam(shader, kUpdateDtPositionsMapped, "_DtPositions", dtLayer.GetPositionsBuffer(pingWrite));
            asyncCb.SetComputeBufferParam(shader, kUpdateDtPositionsMapped, "_DtGlobalNodeMap", dtGlobalNodeMap);
            BindDtGlobalMappingParams(kUpdateDtPositionsMapped, true, 0, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            asyncCb.SetComputeVectorParam(shader, "_DtNormCenter", new Vector4(dtNormCenter.x, dtNormCenter.y, 0f, 0f));
            asyncCb.SetComputeFloatParam(shader, "_DtNormInvHalfExtent", dtNormInvHalfExtent);
        }

        private void PrepareUpdateDtPosParamsUnmapped(DT dtLayer, int baseIndex, int activeCount, float2 dtNormCenter, float dtNormInvHalfExtent, int pingWrite) {
            asyncCb.SetComputeIntParam(shader, "_Base", baseIndex);
            asyncCb.SetComputeIntParam(shader, "_ActiveCount", activeCount);
            asyncCb.SetComputeIntParam(shader, "_DtNeighborCount", dtLayer.NeighborCount);
            asyncCb.SetComputeBufferParam(shader, kUpdateDtPositions, "_Pos", pos);
            asyncCb.SetComputeBufferParam(shader, kUpdateDtPositions, "_DtPositions", dtLayer.GetPositionsBuffer(pingWrite));
            BindDtGlobalMappingParams(kUpdateDtPositions, false, 0, null, null);
            asyncCb.SetComputeVectorParam(shader, "_DtNormCenter", new Vector4(dtNormCenter.x, dtNormCenter.y, 0f, 0f));
            asyncCb.SetComputeFloatParam(shader, "_DtNormInvHalfExtent", dtNormInvHalfExtent);
        }

        void PrepareApplyForcesParams() {
            asyncCb.SetComputeBufferParam(shader, kApplyGameplayForces, "_Vel", vel);
            asyncCb.SetComputeBufferParam(shader, kApplyGameplayForces, "_InvMass", invMass);

            asyncCb.SetComputeBufferParam(shader, kExternalForces, "_Vel", vel);
            asyncCb.SetComputeBufferParam(shader, kExternalForces, "_InvMass", invMass);
        }

        void PrepareRelaxBuffers(INeighborSearch neighborSearch, int baseIndex, int activeCount, int fineCount, int tickIndex, float layerKernelH, bool useDtGlobalNodeMap = false, int dtLocalBase = 0, ComputeBuffer dtGlobalNodeMap = null, ComputeBuffer dtGlobalToLayerLocalMap = null, ComputeBuffer dtOwnerByLocal = null) {
            var matLib = MaterialLibrary.Instance;
            var physicalParams = matLib != null ? matLib.PhysicalParamsBuffer : null;
            int physicalParamCount = (matLib != null && physicalParams != null) ? matLib.MaterialCount : 0;

            asyncCb.SetComputeIntParam(shader, "_Base", baseIndex);
            asyncCb.SetComputeIntParam(shader, "_ActiveCount", activeCount);
            asyncCb.SetComputeIntParam(shader, "_FineCount", fineCount);
            asyncCb.SetComputeIntParam(shader, "_DtNeighborCount", neighborSearch.NeighborCount);
            asyncCb.SetComputeIntParam(shader, "_PhysicalParamCount", physicalParamCount);
            asyncCb.SetComputeFloatParam(shader, "_LayerKernelH", layerKernelH);
            asyncCb.SetComputeIntParam(shader, "_UseDtOwnerFilter", dtOwnerByLocal != null ? 1 : 0);

            asyncCb.SetComputeBufferParam(shader, kClearHierarchicalStats, "_CurrentVolumeBits", currentVolumeBits);
            asyncCb.SetComputeBufferParam(shader, kClearHierarchicalStats, "_CurrentTotalMassBits", currentTotalMassBits);
            asyncCb.SetComputeBufferParam(shader, kClearHierarchicalStats, "_FixedChildPosBits", fixedChildPosBits);
            asyncCb.SetComputeBufferParam(shader, kClearHierarchicalStats, "_FixedChildCount", fixedChildCount);
            asyncCb.SetComputeBufferParam(shader, kClearHierarchicalStats, "_CoarseFixed", coarseFixed);
            asyncCb.SetComputeBufferParam(shader, kClearHierarchicalStats, "_InvMass", invMass);
            asyncCb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_InvMass", invMass);
            asyncCb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_RestVolume", restVolume);
            asyncCb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_Pos", pos);
            asyncCb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_F", F);
            asyncCb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_ParentIndex", parentIndex);
            asyncCb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_CurrentVolumeBits", currentVolumeBits);
            asyncCb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_CurrentTotalMassBits", currentTotalMassBits);
            asyncCb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_FixedChildPosBits", fixedChildPosBits);
            asyncCb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_FixedChildCount", fixedChildCount);
            asyncCb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_CoarseFixed", coarseFixed);
            asyncCb.SetComputeBufferParam(shader, kFinalizeHierarchicalStats, "_FixedChildCount", fixedChildCount);
            asyncCb.SetComputeBufferParam(shader, kFinalizeHierarchicalStats, "_CoarseFixed", coarseFixed);
            asyncCb.SetComputeBufferParam(shader, kComputeCorrectionL, "_Pos", pos);
            asyncCb.SetComputeBufferParam(shader, kComputeCorrectionL, "_CurrentVolumeBits", currentVolumeBits);
            asyncCb.SetComputeBufferParam(shader, kComputeCorrectionL, "_L", L);
            asyncCb.SetComputeBufferParam(shader, kCacheF0AndResetLambda, "_F", F);
            asyncCb.SetComputeBufferParam(shader, kCacheF0AndResetLambda, "_F0", F0);
            asyncCb.SetComputeBufferParam(shader, kCacheF0AndResetLambda, "_Lambda", lambda);
            asyncCb.SetComputeBufferParam(shader, kResetCollisionLambda, "_CollisionLambda", collisionLambda);
            asyncCb.SetComputeBufferParam(shader, kSaveVelPrefix, "_Vel", vel);
            asyncCb.SetComputeBufferParam(shader, kSaveVelPrefix, "_SavedVelPrefix", savedVelPrefix);
            asyncCb.SetComputeBufferParam(shader, kClearVelDelta, "_VelDeltaBits", velDeltaBits);
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_Pos", pos);
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_Vel", vel);
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_MaterialIds", materialIds);
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_InvMass", invMass);
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_RestVolume", restVolume);
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_F0", F0);
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_L", L);
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_CurrentVolumeBits", currentVolumeBits);
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_CurrentTotalMassBits", currentTotalMassBits);
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_FixedChildPosBits", fixedChildPosBits);
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_FixedChildCount", fixedChildCount);
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_Lambda", lambda);
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_CollisionLambda", collisionLambda);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_Pos", pos);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_Vel", vel);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_MaterialIds", materialIds);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_InvMass", invMass);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_RestVolume", restVolume);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_F0", F0);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_L", L);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_CurrentVolumeBits", currentVolumeBits);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_CurrentTotalMassBits", currentTotalMassBits);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_FixedChildPosBits", fixedChildPosBits);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_FixedChildCount", fixedChildCount);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_Lambda", lambda);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_CollisionLambda", collisionLambda);

            asyncCb.SetComputeBufferParam(shader, kProlongate, "_InvMass", invMass);
            asyncCb.SetComputeBufferParam(shader, kProlongate, "_Vel", vel);
            asyncCb.SetComputeBufferParam(shader, kProlongate, "_Pos", pos);
            asyncCb.SetComputeBufferParam(shader, kProlongate, "_MaterialIds", materialIds);
            asyncCb.SetComputeBufferParam(shader, kProlongate, "_RestVolume", restVolume);
            asyncCb.SetComputeBufferParam(shader, kProlongate, "_CurrentVolumeBits", currentVolumeBits);
            asyncCb.SetComputeBufferParam(shader, kProlongate, "_FixedChildPosBits", fixedChildPosBits);
            asyncCb.SetComputeBufferParam(shader, kProlongate, "_FixedChildCount", fixedChildCount);
            asyncCb.SetComputeBufferParam(shader, kProlongate, "_ParentIndex", parentIndex);
            asyncCb.SetComputeBufferParam(shader, kProlongate, "_SavedVelPrefix", savedVelPrefix);


            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_Pos", pos);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_Vel", vel);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_MaterialIds", materialIds);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_InvMass", invMass);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_F0", F0);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_L", L);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_CurrentVolumeBits", currentVolumeBits);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_FixedChildPosBits", fixedChildPosBits);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_FixedChildCount", fixedChildCount);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_F", F);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_Fp", Fp);

            asyncCb.SetComputeBufferParam(shader, kComputeCorrectionL, "_DtNeighbors", neighborSearch.NeighborsBuffer);
            asyncCb.SetComputeBufferParam(shader, kComputeCorrectionL, "_DtNeighborCounts", neighborSearch.NeighborCountsBuffer);
            asyncCb.SetComputeBufferParam(shader, kComputeCorrectionL, "_DtOwnerByLocal", dtOwnerByLocal ?? defaultDtOwnerByLocal);
            BindDtGlobalMappingParams(kComputeCorrectionL, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);

            BindDtGlobalMappingParams(kClearHierarchicalStats, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(kCacheHierarchicalStats, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(kFinalizeHierarchicalStats, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(kCacheF0AndResetLambda, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(kSaveVelPrefix, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(kClearVelDelta, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(kResetCollisionLambda, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(kClearRestrictedDeltaV, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(kRestrictGameplayDeltaVFromEvents, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(kRestrictFineVelocityResidualToActive, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(kApplyRestrictedDeltaVToActiveAndPrefix, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(kRemoveRestrictedDeltaVFromActive, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(kProlongate, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);

            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_DtNeighbors", neighborSearch.NeighborsBuffer);
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_DtNeighborCounts", neighborSearch.NeighborCountsBuffer);
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_DtOwnerByLocal", dtOwnerByLocal ?? defaultDtOwnerByLocal);
            BindDtGlobalMappingParams(kRelaxColored, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_DtNeighbors", neighborSearch.NeighborsBuffer);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_DtNeighborCounts", neighborSearch.NeighborCountsBuffer);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_DtOwnerByLocal", dtOwnerByLocal ?? defaultDtOwnerByLocal);
            BindDtGlobalMappingParams(kRelaxColoredPersistentCoarse, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);

            asyncCb.SetComputeBufferParam(shader, kJRSavePrevAndClear, "_Vel", vel);
            asyncCb.SetComputeBufferParam(shader, kJRSavePrevAndClear, "_Lambda", lambda);
            asyncCb.SetComputeBufferParam(shader, kJRSavePrevAndClear, "_VelPrev", velPrev);
            asyncCb.SetComputeBufferParam(shader, kJRSavePrevAndClear, "_LambdaPrev", lambdaPrev);
            asyncCb.SetComputeBufferParam(shader, kJRSavePrevAndClear, "_JRVelDeltaBits", jrVelDeltaBits);
            asyncCb.SetComputeBufferParam(shader, kJRSavePrevAndClear, "_JRLambdaDelta", jrLambdaDelta);

            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_Pos", pos);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_MaterialIds", materialIds);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_InvMass", invMass);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_RestVolume", restVolume);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_F0", F0);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_L", L);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_CurrentVolumeBits", currentVolumeBits);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_CurrentTotalMassBits", currentTotalMassBits);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_FixedChildPosBits", fixedChildPosBits);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_FixedChildCount", fixedChildCount);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_VelPrev", velPrev);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_LambdaPrev", lambdaPrev);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_JRVelDeltaBits", jrVelDeltaBits);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_JRLambdaDelta", jrLambdaDelta);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_DtNeighbors", neighborSearch.NeighborsBuffer);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_DtNeighborCounts", neighborSearch.NeighborCountsBuffer);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_DtOwnerByLocal", dtOwnerByLocal ?? defaultDtOwnerByLocal);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_CoarseFixed", coarseFixed);

            asyncCb.SetComputeBufferParam(shader, kJRApply, "_Vel", vel);
            asyncCb.SetComputeBufferParam(shader, kJRApply, "_Lambda", lambda);
            asyncCb.SetComputeBufferParam(shader, kJRApply, "_Pos", pos);
            asyncCb.SetComputeBufferParam(shader, kJRApply, "_MaterialIds", materialIds);
            asyncCb.SetComputeBufferParam(shader, kJRApply, "_RestVolume", restVolume);
            asyncCb.SetComputeBufferParam(shader, kJRApply, "_CurrentVolumeBits", currentVolumeBits);
            asyncCb.SetComputeBufferParam(shader, kJRApply, "_FixedChildPosBits", fixedChildPosBits);
            asyncCb.SetComputeBufferParam(shader, kJRApply, "_FixedChildCount", fixedChildCount);
            asyncCb.SetComputeBufferParam(shader, kJRApply, "_JRVelDeltaBits", jrVelDeltaBits);
            asyncCb.SetComputeBufferParam(shader, kJRApply, "_JRLambdaDelta", jrLambdaDelta);

            BindDtGlobalMappingParams(kJRSavePrevAndClear, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(kJRComputeDeltas, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(kJRApply, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);

            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_DtNeighbors", neighborSearch.NeighborsBuffer);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_DtNeighborCounts", neighborSearch.NeighborCountsBuffer);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_DtOwnerByLocal", dtOwnerByLocal ?? defaultDtOwnerByLocal);
            BindDtGlobalMappingParams(kCommitDeformation, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);

            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_CoarseFixed", coarseFixed);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_CoarseFixed", coarseFixed);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_CoarseFixed", coarseFixed);

            asyncCb.SetComputeBufferParam(shader, kClearRestrictedDeltaV, "_RestrictedDeltaVBits", restrictedDeltaVBits);
            asyncCb.SetComputeBufferParam(shader, kClearRestrictedDeltaV, "_RestrictedDeltaVCount", restrictedDeltaVCount);
            asyncCb.SetComputeBufferParam(shader, kClearRestrictedDeltaV, "_RestrictedDeltaVAvg", restrictedDeltaVAvg);

            asyncCb.SetComputeBufferParam(shader, kRestrictGameplayDeltaVFromEvents, "_RestrictedDeltaVBits", restrictedDeltaVBits);
            asyncCb.SetComputeBufferParam(shader, kRestrictGameplayDeltaVFromEvents, "_RestrictedDeltaVCount", restrictedDeltaVCount);
            asyncCb.SetComputeBufferParam(shader, kRestrictGameplayDeltaVFromEvents, "_InvMass", invMass);
            asyncCb.SetComputeBufferParam(shader, kRestrictGameplayDeltaVFromEvents, "_ParentIndex", parentIndex);
            asyncCb.SetComputeBufferParam(shader, kRestrictGameplayDeltaVFromEvents, "_ForceEvents", forceEvents);

            asyncCb.SetComputeBufferParam(shader, kRestrictFineVelocityResidualToActive, "_Vel", vel);
            asyncCb.SetComputeBufferParam(shader, kRestrictFineVelocityResidualToActive, "_InvMass", invMass);
            asyncCb.SetComputeBufferParam(shader, kRestrictFineVelocityResidualToActive, "_ParentIndex", parentIndex);
            asyncCb.SetComputeBufferParam(shader, kRestrictFineVelocityResidualToActive, "_RestrictedDeltaVBits", restrictedDeltaVBits);
            asyncCb.SetComputeBufferParam(shader, kRestrictFineVelocityResidualToActive, "_RestrictedDeltaVCount", restrictedDeltaVCount);

            asyncCb.SetComputeBufferParam(shader, kApplyRestrictedDeltaVToActiveAndPrefix, "_RestrictedDeltaVBits", restrictedDeltaVBits);
            asyncCb.SetComputeBufferParam(shader, kApplyRestrictedDeltaVToActiveAndPrefix, "_RestrictedDeltaVCount", restrictedDeltaVCount);
            asyncCb.SetComputeBufferParam(shader, kApplyRestrictedDeltaVToActiveAndPrefix, "_RestrictedDeltaVAvg", restrictedDeltaVAvg);
            asyncCb.SetComputeBufferParam(shader, kApplyRestrictedDeltaVToActiveAndPrefix, "_CoarseFixed", coarseFixed);
            asyncCb.SetComputeBufferParam(shader, kApplyRestrictedDeltaVToActiveAndPrefix, "_Vel", vel);
            asyncCb.SetComputeBufferParam(shader, kApplyRestrictedDeltaVToActiveAndPrefix, "_SavedVelPrefix", savedVelPrefix);

            asyncCb.SetComputeBufferParam(shader, kRemoveRestrictedDeltaVFromActive, "_RestrictedDeltaVAvg", restrictedDeltaVAvg);
            asyncCb.SetComputeBufferParam(shader, kRemoveRestrictedDeltaVFromActive, "_Vel", vel);

            asyncCb.SetComputeBufferParam(shader, kSmoothProlongatedFineVel, "_Vel", vel);
            asyncCb.SetComputeBufferParam(shader, kSmoothProlongatedFineVel, "_Pos", pos);
            asyncCb.SetComputeBufferParam(shader, kSmoothProlongatedFineVel, "_InvMass", invMass);
            asyncCb.SetComputeBufferParam(shader, kSmoothProlongatedFineVel, "_DtNeighbors", neighborSearch.NeighborsBuffer);
            asyncCb.SetComputeBufferParam(shader, kSmoothProlongatedFineVel, "_DtNeighborCounts", neighborSearch.NeighborCountsBuffer);
            asyncCb.SetComputeBufferParam(shader, kSmoothProlongatedFineVel, "_DtOwnerByLocal", dtOwnerByLocal ?? defaultDtOwnerByLocal);
            BindDtGlobalMappingParams(kSmoothProlongatedFineVel, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);

            if (physicalParams != null) {
                asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_PhysicalParams", physicalParams);
                asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_PhysicalParams", physicalParams);
                asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_PhysicalParams", physicalParams);
                asyncCb.SetComputeBufferParam(shader, kJRApply, "_PhysicalParams", physicalParams);
                asyncCb.SetComputeBufferParam(shader, kProlongate, "_PhysicalParams", physicalParams);
                asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_PhysicalParams", physicalParams);
            }
        }

        void ReleaseBuffers() {
            pos?.Dispose(); pos = null;
            vel?.Dispose(); vel = null;
            materialIds?.Dispose(); materialIds = null;
            invMass?.Dispose(); invMass = null;
            restVolume?.Dispose(); restVolume = null;
            parentIndex?.Dispose(); parentIndex = null;
            F?.Dispose(); F = null;
            Fp?.Dispose(); Fp = null;

            currentVolumeBits?.Dispose(); currentVolumeBits = null;
            currentTotalMassBits?.Dispose(); currentTotalMassBits = null;
            fixedChildPosBits?.Dispose(); fixedChildPosBits = null;
            fixedChildCount?.Dispose(); fixedChildCount = null;
            L?.Dispose(); L = null;
            F0?.Dispose(); F0 = null;
            lambda?.Dispose(); lambda = null;
            collisionLambda?.Dispose(); collisionLambda = null;

            savedVelPrefix?.Dispose(); savedVelPrefix = null;
            velDeltaBits?.Dispose(); velDeltaBits = null;
            velPrev?.Dispose(); velPrev = null;
            lambdaPrev?.Dispose(); lambdaPrev = null;
            jrVelDeltaBits?.Dispose(); jrVelDeltaBits = null;
            jrLambdaDelta?.Dispose(); jrLambdaDelta = null;

            coloringColor?.Dispose(); coloringColor = null;
            coloringProposed?.Dispose(); coloringProposed = null;
            coloringPrio?.Dispose(); coloringPrio = null;

            forceEvents?.Dispose(); forceEvents = null;

            coarseFixed?.Dispose(); coarseFixed = null;
            restrictedDeltaVBits?.Dispose(); restrictedDeltaVBits = null;
            restrictedDeltaVCount?.Dispose(); restrictedDeltaVCount = null;
            restrictedDeltaVAvg?.Dispose(); restrictedDeltaVAvg = null;
            convergenceDebug?.Dispose(); convergenceDebug = null;

            foreach (var kv in globalLayerNodeMapBuffers)
                kv.Value?.Dispose();
            globalLayerNodeMapBuffers.Clear();
            globalLayerNodeMapRefs.Clear();
            globalLayerNodeMapRefCounts.Clear();
            globalLayerNodeMapHashes.Clear();

            foreach (var kv in globalLayerGlobalToLocalBuffers)
                kv.Value?.Dispose();
            globalLayerGlobalToLocalBuffers.Clear();
            globalLayerGlobalToLocalCpu.Clear();
            globalLayerGlobalToLocalRefs.Clear();
            globalLayerGlobalToLocalRefCounts.Clear();
            globalLayerGlobalToLocalHashes.Clear();
            globalLayerGlobalToLocalTotals.Clear();

            foreach (var kv in globalLayerOwnerByLocalBuffers)
                kv.Value?.Dispose();
            globalLayerOwnerByLocalBuffers.Clear();
            globalLayerOwnerByLocalRefs.Clear();
            globalLayerOwnerByLocalRefCounts.Clear();
            globalLayerOwnerByLocalHashes.Clear();

            defaultDtGlobalNodeMap?.Dispose();
            defaultDtGlobalNodeMap = null;
            defaultDtGlobalToLocalMap?.Dispose();
            defaultDtGlobalToLocalMap = null;
            defaultDtOwnerByLocal?.Dispose();
            defaultDtOwnerByLocal = null;

            initializedCount = -1;
        }
    }
}
