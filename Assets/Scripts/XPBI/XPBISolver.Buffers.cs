using Unity.Mathematics;
using UnityEngine;
using GPU.Delaunay;
using System;

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
        private ComputeBuffer savedVelPrefix;
        private ComputeBuffer velDeltaBits;
        private ComputeBuffer coloringColor;
        private ComputeBuffer coloringProposed;
        private ComputeBuffer coloringPrio;
        private ComputeBuffer forceEvents;
        private ComputeBuffer coarseFixed;
        private ComputeBuffer restrictedDeltaVBits;
        private ComputeBuffer restrictedDeltaVCount;
        private ComputeBuffer restrictedDeltaVAvg;
        private ComputeBuffer convergenceDebug;
        private ComputeBuffer[] relaxArgsByLayer;


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
        private float[] layerCellSizeCpu;

        // Capacity and event counts.
        private int capacity;
        private int forceEventsCapacity;
        private int forceEventsCount;
        private int convergenceDebugRequiredUInts;
        private int convergenceDebugMaxIter;
        private int convergenceDebugLayers;

        private void InitializeFromMeshless(Meshless m) {
            meshless = m;
            int n = m.nodes.Count;
            EnsureCapacity(n);

            for (int i = 0; i < n; i++) {
                var node = m.nodes[i];
                posCpu[i] = node.pos;
                velCpu[i] = float2.zero;
                materialIdsCpu[i] = node.materialId;
                invMassCpu[i] = node.invMass;
                restVolumeCpu[i] = node.restVolume;
                parentIndexCpu[i] = -1;
                FCpu[i] = new float4(1f, 0f, 0f, 1f);
                FpCpu[i] = new float4(1f, 0f, 0f, 1f);
            }

            pos.SetData(posCpu, 0, 0, n);
            vel.SetData(velCpu, 0, 0, n);
            materialIds.SetData(materialIdsCpu, 0, 0, n);
            invMass.SetData(invMassCpu, 0, 0, n);
            restVolume.SetData(restVolumeCpu, 0, 0, n);
            parentIndex.SetData(parentIndexCpu, 0, 0, n);
            F.SetData(FCpu, 0, 0, n);
            Fp.SetData(FpCpu, 0, 0, n);

            if (coloringPerLayer == null || coloringPerLayer.Length <= m.maxLayer) {
                int newSize = m.maxLayer + 1;
                Array.Resize(ref coloringPerLayer, newSize);
                Array.Resize(ref relaxArgsByLayer, newSize);
                Array.Resize(ref layerCellSizeCpu, newSize);
                for (int i = 0; i <= m.maxLayer; i++)
                    layerCellSizeCpu[i] = m.layerRadii[i];
            }

            initializedCount = n;
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

            savedVelPrefix = new ComputeBuffer(capacity, sizeof(float) * 2, ComputeBufferType.Structured);
            velDeltaBits = new ComputeBuffer(capacity * 2, sizeof(uint), ComputeBufferType.Structured);

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

        private void SetCommonShaderParams(float dt, float gravity, float compliance, int total) {
            asyncCb.SetComputeFloatParam(shader, "_Dt", dt);
            asyncCb.SetComputeFloatParam(shader, "_Gravity", gravity);
            asyncCb.SetComputeFloatParam(shader, "_Compliance", compliance);
            asyncCb.SetComputeFloatParam(shader, "_MaxSpeed", Const.MaxVelocity);
            asyncCb.SetComputeFloatParam(shader, "_MaxStep", Const.MaxDisplacementPerTick);
            asyncCb.SetComputeIntParam(shader, "_TotalCount", total);
            asyncCb.SetComputeIntParam(shader, "_Base", 0);
            asyncCb.SetComputeFloatParam(shader, "_ProlongationScale", Const.ProlongationScale);
            asyncCb.SetComputeFloatParam(shader, "_PostProlongSmoothing", Const.PostProlongSmoothing);
            asyncCb.SetComputeFloatParam(shader, "_WendlandSupport", Const.WendlandSupport);
            asyncCb.SetComputeIntParam(shader, "_UseAffineProlongation", Const.UseAffineProlongation ? 1 : 0);
        }

        void PrepareParentRebuildBuffers(DT dtLayer, int activeCount, int fineCount) {
            asyncCb.SetComputeIntParam(shader, "_DtNeighborCount", dtLayer.NeighborCount);
            asyncCb.SetComputeIntParam(shader, "_ParentRangeStart", activeCount);
            asyncCb.SetComputeIntParam(shader, "_ParentRangeEnd", fineCount);
            asyncCb.SetComputeIntParam(shader, "_ParentCoarseCount", activeCount);

            asyncCb.SetComputeBufferParam(shader, kRebuildParentsAtLayer, "_Pos", pos);
            asyncCb.SetComputeBufferParam(shader, kRebuildParentsAtLayer, "_ParentIndex", parentIndex);
            asyncCb.SetComputeBufferParam(shader, kRebuildParentsAtLayer, "_DtNeighbors", dtLayer.NeighborsBuffer);
            asyncCb.SetComputeBufferParam(shader, kRebuildParentsAtLayer, "_DtNeighborCounts", dtLayer.NeighborCountsBuffer);
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

        private void PrepareUpdateDtPosParams(int layer, DT dtLayer, int activeCount,
            Meshless m, int pingWrite) {
            asyncCb.SetComputeIntParam(shader, "_ActiveCount", activeCount);
            asyncCb.SetComputeIntParam(shader, "_DtNeighborCount", dtLayer.NeighborCount);
            asyncCb.SetComputeBufferParam(shader, kUpdateDtPositions, "_Pos", pos);
            asyncCb.SetComputeBufferParam(shader, kUpdateDtPositions, "_DtPositions",
                dtLayer.GetPositionsBuffer(pingWrite));
            asyncCb.SetComputeVectorParam(shader, "_DtNormCenter",
                new Vector4(m.DtNormCenter.x, m.DtNormCenter.y, 0f, 0f));
            asyncCb.SetComputeFloatParam(shader, "_DtNormInvHalfExtent", m.DtNormInvHalfExtent);
        }

        void PrepareApplyForcesParams() {
            asyncCb.SetComputeBufferParam(shader, kApplyGameplayForces, "_Vel", vel);
            asyncCb.SetComputeBufferParam(shader, kApplyGameplayForces, "_InvMass", invMass);

            asyncCb.SetComputeBufferParam(shader, kExternalForces, "_Vel", vel);
            asyncCb.SetComputeBufferParam(shader, kExternalForces, "_InvMass", invMass);
        }

        void PrepareRelaxBuffers(DT dtLayer, int activeCount, int fineCount, int tickIndex, float layerKernelH) {
            var matLib = MaterialLibrary.Instance;
            var physicalParams = matLib != null ? matLib.PhysicalParamsBuffer : null;
            int physicalParamCount = (matLib != null && physicalParams != null) ? matLib.MaterialCount : 0;

            asyncCb.SetComputeIntParam(shader, "_ActiveCount", activeCount);
            asyncCb.SetComputeIntParam(shader, "_FineCount", fineCount);
            asyncCb.SetComputeIntParam(shader, "_DtNeighborCount", dtLayer.NeighborCount);
            asyncCb.SetComputeIntParam(shader, "_PhysicalParamCount", physicalParamCount);
            asyncCb.SetComputeFloatParam(shader, "_LayerKernelH", layerKernelH);

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
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_F", F);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_Fp", Fp);

            asyncCb.SetComputeBufferParam(shader, kComputeCorrectionL, "_DtNeighbors", dtLayer.NeighborsBuffer);
            asyncCb.SetComputeBufferParam(shader, kComputeCorrectionL, "_DtNeighborCounts", dtLayer.NeighborCountsBuffer);

            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_DtNeighbors", dtLayer.NeighborsBuffer);
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_DtNeighborCounts", dtLayer.NeighborCountsBuffer);

            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_DtNeighbors", dtLayer.NeighborsBuffer);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_DtNeighborCounts", dtLayer.NeighborCountsBuffer);

            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_CoarseFixed", coarseFixed);
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
            asyncCb.SetComputeBufferParam(shader, kSmoothProlongatedFineVel, "_DtNeighbors", dtLayer.NeighborsBuffer);
            asyncCb.SetComputeBufferParam(shader, kSmoothProlongatedFineVel, "_DtNeighborCounts", dtLayer.NeighborCountsBuffer);

            if (physicalParams != null) {
                asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_PhysicalParams", physicalParams);
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

            savedVelPrefix?.Dispose(); savedVelPrefix = null;
            velDeltaBits?.Dispose(); velDeltaBits = null;

            coloringColor?.Dispose(); coloringColor = null;
            coloringProposed?.Dispose(); coloringProposed = null;
            coloringPrio?.Dispose(); coloringPrio = null;

            forceEvents?.Dispose(); forceEvents = null;

            coarseFixed?.Dispose(); coarseFixed = null;
            restrictedDeltaVBits?.Dispose(); restrictedDeltaVBits = null;
            restrictedDeltaVCount?.Dispose(); restrictedDeltaVCount = null;
            restrictedDeltaVAvg?.Dispose(); restrictedDeltaVAvg = null;
            convergenceDebug?.Dispose(); convergenceDebug = null;

            initializedCount = -1;
        }
    }
}
