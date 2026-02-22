using Unity.Mathematics;
using UnityEngine;
using GPU.Delaunay;
using System;

namespace GPU.Solver {
    public sealed partial class XPBISolver {
        private ComputeBuffer pos;
        private ComputeBuffer vel;
        private ComputeBuffer invMass;
        private ComputeBuffer flags;
        private ComputeBuffer restVolume;
        private ComputeBuffer parentIndex;
        private ComputeBuffer F;
        private ComputeBuffer Fp;
        private ComputeBuffer currentVolumeBits;
        private ComputeBuffer kernelH;
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
        private ComputeBuffer[] relaxArgsByLevel;


        // CPU mirror arrays.
        private float2[] posCpu;
        private float2[] velCpu;
        private float[] invMassCpu;
        private uint[] flagsCpu;
        private float[] restVolumeCpu;
        private int[] parentIndexCpu;
        private float4[] FCpu;
        private float4[] FpCpu;
        private ForceEvent[] forceEventsCpu;
        private uint[] convergenceDebugCpu;
        private float[] levelCellSizeCpu;

        // Capacity and event counts.
        private int capacity;
        private int forceEventsCapacity;
        private int forceEventsCount;
        private int convergenceDebugRequiredUInts;
        private int convergenceDebugMaxIter;
        private int convergenceDebugLevels;

        private void InitializeFromMeshless(Meshless m) {
            meshless = m;
            int n = m.nodes.Count;
            EnsureCapacity(n);

            for (int i = 0; i < n; i++) {
                var node = m.nodes[i];
                posCpu[i] = node.pos;
                velCpu[i] = node.vel;
                invMassCpu[i] = node.invMass;
                flagsCpu[i] = node.isFixed || node.invMass <= 0f ? 1u : 0u;
                restVolumeCpu[i] = node.restVolume;
                parentIndexCpu[i] = -1;
                FCpu[i] = new float4(node.F.c0, node.F.c1);
                FpCpu[i] = new float4(node.Fp.c0, node.Fp.c1);
            }

            pos.SetData(posCpu, 0, 0, n);
            vel.SetData(velCpu, 0, 0, n);
            invMass.SetData(invMassCpu, 0, 0, n);
            flags.SetData(flagsCpu, 0, 0, n);
            restVolume.SetData(restVolumeCpu, 0, 0, n);
            parentIndex.SetData(parentIndexCpu, 0, 0, n);
            F.SetData(FCpu, 0, 0, n);
            Fp.SetData(FpCpu, 0, 0, n);

            if (coloringPerLevel == null || coloringPerLevel.Length <= m.maxLayer) {
                int newSize = m.maxLayer + 1;
                Array.Resize(ref coloringPerLevel, newSize);
                Array.Resize(ref relaxArgsByLevel, newSize);
                Array.Resize(ref levelCellSizeCpu, newSize);
                for (int i = 0; i <= m.maxLayer; i++)
                    levelCellSizeCpu[i] = m.layerRadii[i];
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
            invMass = new ComputeBuffer(capacity, sizeof(float), ComputeBufferType.Structured);
            flags = new ComputeBuffer(capacity, sizeof(uint), ComputeBufferType.Structured);
            restVolume = new ComputeBuffer(capacity, sizeof(float), ComputeBufferType.Structured);
            parentIndex = new ComputeBuffer(capacity, sizeof(int), ComputeBufferType.Structured);
            F = new ComputeBuffer(capacity, sizeof(float) * 4, ComputeBufferType.Structured);
            Fp = new ComputeBuffer(capacity, sizeof(float) * 4, ComputeBufferType.Structured);

            currentVolumeBits = new ComputeBuffer(capacity, sizeof(uint), ComputeBufferType.Structured);
            kernelH = new ComputeBuffer(capacity, sizeof(float), ComputeBufferType.Structured);
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
            invMassCpu = new float[capacity];
            flagsCpu = new uint[capacity];
            restVolumeCpu = new float[capacity];
            parentIndexCpu = new int[capacity];
            FCpu = new float4[capacity];
            FpCpu = new float4[capacity];

            EnsureForceEventCapacity(64);

            initializedCount = -1;
        }

        private void EnsureConvergenceDebugCapacity(int levels, int maxIter) {
            int requiredUInts = levels * maxIter * ConvergenceDebugIterBufSize;

            if (convergenceDebug != null &&
                convergenceDebug.IsValid() &&
                convergenceDebug.count == requiredUInts &&
                convergenceDebugMaxIter == maxIter &&
                convergenceDebugLevels == levels &&
                convergenceDebugCpu != null &&
                convergenceDebugCpu.Length == requiredUInts)
                return;

            convergenceDebug?.Dispose();

            convergenceDebug = new ComputeBuffer(requiredUInts, sizeof(uint), ComputeBufferType.Structured);
            convergenceDebugCpu = new uint[requiredUInts];
            convergenceDebugRequiredUInts = requiredUInts;
            convergenceDebugMaxIter = maxIter;
            convergenceDebugLevels = levels;
            return;
        }

        void EnsureForceEventCapacity(int n) {
            if (n <= forceEventsCapacity) return;

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
            asyncCb.SetComputeIntParam(shader, "_TotalCount", total);
            asyncCb.SetComputeIntParam(shader, "_Base", 0);
        }

        void PrepareParentRebuildBuffers(DT dtLevel, int activeCount, int fineCount) {
            asyncCb.SetComputeIntParam(shader, "_DtNeighborCount", dtLevel.NeighborCount);
            asyncCb.SetComputeIntParam(shader, "_ParentRangeStart", activeCount);
            asyncCb.SetComputeIntParam(shader, "_ParentRangeEnd", fineCount);
            asyncCb.SetComputeIntParam(shader, "_ParentCoarseCount", activeCount);

            asyncCb.SetComputeBufferParam(shader, kRebuildParentsAtLevel, "_Pos", pos);
            asyncCb.SetComputeBufferParam(shader, kRebuildParentsAtLevel, "_ParentIndex", parentIndex);
            asyncCb.SetComputeBufferParam(shader, kRebuildParentsAtLevel, "_DtNeighbors", dtLevel.NeighborsBuffer);
            asyncCb.SetComputeBufferParam(shader, kRebuildParentsAtLevel, "_DtNeighborCounts", dtLevel.NeighborCountsBuffer);
        }

        void PrepareIntegratePosParams() {
            asyncCb.SetComputeBufferParam(shader, kIntegratePositions, "_Pos", pos);
            asyncCb.SetComputeBufferParam(shader, kIntegratePositions, "_Vel", vel);
            asyncCb.SetComputeBufferParam(shader, kIntegratePositions, "_InvMass", invMass);
            asyncCb.SetComputeBufferParam(shader, kIntegratePositions, "_Flags", flags);
        }

        private void PrepareUpdateDtPosParams(int level, DT dtLevel, int activeCount,
            Meshless m, int pingWrite) {
            asyncCb.SetComputeIntParam(shader, "_ActiveCount", activeCount);
            asyncCb.SetComputeIntParam(shader, "_DtNeighborCount", dtLevel.NeighborCount);
            asyncCb.SetComputeBufferParam(shader, kUpdateDtPositions, "_Pos", pos);
            asyncCb.SetComputeBufferParam(shader, kUpdateDtPositions, "_DtPositions",
                dtLevel.GetPositionsBuffer(pingWrite));
            asyncCb.SetComputeVectorParam(shader, "_DtNormCenter",
                new Vector4(m.DtNormCenter.x, m.DtNormCenter.y, 0f, 0f));
            asyncCb.SetComputeFloatParam(shader, "_DtNormInvHalfExtent", m.DtNormInvHalfExtent);
        }

        void PrepareApplyForcesParams() {
            asyncCb.SetComputeBufferParam(shader, kApplyGameplayForces, "_Vel", vel);
            asyncCb.SetComputeBufferParam(shader, kApplyGameplayForces, "_InvMass", invMass);
            asyncCb.SetComputeBufferParam(shader, kApplyGameplayForces, "_Flags", flags);

            asyncCb.SetComputeBufferParam(shader, kExternalForces, "_Vel", vel);
            asyncCb.SetComputeBufferParam(shader, kExternalForces, "_InvMass", invMass);
            asyncCb.SetComputeBufferParam(shader, kExternalForces, "_Flags", flags);
        }

        void PrepareRelaxBuffers(DT dtLevel, int activeCount, int fineCount, int tickIndex) {
            asyncCb.SetComputeIntParam(shader, "_ActiveCount", activeCount);
            asyncCb.SetComputeIntParam(shader, "_FineCount", fineCount);
            asyncCb.SetComputeIntParam(shader, "_DtNeighborCount", dtLevel.NeighborCount);

            asyncCb.SetComputeBufferParam(shader, kClearHierarchicalStats, "_CurrentVolumeBits", currentVolumeBits);
            asyncCb.SetComputeBufferParam(shader, kClearHierarchicalStats, "_CoarseFixed", coarseFixed);
            asyncCb.SetComputeBufferParam(shader, kClearHierarchicalStats, "_InvMass", invMass);
            asyncCb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_InvMass", invMass);
            asyncCb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_RestVolume", restVolume);
            asyncCb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_F", F);
            asyncCb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_ParentIndex", parentIndex);
            asyncCb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_CurrentVolumeBits", currentVolumeBits);
            asyncCb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_CoarseFixed", coarseFixed);
            asyncCb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_Flags", flags);
            asyncCb.SetComputeBufferParam(shader, kCacheKernelH, "_Pos", pos);
            asyncCb.SetComputeBufferParam(shader, kCacheKernelH, "_KernelH", kernelH);
            asyncCb.SetComputeBufferParam(shader, kComputeCorrectionL, "_Pos", pos);
            asyncCb.SetComputeBufferParam(shader, kComputeCorrectionL, "_KernelH", kernelH);
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
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_InvMass", invMass);
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_Flags", flags);
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_RestVolume", restVolume);
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_F0", F0);
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_L", L);
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_KernelH", kernelH);
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_CurrentVolumeBits", currentVolumeBits);
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_Lambda", lambda);
            asyncCb.SetComputeBufferParam(shader, kProlongate, "_Vel", vel);
            asyncCb.SetComputeBufferParam(shader, kProlongate, "_ParentIndex", parentIndex);
            asyncCb.SetComputeBufferParam(shader, kProlongate, "_SavedVelPrefix", savedVelPrefix);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_Pos", pos);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_Vel", vel);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_InvMass", invMass);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_Flags", flags);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_F0", F0);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_L", L);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_KernelH", kernelH);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_CurrentVolumeBits", currentVolumeBits);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_F", F);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_Fp", Fp);

            asyncCb.SetComputeBufferParam(shader, kCacheKernelH, "_DtNeighbors", dtLevel.NeighborsBuffer);
            asyncCb.SetComputeBufferParam(shader, kCacheKernelH, "_DtNeighborCounts", dtLevel.NeighborCountsBuffer);

            asyncCb.SetComputeBufferParam(shader, kComputeCorrectionL, "_DtNeighbors", dtLevel.NeighborsBuffer);
            asyncCb.SetComputeBufferParam(shader, kComputeCorrectionL, "_DtNeighborCounts", dtLevel.NeighborCountsBuffer);

            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_DtNeighbors", dtLevel.NeighborsBuffer);
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_DtNeighborCounts", dtLevel.NeighborCountsBuffer);

            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_DtNeighbors", dtLevel.NeighborsBuffer);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_DtNeighborCounts", dtLevel.NeighborCountsBuffer);

            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_CoarseFixed", coarseFixed);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_CoarseFixed", coarseFixed);

            asyncCb.SetComputeBufferParam(shader, kClearRestrictedDeltaV, "_RestrictedDeltaVBits", restrictedDeltaVBits);
            asyncCb.SetComputeBufferParam(shader, kClearRestrictedDeltaV, "_RestrictedDeltaVCount", restrictedDeltaVCount);
            asyncCb.SetComputeBufferParam(shader, kClearRestrictedDeltaV, "_RestrictedDeltaVAvg", restrictedDeltaVAvg);

            asyncCb.SetComputeBufferParam(shader, kRestrictGameplayDeltaVFromEvents, "_RestrictedDeltaVBits", restrictedDeltaVBits);
            asyncCb.SetComputeBufferParam(shader, kRestrictGameplayDeltaVFromEvents, "_RestrictedDeltaVCount", restrictedDeltaVCount);
            asyncCb.SetComputeBufferParam(shader, kRestrictGameplayDeltaVFromEvents, "_InvMass", invMass);
            asyncCb.SetComputeBufferParam(shader, kRestrictGameplayDeltaVFromEvents, "_Flags", flags);
            asyncCb.SetComputeBufferParam(shader, kRestrictGameplayDeltaVFromEvents, "_ParentIndex", parentIndex);
            asyncCb.SetComputeBufferParam(shader, kRestrictGameplayDeltaVFromEvents, "_ForceEvents", forceEvents);

            asyncCb.SetComputeBufferParam(shader, kApplyRestrictedDeltaVToActiveAndPrefix, "_RestrictedDeltaVBits", restrictedDeltaVBits);
            asyncCb.SetComputeBufferParam(shader, kApplyRestrictedDeltaVToActiveAndPrefix, "_RestrictedDeltaVCount", restrictedDeltaVCount);
            asyncCb.SetComputeBufferParam(shader, kApplyRestrictedDeltaVToActiveAndPrefix, "_RestrictedDeltaVAvg", restrictedDeltaVAvg);
            asyncCb.SetComputeBufferParam(shader, kApplyRestrictedDeltaVToActiveAndPrefix, "_CoarseFixed", coarseFixed);
            asyncCb.SetComputeBufferParam(shader, kApplyRestrictedDeltaVToActiveAndPrefix, "_Vel", vel);
            asyncCb.SetComputeBufferParam(shader, kApplyRestrictedDeltaVToActiveAndPrefix, "_SavedVelPrefix", savedVelPrefix);

            asyncCb.SetComputeBufferParam(shader, kRemoveRestrictedDeltaVFromActive, "_RestrictedDeltaVAvg", restrictedDeltaVAvg);
            asyncCb.SetComputeBufferParam(shader, kRemoveRestrictedDeltaVFromActive, "_Vel", vel);
        }

        void ReleaseBuffers() {
            pos?.Dispose(); pos = null;
            vel?.Dispose(); vel = null;
            invMass?.Dispose(); invMass = null;
            flags?.Dispose(); flags = null;
            restVolume?.Dispose(); restVolume = null;
            parentIndex?.Dispose(); parentIndex = null;
            F?.Dispose(); F = null;
            Fp?.Dispose(); Fp = null;

            currentVolumeBits?.Dispose(); currentVolumeBits = null;
            kernelH?.Dispose(); kernelH = null;
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
