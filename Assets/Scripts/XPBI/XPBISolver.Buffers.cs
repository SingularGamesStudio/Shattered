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

        // Capacity and event counts.
        private int capacity;
        private int forceEventsCapacity;
        private int forceEventsCount;

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

            posCpu = new float2[capacity];
            velCpu = new float2[capacity];
            invMassCpu = new float[capacity];
            flagsCpu = new uint[capacity];
            restVolumeCpu = new float[capacity];
            parentIndexCpu = new int[capacity];
            FCpu = new float4[capacity];
            FpCpu = new float4[capacity];

            EnsureForceEventCapacity(64);

            initialized = false;
            initializedCount = -1;
            parentsBuilt = false;
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

        void PrepareRelaxBuffers(DT dtLevel, int activeCount, int fineCount, int tickIndex) {
            asyncCb.SetComputeIntParam(shader, "_ActiveCount", activeCount);
            asyncCb.SetComputeIntParam(shader, "_FineCount", fineCount);
            asyncCb.SetComputeIntParam(shader, "_DtNeighborCount", dtLevel.NeighborCount);

            // Common buffers that are used by multiple kernels
            asyncCb.SetComputeBufferParam(shader, kClearCurrentVolume, "_CurrentVolumeBits", currentVolumeBits);
            asyncCb.SetComputeBufferParam(shader, kCacheVolumesHierarchical, "_RestVolume", restVolume);
            asyncCb.SetComputeBufferParam(shader, kCacheVolumesHierarchical, "_F", F);
            asyncCb.SetComputeBufferParam(shader, kCacheVolumesHierarchical, "_ParentIndex", parentIndex);
            asyncCb.SetComputeBufferParam(shader, kCacheVolumesHierarchical, "_CurrentVolumeBits", currentVolumeBits);
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

            // DT neighbor buffers
            asyncCb.SetComputeBufferParam(shader, kCacheKernelH, "_DtNeighbors", dtLevel.NeighborsBuffer);
            asyncCb.SetComputeBufferParam(shader, kCacheKernelH, "_DtNeighborCounts", dtLevel.NeighborCountsBuffer);

            asyncCb.SetComputeBufferParam(shader, kComputeCorrectionL, "_DtNeighbors", dtLevel.NeighborsBuffer);
            asyncCb.SetComputeBufferParam(shader, kComputeCorrectionL, "_DtNeighborCounts", dtLevel.NeighborCountsBuffer);

            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_DtNeighbors", dtLevel.NeighborsBuffer);
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_DtNeighborCounts", dtLevel.NeighborCountsBuffer);

            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_DtNeighbors", dtLevel.NeighborsBuffer);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_DtNeighborCounts", dtLevel.NeighborCountsBuffer);
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
            initialized = false;
            initializedCount = -1;
            parentsBuilt = false;
        }
    }
}