using System;
using GPU.Delaunay;
using Unity.Mathematics;
using UnityEngine;

namespace GPU.Solver {
    public sealed class XPBISolver : IDisposable {
        const uint FixedFlag = 1u;

        const int ColoringMaxColors = 64;
        const int ColoringConflictRounds = 24;

        [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
        public struct ForceEvent {
            public int node;
            public float2 force;
        }

        readonly ComputeShader shader;

        ComputeBuffer pos;
        ComputeBuffer vel;
        ComputeBuffer invMass;
        ComputeBuffer flags;
        ComputeBuffer restVolume;
        ComputeBuffer parentIndex;
        ComputeBuffer F;
        ComputeBuffer Fp;

        ComputeBuffer currentVolumeBits;
        ComputeBuffer kernelH;
        ComputeBuffer L;
        ComputeBuffer F0;
        ComputeBuffer lambda;

        ComputeBuffer savedVelPrefix;
        ComputeBuffer velDeltaBits;

        ComputeBuffer coloringColor;
        ComputeBuffer coloringProposed;
        ComputeBuffer coloringPrio;

        ComputeBuffer[] colorOrders;
        ComputeBuffer[] colorCountsByLevel;
        ComputeBuffer[] colorStartsByLevel;
        ComputeBuffer[] colorWriteByLevel;
        ComputeBuffer[] relaxArgsByLevel;

        int[] cachedActiveCountByLevel;
        uint[] cachedAdjacencyVersionByLevel;

        float2[] posCpu;
        float2[] velCpu;
        float[] invMassCpu;
        uint[] flagsCpu;
        float[] restVolumeCpu;
        int[] parentIndexCpu;
        float4[] FCpu;
        float4[] FpCpu;

        ComputeBuffer forceEvents;
        ForceEvent[] forceEventsCpu;
        int forceEventsCapacity;
        int forceEventsCount;

        int capacity;
        bool loggedKernelError;

        bool initialized;
        int initializedCount = -1;

        bool parentsBuilt;

        public ComputeBuffer PositionBuffer => pos;

        public XPBISolver(ComputeShader shader) {
            this.shader = shader ? shader : throw new ArgumentNullException(nameof(shader));
        }

        public void Dispose() {
            Release();
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

            posCpu = new float2[capacity];
            velCpu = new float2[capacity];
            invMassCpu = new float[capacity];
            flagsCpu = new uint[capacity];
            restVolumeCpu = new float[capacity];
            parentIndexCpu = new int[capacity];
            FCpu = new float4[capacity];
            FpCpu = new float4[capacity];

            colorOrders = null;
            colorCountsByLevel = null;
            colorStartsByLevel = null;
            colorWriteByLevel = null;
            relaxArgsByLevel = null;

            cachedActiveCountByLevel = null;
            cachedAdjacencyVersionByLevel = null;

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

        public void InitializeFromMeshless(Meshless m) {
            int n = m.nodes.Count;
            EnsureCapacity(n);

            for (int i = 0; i < n; i++) {
                var node = m.nodes[i];
                posCpu[i] = node.pos;
                velCpu[i] = node.vel;
                invMassCpu[i] = node.invMass;
                flagsCpu[i] = node.isFixed || node.invMass <= 0f ? FixedFlag : 0u;
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

            initialized = true;
            initializedCount = n;
            parentsBuilt = false;
        }

        public void SetGameplayForces(ForceEvent[] events, int count) {
            if (events == null) throw new ArgumentNullException(nameof(events));
            if (count < 0 || count > events.Length) throw new ArgumentOutOfRangeException(nameof(count));

            forceEventsCount = count;
            if (forceEventsCount <= 0) return;

            EnsureForceEventCapacity(forceEventsCount);

            Array.Copy(events, 0, forceEventsCpu, 0, forceEventsCount);
            forceEvents.SetData(forceEventsCpu, 0, 0, forceEventsCount);
        }

        public void ClearGameplayForces() {
            forceEventsCount = 0;
        }

        void EnsurePerLevelCaches(int levelCount) {
            if (levelCount <= 0) return;

            if (colorOrders == null || colorOrders.Length < levelCount) {
                var next = new ComputeBuffer[levelCount];
                if (colorOrders != null) Array.Copy(colorOrders, next, colorOrders.Length);
                colorOrders = next;
            }

            if (colorCountsByLevel == null || colorCountsByLevel.Length < levelCount) {
                var next = new ComputeBuffer[levelCount];
                if (colorCountsByLevel != null) Array.Copy(colorCountsByLevel, next, colorCountsByLevel.Length);
                colorCountsByLevel = next;
            }

            if (colorStartsByLevel == null || colorStartsByLevel.Length < levelCount) {
                var next = new ComputeBuffer[levelCount];
                if (colorStartsByLevel != null) Array.Copy(colorStartsByLevel, next, colorStartsByLevel.Length);
                colorStartsByLevel = next;
            }

            if (colorWriteByLevel == null || colorWriteByLevel.Length < levelCount) {
                var next = new ComputeBuffer[levelCount];
                if (colorWriteByLevel != null) Array.Copy(colorWriteByLevel, next, colorWriteByLevel.Length);
                colorWriteByLevel = next;
            }

            if (relaxArgsByLevel == null || relaxArgsByLevel.Length < levelCount) {
                var next = new ComputeBuffer[levelCount];
                if (relaxArgsByLevel != null) Array.Copy(relaxArgsByLevel, next, relaxArgsByLevel.Length);
                relaxArgsByLevel = next;
            }

            if (cachedActiveCountByLevel == null || cachedActiveCountByLevel.Length < levelCount) {
                cachedActiveCountByLevel = new int[levelCount];
                cachedAdjacencyVersionByLevel = new uint[levelCount];
            } else if (cachedActiveCountByLevel.Length < levelCount) {
                Array.Resize(ref cachedActiveCountByLevel, levelCount);
                Array.Resize(ref cachedAdjacencyVersionByLevel, levelCount);
            }
        }

        ComputeBuffer EnsureColorOrderBufferForLevel(int level) {
            var buf = colorOrders[level];
            if (buf != null && buf.count >= capacity) return buf;

            buf?.Dispose();
            buf = new ComputeBuffer(capacity, sizeof(int), ComputeBufferType.Structured);
            colorOrders[level] = buf;
            return buf;
        }

        ComputeBuffer EnsureColorCountsBufferForLevel(int level) {
            var buf = colorCountsByLevel[level];
            if (buf != null && buf.count >= ColoringMaxColors) return buf;

            buf?.Dispose();
            buf = new ComputeBuffer(ColoringMaxColors, sizeof(int), ComputeBufferType.Structured);
            colorCountsByLevel[level] = buf;
            return buf;
        }

        ComputeBuffer EnsureColorStartsBufferForLevel(int level) {
            var buf = colorStartsByLevel[level];
            if (buf != null && buf.count >= ColoringMaxColors) return buf;

            buf?.Dispose();
            buf = new ComputeBuffer(ColoringMaxColors, sizeof(int), ComputeBufferType.Structured);
            colorStartsByLevel[level] = buf;
            return buf;
        }

        ComputeBuffer EnsureColorWriteBufferForLevel(int level) {
            var buf = colorWriteByLevel[level];
            if (buf != null && buf.count >= ColoringMaxColors) return buf;

            buf?.Dispose();
            buf = new ComputeBuffer(ColoringMaxColors, sizeof(int), ComputeBufferType.Structured);
            colorWriteByLevel[level] = buf;
            return buf;
        }

        ComputeBuffer EnsureRelaxArgsBufferForLevel(int level) {
            var buf = relaxArgsByLevel[level];
            if (buf != null && buf.count >= ColoringMaxColors * 3) return buf;

            buf?.Dispose();
            buf = new ComputeBuffer(ColoringMaxColors * 3, sizeof(uint), ComputeBufferType.IndirectArguments);
            relaxArgsByLevel[level] = buf;
            return buf;
        }

        bool HasAllKernels() {
            return
                shader.HasKernel("ClearCurrentVolume") &&
                shader.HasKernel("CacheVolumesHierarchical") &&
                shader.HasKernel("CacheKernelH") &&
                shader.HasKernel("ComputeCorrectionL") &&
                shader.HasKernel("CacheF0AndResetLambda") &&
                shader.HasKernel("SaveVelPrefix") &&
                shader.HasKernel("ClearVelDelta") &&
                shader.HasKernel("RelaxColored") &&
                shader.HasKernel("Prolongate") &&
                shader.HasKernel("CommitDeformation") &&
                shader.HasKernel("ExternalForces") &&

                shader.HasKernel("ApplyGameplayForces") &&
                shader.HasKernel("IntegratePositions") &&
                shader.HasKernel("UpdateDtPositions") &&
                shader.HasKernel("RebuildParentsAtLevel") &&

                shader.HasKernel("ColoringInit") &&
                shader.HasKernel("ColoringDetectConflicts") &&
                shader.HasKernel("ColoringApply") &&
                shader.HasKernel("ColoringChoose") &&
                shader.HasKernel("ColoringClearMeta") &&
                shader.HasKernel("ColoringBuildCounts") &&
                shader.HasKernel("ColoringBuildStarts") &&
                shader.HasKernel("ColoringScatterOrder") &&
                shader.HasKernel("ColoringBuildRelaxArgs");
        }

        public void StepGpuTruth(
            Meshless m,
            float dt,
            bool useHierarchical,
            int dtFixIterations,
            int dtLegalizeIterations,
            bool rebuildParents
        ) {
            if (!HasAllKernels()) {
                if (!loggedKernelError) {
                    loggedKernelError = true;
                    Debug.LogError(
                        $"XPBIGpuSolver: Compute shader '{shader.name}' is missing kernels or failed to compile. " +
                        "Check Console for shader compilation errors from XPBISolver.compute (often bad #include path / HLSL compile error).");
                }
                return;
            }

            int total = m.nodes.Count;
            if (total == 0) return;

            EnsureCapacity(total);
            if (!initialized || initializedCount != total)
                InitializeFromMeshless(m);

            int kApplyGameplayForces = shader.FindKernel("ApplyGameplayForces");
            int kExternalForces = shader.FindKernel("ExternalForces");

            int kClearCurrentVolume = shader.FindKernel("ClearCurrentVolume");
            int kCacheVolumesHierarchical = shader.FindKernel("CacheVolumesHierarchical");
            int kCacheKernelH = shader.FindKernel("CacheKernelH");
            int kComputeCorrectionL = shader.FindKernel("ComputeCorrectionL");
            int kCacheF0AndResetLambda = shader.FindKernel("CacheF0AndResetLambda");
            int kSaveVelPrefix = shader.FindKernel("SaveVelPrefix");
            int kClearVelDelta = shader.FindKernel("ClearVelDelta");
            int kRelaxColored = shader.FindKernel("RelaxColored");
            int kProlongate = shader.FindKernel("Prolongate");
            int kCommitDeformation = shader.FindKernel("CommitDeformation");

            int kIntegratePositions = shader.FindKernel("IntegratePositions");
            int kUpdateDtPositions = shader.FindKernel("UpdateDtPositions");
            int kRebuildParentsAtLevel = shader.FindKernel("RebuildParentsAtLevel");

            int kColoringInit = shader.FindKernel("ColoringInit");
            int kColoringDetectConflicts = shader.FindKernel("ColoringDetectConflicts");
            int kColoringApply = shader.FindKernel("ColoringApply");
            int kColoringChoose = shader.FindKernel("ColoringChoose");
            int kColoringClearMeta = shader.FindKernel("ColoringClearMeta");
            int kColoringBuildCounts = shader.FindKernel("ColoringBuildCounts");
            int kColoringBuildStarts = shader.FindKernel("ColoringBuildStarts");
            int kColoringScatterOrder = shader.FindKernel("ColoringScatterOrder");
            int kColoringBuildRelaxArgs = shader.FindKernel("ColoringBuildRelaxArgs");

            shader.SetInt("_Base", 0);
            shader.SetInt("_TotalCount", total);
            shader.SetFloat("_Dt", dt);
            shader.SetFloat("_Gravity", m.gravity);
            shader.SetFloat("_Compliance", m.compliance);

            bool rebuildAllParents = useHierarchical && m.maxLayer > 0 && (rebuildParents || !parentsBuilt);
            if (rebuildAllParents) {
                for (int level = m.maxLayer; level >= 1; level--) {
                    if (!m.TryGetLevelDt(level, out DT dtLevel) || dtLevel == null)
                        continue;

                    int activeCount = m.NodeCount(level);
                    int fineCount = level > 1 ? m.NodeCount(level - 1) : total;
                    if (fineCount <= activeCount)
                        continue;

                    shader.SetInt("_DtNeighborCount", dtLevel.NeighborCount);

                    shader.SetInt("_ParentRangeStart", activeCount);
                    shader.SetInt("_ParentRangeEnd", fineCount);
                    shader.SetInt("_ParentCoarseCount", activeCount);

                    shader.SetBuffer(kRebuildParentsAtLevel, "_Pos", pos);
                    shader.SetBuffer(kRebuildParentsAtLevel, "_ParentIndex", parentIndex);
                    shader.SetBuffer(kRebuildParentsAtLevel, "_DtNeighbors", dtLevel.NeighborsBuffer);
                    shader.SetBuffer(kRebuildParentsAtLevel, "_DtNeighborCounts", dtLevel.NeighborCountsBuffer);

                    shader.Dispatch(kRebuildParentsAtLevel, ((fineCount - activeCount) + 255) / 256, 1, 1);
                }

                parentsBuilt = true;
            }

            // Force stage.
            shader.SetBuffer(kApplyGameplayForces, "_Vel", vel);
            shader.SetBuffer(kApplyGameplayForces, "_InvMass", invMass);
            shader.SetBuffer(kApplyGameplayForces, "_Flags", flags);

            shader.SetBuffer(kExternalForces, "_Vel", vel);
            shader.SetBuffer(kExternalForces, "_InvMass", invMass);
            shader.SetBuffer(kExternalForces, "_Flags", flags);

            if (forceEventsCount > 0) {
                shader.SetInt("_ForceEventCount", forceEventsCount);
                shader.SetBuffer(kApplyGameplayForces, "_ForceEvents", forceEvents);
                shader.Dispatch(kApplyGameplayForces, (forceEventsCount + 255) / 256, 1, 1);
                forceEventsCount = 0;
            } else {
                shader.SetInt("_ForceEventCount", 0);
            }

            shader.Dispatch(kExternalForces, (total + 255) / 256, 1, 1);

            int maxSolveLevel = (useHierarchical && m.levelEndIndex != null) ? m.maxLayer : 0;
            EnsurePerLevelCaches(maxSolveLevel + 1);

            for (int level = maxSolveLevel; level >= 0; level--) {
                if (!m.TryGetLevelDt(level, out DT dtLevel))
                    continue;

                int activeCount = level > 0 ? m.NodeCount(level) : total;
                if (activeCount < 3) continue;

                int fineCount = level > 1 ? m.NodeCount(level - 1) : total;

                shader.SetInt("_ActiveCount", activeCount);
                shader.SetInt("_FineCount", fineCount);

                int neighborCount = dtLevel.NeighborCount;
                shader.SetInt("_DtNeighborCount", neighborCount);

                shader.SetBuffer(kClearCurrentVolume, "_CurrentVolumeBits", currentVolumeBits);

                shader.SetBuffer(kCacheVolumesHierarchical, "_RestVolume", restVolume);
                shader.SetBuffer(kCacheVolumesHierarchical, "_F", F);
                shader.SetBuffer(kCacheVolumesHierarchical, "_ParentIndex", parentIndex);
                shader.SetBuffer(kCacheVolumesHierarchical, "_CurrentVolumeBits", currentVolumeBits);

                shader.SetBuffer(kCacheKernelH, "_Pos", pos);
                shader.SetBuffer(kCacheKernelH, "_KernelH", kernelH);

                shader.SetBuffer(kComputeCorrectionL, "_Pos", pos);
                shader.SetBuffer(kComputeCorrectionL, "_KernelH", kernelH);
                shader.SetBuffer(kComputeCorrectionL, "_CurrentVolumeBits", currentVolumeBits);
                shader.SetBuffer(kComputeCorrectionL, "_L", L);

                shader.SetBuffer(kCacheF0AndResetLambda, "_F", F);
                shader.SetBuffer(kCacheF0AndResetLambda, "_F0", F0);
                shader.SetBuffer(kCacheF0AndResetLambda, "_Lambda", lambda);

                shader.SetBuffer(kSaveVelPrefix, "_Vel", vel);
                shader.SetBuffer(kSaveVelPrefix, "_SavedVelPrefix", savedVelPrefix);

                shader.SetBuffer(kClearVelDelta, "_VelDeltaBits", velDeltaBits);

                shader.SetBuffer(kRelaxColored, "_Pos", pos);
                shader.SetBuffer(kRelaxColored, "_Vel", vel);
                shader.SetBuffer(kRelaxColored, "_InvMass", invMass);
                shader.SetBuffer(kRelaxColored, "_Flags", flags);
                shader.SetBuffer(kRelaxColored, "_RestVolume", restVolume);
                shader.SetBuffer(kRelaxColored, "_F0", F0);
                shader.SetBuffer(kRelaxColored, "_L", L);
                shader.SetBuffer(kRelaxColored, "_KernelH", kernelH);
                shader.SetBuffer(kRelaxColored, "_CurrentVolumeBits", currentVolumeBits);
                shader.SetBuffer(kRelaxColored, "_Lambda", lambda);

                shader.SetBuffer(kProlongate, "_Vel", vel);
                shader.SetBuffer(kProlongate, "_ParentIndex", parentIndex);
                shader.SetBuffer(kProlongate, "_SavedVelPrefix", savedVelPrefix);

                shader.SetBuffer(kCommitDeformation, "_Pos", pos);
                shader.SetBuffer(kCommitDeformation, "_Vel", vel);
                shader.SetBuffer(kCommitDeformation, "_InvMass", invMass);
                shader.SetBuffer(kCommitDeformation, "_Flags", flags);
                shader.SetBuffer(kCommitDeformation, "_F0", F0);
                shader.SetBuffer(kCommitDeformation, "_L", L);
                shader.SetBuffer(kCommitDeformation, "_KernelH", kernelH);
                shader.SetBuffer(kCommitDeformation, "_CurrentVolumeBits", currentVolumeBits);
                shader.SetBuffer(kCommitDeformation, "_F", F);
                shader.SetBuffer(kCommitDeformation, "_Fp", Fp);

                shader.SetBuffer(kCacheKernelH, "_DtNeighbors", dtLevel.NeighborsBuffer);
                shader.SetBuffer(kCacheKernelH, "_DtNeighborCounts", dtLevel.NeighborCountsBuffer);

                shader.SetBuffer(kComputeCorrectionL, "_DtNeighbors", dtLevel.NeighborsBuffer);
                shader.SetBuffer(kComputeCorrectionL, "_DtNeighborCounts", dtLevel.NeighborCountsBuffer);

                shader.SetBuffer(kRelaxColored, "_DtNeighbors", dtLevel.NeighborsBuffer);
                shader.SetBuffer(kRelaxColored, "_DtNeighborCounts", dtLevel.NeighborCountsBuffer);

                shader.SetBuffer(kCommitDeformation, "_DtNeighbors", dtLevel.NeighborsBuffer);
                shader.SetBuffer(kCommitDeformation, "_DtNeighborCounts", dtLevel.NeighborCountsBuffer);

                shader.Dispatch(kSaveVelPrefix, (activeCount + 255) / 256, 1, 1);

                shader.Dispatch(kClearCurrentVolume, (activeCount + 255) / 256, 1, 1);
                shader.Dispatch(kCacheVolumesHierarchical, (total + 255) / 256, 1, 1);

                shader.Dispatch(kCacheKernelH, (activeCount + 255) / 256, 1, 1);
                shader.Dispatch(kComputeCorrectionL, (activeCount + 255) / 256, 1, 1);
                shader.Dispatch(kCacheF0AndResetLambda, (activeCount + 255) / 256, 1, 1);

                uint adjacencyVersion = dtLevel.AdjacencyVersion;
                bool rebuildColors =
                    cachedActiveCountByLevel[level] != activeCount ||
                    cachedAdjacencyVersionByLevel[level] != adjacencyVersion ||
                    colorOrders[level] == null ||
                    colorCountsByLevel[level] == null ||
                    colorStartsByLevel[level] == null ||
                    relaxArgsByLevel[level] == null;

                var orderBuf = EnsureColorOrderBufferForLevel(level);
                var countsBuf = EnsureColorCountsBufferForLevel(level);
                var startsBuf = EnsureColorStartsBufferForLevel(level);
                var writeBuf = EnsureColorWriteBufferForLevel(level);
                var relaxArgsBuf = EnsureRelaxArgsBufferForLevel(level);

                if (rebuildColors) {
                    uint seed = adjacencyVersion ^ (uint)(level * 2654435761);

                    shader.SetInt("_ColoringActiveCount", activeCount);
                    shader.SetInt("_ColoringDtNeighborCount", neighborCount);
                    shader.SetInt("_ColoringMaxColors", ColoringMaxColors);
                    shader.SetInt("_ColoringSeed", unchecked((int)seed));

                    shader.SetBuffer(kColoringInit, "_ColoringColor", coloringColor);
                    shader.SetBuffer(kColoringInit, "_ColoringProposed", coloringProposed);
                    shader.SetBuffer(kColoringInit, "_ColoringPrio", coloringPrio);

                    shader.SetBuffer(kColoringDetectConflicts, "_ColoringColor", coloringColor);
                    shader.SetBuffer(kColoringDetectConflicts, "_ColoringProposed", coloringProposed);
                    shader.SetBuffer(kColoringDetectConflicts, "_ColoringPrio", coloringPrio);

                    shader.SetBuffer(kColoringApply, "_ColoringColor", coloringColor);
                    shader.SetBuffer(kColoringApply, "_ColoringProposed", coloringProposed);

                    shader.SetBuffer(kColoringChoose, "_ColoringColor", coloringColor);
                    shader.SetBuffer(kColoringChoose, "_ColoringProposed", coloringProposed);

                    shader.SetBuffer(kColoringInit, "_ColoringDtNeighbors", dtLevel.NeighborsBuffer);
                    shader.SetBuffer(kColoringInit, "_ColoringDtNeighborCounts", dtLevel.NeighborCountsBuffer);

                    shader.SetBuffer(kColoringDetectConflicts, "_ColoringDtNeighbors", dtLevel.NeighborsBuffer);
                    shader.SetBuffer(kColoringDetectConflicts, "_ColoringDtNeighborCounts", dtLevel.NeighborCountsBuffer);

                    shader.SetBuffer(kColoringChoose, "_ColoringDtNeighbors", dtLevel.NeighborsBuffer);
                    shader.SetBuffer(kColoringChoose, "_ColoringDtNeighborCounts", dtLevel.NeighborCountsBuffer);

                    shader.SetBuffer(kColoringClearMeta, "_ColoringCounts", countsBuf);
                    shader.SetBuffer(kColoringClearMeta, "_ColoringStarts", startsBuf);
                    shader.SetBuffer(kColoringClearMeta, "_ColoringWrite", writeBuf);

                    shader.SetBuffer(kColoringBuildCounts, "_ColoringColor", coloringColor);
                    shader.SetBuffer(kColoringBuildCounts, "_ColoringCounts", countsBuf);

                    shader.SetBuffer(kColoringBuildStarts, "_ColoringCounts", countsBuf);
                    shader.SetBuffer(kColoringBuildStarts, "_ColoringStarts", startsBuf);
                    shader.SetBuffer(kColoringBuildStarts, "_ColoringWrite", writeBuf);

                    shader.SetBuffer(kColoringScatterOrder, "_ColoringColor", coloringColor);
                    shader.SetBuffer(kColoringScatterOrder, "_ColoringWrite", writeBuf);
                    shader.SetBuffer(kColoringScatterOrder, "_ColoringOrderOut", orderBuf);

                    shader.SetBuffer(kColoringBuildRelaxArgs, "_ColoringCounts", countsBuf);
                    shader.SetBuffer(kColoringBuildRelaxArgs, "_RelaxArgs", relaxArgsBuf);

                    int groups = (activeCount + 255) / 256;

                    shader.Dispatch(kColoringInit, groups, 1, 1);

                    for (int r = 0; r < ColoringConflictRounds; r++) {
                        shader.Dispatch(kColoringDetectConflicts, groups, 1, 1);
                        shader.Dispatch(kColoringApply, groups, 1, 1);

                        shader.Dispatch(kColoringChoose, groups, 1, 1);
                        shader.Dispatch(kColoringApply, groups, 1, 1);
                    }

                    shader.Dispatch(kColoringClearMeta, 1, 1, 1);
                    shader.Dispatch(kColoringBuildCounts, groups, 1, 1);
                    shader.Dispatch(kColoringBuildStarts, 1, 1, 1);
                    shader.Dispatch(kColoringScatterOrder, groups, 1, 1);
                    shader.Dispatch(kColoringBuildRelaxArgs, 1, 1, 1);

                    cachedActiveCountByLevel[level] = activeCount;
                    cachedAdjacencyVersionByLevel[level] = adjacencyVersion;
                }

                shader.SetBuffer(kRelaxColored, "_ColorOrder", orderBuf);
                shader.SetBuffer(kRelaxColored, "_ColorCounts", countsBuf);
                shader.SetBuffer(kRelaxColored, "_ColorStarts", startsBuf);

                int iterations = level == 0 ? Const.Iterations : Const.HPBDIterations;

                for (int iter = 0; iter < iterations; iter++) {
                    for (int c = 0; c < ColoringMaxColors; c++) {
                        shader.SetInt("_ColorIndex", c);
                        shader.DispatchIndirect(kRelaxColored, relaxArgsBuf, (uint)c * 12);
                    }
                }

                if (level > 0 && fineCount > activeCount) {
                    shader.Dispatch(kProlongate, ((fineCount - activeCount) + 255) / 256, 1, 1);
                }

                if (level == 0) {
                    shader.Dispatch(kCommitDeformation, (activeCount + 255) / 256, 1, 1);
                }
            }

            shader.SetBuffer(kIntegratePositions, "_Pos", pos);
            shader.SetBuffer(kIntegratePositions, "_Vel", vel);
            shader.SetBuffer(kIntegratePositions, "_InvMass", invMass);
            shader.SetBuffer(kIntegratePositions, "_Flags", flags);
            shader.Dispatch(kIntegratePositions, (total + 255) / 256, 1, 1);

            int maxDtLevel = m.maxLayer;
            for (int level = maxDtLevel; level >= 0; level--) {
                if (!m.TryGetLevelDt(level, out DT dtLevel))
                    continue;

                int activeCount = (level > 0) ? m.NodeCount(level) : total;
                if (activeCount < 3) continue;

                int fineCount = (level > 1) ? m.NodeCount(level - 1) : total;

                shader.SetInt("_ActiveCount", activeCount);
                shader.SetInt("_FineCount", fineCount);
                shader.SetInt("_DtNeighborCount", dtLevel.NeighborCount);

                shader.SetBuffer(kUpdateDtPositions, "_Pos", pos);
                shader.SetBuffer(kUpdateDtPositions, "_DtPositions", dtLevel.PositionsWriteBuffer);
                shader.SetVector("_DtNormCenter", new Vector4(m.DtNormCenter.x, m.DtNormCenter.y, 0f, 0f));
                shader.SetFloat("_DtNormInvHalfExtent", m.DtNormInvHalfExtent);
                shader.Dispatch(kUpdateDtPositions, (activeCount + 255) / 256, 1, 1);

                dtLevel.BindPositionsForMaintain(dtLevel.PositionsWriteBuffer);
                dtLevel.Maintain(dtFixIterations, dtLegalizeIterations);
                dtLevel.SwapPositionsAfterTick();
            }
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

            if (colorOrders != null) {
                for (int i = 0; i < colorOrders.Length; i++) {
                    colorOrders[i]?.Dispose();
                    colorOrders[i] = null;
                }
                colorOrders = null;
            }

            if (colorCountsByLevel != null) {
                for (int i = 0; i < colorCountsByLevel.Length; i++) {
                    colorCountsByLevel[i]?.Dispose();
                    colorCountsByLevel[i] = null;
                }
                colorCountsByLevel = null;
            }

            if (colorStartsByLevel != null) {
                for (int i = 0; i < colorStartsByLevel.Length; i++) {
                    colorStartsByLevel[i]?.Dispose();
                    colorStartsByLevel[i] = null;
                }
                colorStartsByLevel = null;
            }

            if (colorWriteByLevel != null) {
                for (int i = 0; i < colorWriteByLevel.Length; i++) {
                    colorWriteByLevel[i]?.Dispose();
                    colorWriteByLevel[i] = null;
                }
                colorWriteByLevel = null;
            }

            if (relaxArgsByLevel != null) {
                for (int i = 0; i < relaxArgsByLevel.Length; i++) {
                    relaxArgsByLevel[i]?.Dispose();
                    relaxArgsByLevel[i] = null;
                }
                relaxArgsByLevel = null;
            }

            initialized = false;
            initializedCount = -1;
            parentsBuilt = false;
        }

        void Release() {
            ReleaseBuffers();
            capacity = 0;

            posCpu = null;
            velCpu = null;
            invMassCpu = null;
            flagsCpu = null;
            restVolumeCpu = null;
            parentIndexCpu = null;
            FCpu = null;
            FpCpu = null;

            forceEventsCpu = null;
            forceEventsCapacity = 0;
            forceEventsCount = 0;

            cachedActiveCountByLevel = null;
            cachedAdjacencyVersionByLevel = null;
        }
    }
}
