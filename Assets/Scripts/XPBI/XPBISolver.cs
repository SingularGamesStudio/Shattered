using System;
using GPU.Delaunay;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace GPU.Solver {
    public sealed class XPBISolver : IDisposable {
        const uint FixedFlag = 1u;

        const int ColoringMaxColors = 64;
        const int ColoringConflictRounds = 24;

        [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
        public struct ForceEvent {
            public uint node;
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

        // Cached kernel IDs (avoid FindKernel per tick).
        bool kernelsCached;

        int kApplyGameplayForces;
        int kExternalForces;

        int kClearCurrentVolume;
        int kCacheVolumesHierarchical;
        int kCacheKernelH;
        int kComputeCorrectionL;
        int kCacheF0AndResetLambda;
        int kSaveVelPrefix;
        int kClearVelDelta;
        int kRelaxColored;
        int kProlongate;
        int kCommitDeformation;

        int kIntegratePositions;
        int kUpdateDtPositions;
        int kRebuildParentsAtLevel;

        int kColoringInit;
        int kColoringDetectConflicts;
        int kColoringApply;
        int kColoringChoose;
        int kColoringClearMeta;
        int kColoringBuildCounts;
        int kColoringBuildStarts;
        int kColoringScatterOrder;
        int kColoringBuildRelaxArgs;

        CommandBuffer asyncCb;

        public ComputeBuffer PositionBuffer => pos;

        public XPBISolver(ComputeShader shader) {
            this.shader = shader ? shader : throw new ArgumentNullException(nameof(shader));
        }

        public void Dispose() {
            Release();
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

        void EnsureKernelsCached() {
            if (kernelsCached) return;

            kApplyGameplayForces = shader.FindKernel("ApplyGameplayForces");
            kExternalForces = shader.FindKernel("ExternalForces");

            kClearCurrentVolume = shader.FindKernel("ClearCurrentVolume");
            kCacheVolumesHierarchical = shader.FindKernel("CacheVolumesHierarchical");
            kCacheKernelH = shader.FindKernel("CacheKernelH");
            kComputeCorrectionL = shader.FindKernel("ComputeCorrectionL");
            kCacheF0AndResetLambda = shader.FindKernel("CacheF0AndResetLambda");
            kSaveVelPrefix = shader.FindKernel("SaveVelPrefix");
            kClearVelDelta = shader.FindKernel("ClearVelDelta");
            kRelaxColored = shader.FindKernel("RelaxColored");
            kProlongate = shader.FindKernel("Prolongate");
            kCommitDeformation = shader.FindKernel("CommitDeformation");

            kIntegratePositions = shader.FindKernel("IntegratePositions");
            kUpdateDtPositions = shader.FindKernel("UpdateDtPositions");
            kRebuildParentsAtLevel = shader.FindKernel("RebuildParentsAtLevel");

            kColoringInit = shader.FindKernel("ColoringInit");
            kColoringDetectConflicts = shader.FindKernel("ColoringDetectConflicts");
            kColoringApply = shader.FindKernel("ColoringApply");
            kColoringChoose = shader.FindKernel("ColoringChoose");
            kColoringClearMeta = shader.FindKernel("ColoringClearMeta");
            kColoringBuildCounts = shader.FindKernel("ColoringBuildCounts");
            kColoringBuildStarts = shader.FindKernel("ColoringBuildStarts");
            kColoringScatterOrder = shader.FindKernel("ColoringScatterOrder");
            kColoringBuildRelaxArgs = shader.FindKernel("ColoringBuildRelaxArgs");

            kernelsCached = true;
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
            buf = new ComputeBuffer(capacity, sizeof(uint), ComputeBufferType.Structured);  // changed to uint
            colorOrders[level] = buf;
            return buf;
        }

        ComputeBuffer EnsureColorCountsBufferForLevel(int level) {
            var buf = colorCountsByLevel[level];
            if (buf != null && buf.count >= ColoringMaxColors) return buf;

            buf?.Dispose();
            buf = new ComputeBuffer(ColoringMaxColors, sizeof(uint), ComputeBufferType.Structured);  // changed to uint
            colorCountsByLevel[level] = buf;
            return buf;
        }

        ComputeBuffer EnsureColorStartsBufferForLevel(int level) {
            var buf = colorStartsByLevel[level];
            if (buf != null && buf.count >= ColoringMaxColors) return buf;

            buf?.Dispose();
            buf = new ComputeBuffer(ColoringMaxColors, sizeof(uint), ComputeBufferType.Structured);  // changed to uint
            colorStartsByLevel[level] = buf;
            return buf;
        }

        ComputeBuffer EnsureColorWriteBufferForLevel(int level) {
            var buf = colorWriteByLevel[level];
            if (buf != null && buf.count >= ColoringMaxColors) return buf;

            buf?.Dispose();
            buf = new ComputeBuffer(ColoringMaxColors, sizeof(uint), ComputeBufferType.Structured);  // changed to uint
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

        public void CompleteDtSwapAfterAsync(Meshless m, int maxLevelToSwap) {
            if (m == null) return;

            int max = math.max(0, math.min(m.maxLayer, maxLevelToSwap));
            for (int level = max; level >= 0; level--) {
                if (!m.TryGetLevelDt(level, out DT dtLevel) || dtLevel == null)
                    continue;

                dtLevel.SwapTopologyAfterTick();
                dtLevel.SwapPositionsAfterTick();
            }
        }


        public GraphicsFence SubmitGpuTruthAsyncBatch(
    Meshless m,
    float dtPerTick,
    int tickCount,
    bool useHierarchical,
    int dtFixIterations,
    int dtLegalizeIterations,
    bool rebuildParents,
    bool updateDtPositionsForRender,
    int dtSwapMaxLevel,
    bool runDtMaintain,
    ComputeQueueType queueType
) {
            if (!HasAllKernels()) {
                if (!loggedKernelError) {
                    loggedKernelError = true;
                    Debug.LogError(
                        $"XPBIGpuSolver: Compute shader '{shader.name}' is missing kernels or failed to compile. " +
                        "Check Console for shader compilation errors from XPBISolver.compute (often bad #include path / HLSL compile error).");
                }
                return default;
            }

            EnsureKernelsCached();

            int total = m.nodes.Count;
            if (total == 0 || tickCount <= 0)
                return default;

            EnsureCapacity(total);
            if (!initialized || initializedCount != total)
                InitializeFromMeshless(m);

            asyncCb ??= new CommandBuffer { name = "XPBI Async Batch" };
            asyncCb.Clear();
            asyncCb.SetExecutionFlags(CommandBufferExecutionFlags.AsyncCompute);

            int maxSolveLevel = (useHierarchical && m.levelEndIndex != null) ? m.maxLayer : 0;
            EnsurePerLevelCaches(maxSolveLevel + 1);

            // In the "fully integrated DT" path, we update DT positions for all levels whenever
            // rendering wants them; this fixes coarse-level wireframe freezing.
            int maxDtLevel = (updateDtPositionsForRender || runDtMaintain) ? m.maxLayer : -1;

            for (int tick = 0; tick < tickCount; tick++) {
                asyncCb.SetComputeIntParam(shader, "_Base", 0);
                asyncCb.SetComputeIntParam(shader, "_TotalCount", total);
                asyncCb.SetComputeFloatParam(shader, "_Dt", dtPerTick);
                asyncCb.SetComputeFloatParam(shader, "_Gravity", m.gravity);
                asyncCb.SetComputeFloatParam(shader, "_Compliance", m.compliance);

                bool rebuildAllParents = useHierarchical && m.maxLayer > 0 && (rebuildParents || !parentsBuilt);
                if (rebuildAllParents) {
                    for (int level = m.maxLayer; level >= 1; level--) {
                        if (!m.TryGetLevelDt(level, out DT dtLevel) || dtLevel == null)
                            continue;

                        int activeCount = m.NodeCount(level);
                        int fineCount = level > 1 ? m.NodeCount(level - 1) : total;
                        if (fineCount <= activeCount)
                            continue;

                        int pingRead = dtLevel.RenderPing ^ (maxDtLevel >= 0 ? (tick & 1) : 0);

                        asyncCb.SetComputeIntParam(shader, "_DtNeighborCount", dtLevel.NeighborCount);

                        asyncCb.SetComputeIntParam(shader, "_ParentRangeStart", activeCount);
                        asyncCb.SetComputeIntParam(shader, "_ParentRangeEnd", fineCount);
                        asyncCb.SetComputeIntParam(shader, "_ParentCoarseCount", activeCount);

                        asyncCb.SetComputeBufferParam(shader, kRebuildParentsAtLevel, "_Pos", pos);
                        asyncCb.SetComputeBufferParam(shader, kRebuildParentsAtLevel, "_ParentIndex", parentIndex);
                        asyncCb.SetComputeBufferParam(shader, kRebuildParentsAtLevel, "_DtNeighbors", dtLevel.GetNeighborsBuffer(pingRead));
                        asyncCb.SetComputeBufferParam(shader, kRebuildParentsAtLevel, "_DtNeighborCounts", dtLevel.GetNeighborCountsBuffer(pingRead));

                        asyncCb.DispatchCompute(shader, kRebuildParentsAtLevel, ((fineCount - activeCount) + 255) / 256, 1, 1);
                    }

                    parentsBuilt = true;
                }

                asyncCb.SetComputeBufferParam(shader, kApplyGameplayForces, "_Vel", vel);
                asyncCb.SetComputeBufferParam(shader, kApplyGameplayForces, "_InvMass", invMass);
                asyncCb.SetComputeBufferParam(shader, kApplyGameplayForces, "_Flags", flags);

                asyncCb.SetComputeBufferParam(shader, kExternalForces, "_Vel", vel);
                asyncCb.SetComputeBufferParam(shader, kExternalForces, "_InvMass", invMass);
                asyncCb.SetComputeBufferParam(shader, kExternalForces, "_Flags", flags);

                if (forceEventsCount > 0) {
                    asyncCb.SetComputeIntParam(shader, "_ForceEventCount", forceEventsCount);
                    asyncCb.SetComputeBufferParam(shader, kApplyGameplayForces, "_ForceEvents", forceEvents);
                    asyncCb.DispatchCompute(shader, kApplyGameplayForces, (forceEventsCount + 255) / 256, 1, 1);
                    forceEventsCount = 0;
                } else {
                    asyncCb.SetComputeIntParam(shader, "_ForceEventCount", 0);
                }

                asyncCb.DispatchCompute(shader, kExternalForces, (total + 255) / 256, 1, 1);

                for (int level = maxSolveLevel; level >= 0; level--) {
                    if (!m.TryGetLevelDt(level, out DT dtLevel) || dtLevel == null)
                        continue;

                    int activeCount = level > 0 ? m.NodeCount(level) : total;
                    if (activeCount < 3) continue;

                    int fineCount = level > 1 ? m.NodeCount(level - 1) : total;

                    int pingRead = dtLevel.RenderPing ^ (maxDtLevel >= 0 ? (tick & 1) : 0);
                    uint adjacencyVersion = dtLevel.GetAdjacencyVersion(pingRead);

                    asyncCb.SetComputeIntParam(shader, "_ActiveCount", activeCount);
                    asyncCb.SetComputeIntParam(shader, "_FineCount", fineCount);

                    int neighborCount = dtLevel.NeighborCount;
                    asyncCb.SetComputeIntParam(shader, "_DtNeighborCount", neighborCount);

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

                    asyncCb.SetComputeBufferParam(shader, kCacheKernelH, "_DtNeighbors", dtLevel.GetNeighborsBuffer(pingRead));
                    asyncCb.SetComputeBufferParam(shader, kCacheKernelH, "_DtNeighborCounts", dtLevel.GetNeighborCountsBuffer(pingRead));

                    asyncCb.SetComputeBufferParam(shader, kComputeCorrectionL, "_DtNeighbors", dtLevel.GetNeighborsBuffer(pingRead));
                    asyncCb.SetComputeBufferParam(shader, kComputeCorrectionL, "_DtNeighborCounts", dtLevel.GetNeighborCountsBuffer(pingRead));

                    asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_DtNeighbors", dtLevel.GetNeighborsBuffer(pingRead));
                    asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_DtNeighborCounts", dtLevel.GetNeighborCountsBuffer(pingRead));

                    asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_DtNeighbors", dtLevel.GetNeighborsBuffer(pingRead));
                    asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_DtNeighborCounts", dtLevel.GetNeighborCountsBuffer(pingRead));

                    asyncCb.DispatchCompute(shader, kSaveVelPrefix, (activeCount + 255) / 256, 1, 1);

                    asyncCb.DispatchCompute(shader, kClearCurrentVolume, (activeCount + 255) / 256, 1, 1);
                    asyncCb.DispatchCompute(shader, kCacheVolumesHierarchical, (total + 255) / 256, 1, 1);

                    asyncCb.DispatchCompute(shader, kCacheKernelH, (activeCount + 255) / 256, 1, 1);
                    asyncCb.DispatchCompute(shader, kComputeCorrectionL, (activeCount + 255) / 256, 1, 1);
                    asyncCb.DispatchCompute(shader, kCacheF0AndResetLambda, (activeCount + 255) / 256, 1, 1);

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

                        asyncCb.SetComputeIntParam(shader, "_ColoringActiveCount", activeCount);
                        asyncCb.SetComputeIntParam(shader, "_ColoringDtNeighborCount", neighborCount);
                        asyncCb.SetComputeIntParam(shader, "_ColoringMaxColors", ColoringMaxColors);
                        asyncCb.SetComputeIntParam(shader, "_ColoringSeed", unchecked((int)seed));

                        asyncCb.SetComputeBufferParam(shader, kColoringInit, "_ColoringColor", coloringColor);
                        asyncCb.SetComputeBufferParam(shader, kColoringInit, "_ColoringProposed", coloringProposed);
                        asyncCb.SetComputeBufferParam(shader, kColoringInit, "_ColoringPrio", coloringPrio);

                        asyncCb.SetComputeBufferParam(shader, kColoringDetectConflicts, "_ColoringColor", coloringColor);
                        asyncCb.SetComputeBufferParam(shader, kColoringDetectConflicts, "_ColoringProposed", coloringProposed);
                        asyncCb.SetComputeBufferParam(shader, kColoringDetectConflicts, "_ColoringPrio", coloringPrio);

                        asyncCb.SetComputeBufferParam(shader, kColoringApply, "_ColoringColor", coloringColor);
                        asyncCb.SetComputeBufferParam(shader, kColoringApply, "_ColoringProposed", coloringProposed);

                        asyncCb.SetComputeBufferParam(shader, kColoringChoose, "_ColoringColor", coloringColor);
                        asyncCb.SetComputeBufferParam(shader, kColoringChoose, "_ColoringProposed", coloringProposed);

                        asyncCb.SetComputeBufferParam(shader, kColoringInit, "_ColoringDtNeighbors", dtLevel.GetNeighborsBuffer(pingRead));
                        asyncCb.SetComputeBufferParam(shader, kColoringInit, "_ColoringDtNeighborCounts", dtLevel.GetNeighborCountsBuffer(pingRead));

                        asyncCb.SetComputeBufferParam(shader, kColoringDetectConflicts, "_ColoringDtNeighbors", dtLevel.GetNeighborsBuffer(pingRead));
                        asyncCb.SetComputeBufferParam(shader, kColoringDetectConflicts, "_ColoringDtNeighborCounts", dtLevel.GetNeighborCountsBuffer(pingRead));

                        asyncCb.SetComputeBufferParam(shader, kColoringChoose, "_ColoringDtNeighbors", dtLevel.GetNeighborsBuffer(pingRead));
                        asyncCb.SetComputeBufferParam(shader, kColoringChoose, "_ColoringDtNeighborCounts", dtLevel.GetNeighborCountsBuffer(pingRead));

                        asyncCb.SetComputeBufferParam(shader, kColoringClearMeta, "_ColoringCounts", countsBuf);
                        asyncCb.SetComputeBufferParam(shader, kColoringClearMeta, "_ColoringStarts", startsBuf);
                        asyncCb.SetComputeBufferParam(shader, kColoringClearMeta, "_ColoringWrite", writeBuf);

                        asyncCb.SetComputeBufferParam(shader, kColoringBuildCounts, "_ColoringColor", coloringColor);
                        asyncCb.SetComputeBufferParam(shader, kColoringBuildCounts, "_ColoringCounts", countsBuf);

                        asyncCb.SetComputeBufferParam(shader, kColoringBuildStarts, "_ColoringCounts", countsBuf);
                        asyncCb.SetComputeBufferParam(shader, kColoringBuildStarts, "_ColoringStarts", startsBuf);
                        asyncCb.SetComputeBufferParam(shader, kColoringBuildStarts, "_ColoringWrite", writeBuf);

                        asyncCb.SetComputeBufferParam(shader, kColoringScatterOrder, "_ColoringColor", coloringColor);
                        asyncCb.SetComputeBufferParam(shader, kColoringScatterOrder, "_ColoringWrite", writeBuf);
                        asyncCb.SetComputeBufferParam(shader, kColoringScatterOrder, "_ColoringOrderOut", orderBuf);

                        asyncCb.SetComputeBufferParam(shader, kColoringBuildRelaxArgs, "_ColoringCounts", countsBuf);
                        asyncCb.SetComputeBufferParam(shader, kColoringBuildRelaxArgs, "_RelaxArgs", relaxArgsBuf);

                        int groups = (activeCount + 255) / 256;

                        asyncCb.DispatchCompute(shader, kColoringInit, groups, 1, 1);

                        for (int r = 0; r < ColoringConflictRounds; r++) {
                            asyncCb.DispatchCompute(shader, kColoringDetectConflicts, groups, 1, 1);
                            asyncCb.DispatchCompute(shader, kColoringApply, groups, 1, 1);

                            asyncCb.DispatchCompute(shader, kColoringChoose, groups, 1, 1);
                            asyncCb.DispatchCompute(shader, kColoringApply, groups, 1, 1);
                        }

                        asyncCb.DispatchCompute(shader, kColoringClearMeta, 1, 1, 1);
                        asyncCb.DispatchCompute(shader, kColoringBuildCounts, groups, 1, 1);
                        asyncCb.DispatchCompute(shader, kColoringBuildStarts, 1, 1, 1);
                        asyncCb.DispatchCompute(shader, kColoringScatterOrder, groups, 1, 1);
                        asyncCb.DispatchCompute(shader, kColoringBuildRelaxArgs, 1, 1, 1);

                        cachedActiveCountByLevel[level] = activeCount;
                        cachedAdjacencyVersionByLevel[level] = adjacencyVersion;
                    }

                    asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_ColorOrder", orderBuf);
                    asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_ColorCounts", countsBuf);
                    asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_ColorStarts", startsBuf);

                    int iterations = level == 0 ? Const.Iterations : Const.HPBDIterations;

                    for (int iter = 0; iter < iterations; iter++) {
                        for (int c = 0; c < ColoringMaxColors; c++) {
                            asyncCb.SetComputeIntParam(shader, "_ColorIndex", c);
                            asyncCb.DispatchCompute(shader, kRelaxColored, relaxArgsBuf, (uint)c * 12);
                        }
                    }

                    if (level > 0 && fineCount > activeCount)
                        asyncCb.DispatchCompute(shader, kProlongate, ((fineCount - activeCount) + 255) / 256, 1, 1);

                    if (level == 0)
                        asyncCb.DispatchCompute(shader, kCommitDeformation, (activeCount + 255) / 256, 1, 1);
                }

                asyncCb.SetComputeBufferParam(shader, kIntegratePositions, "_Pos", pos);
                asyncCb.SetComputeBufferParam(shader, kIntegratePositions, "_Vel", vel);
                asyncCb.SetComputeBufferParam(shader, kIntegratePositions, "_InvMass", invMass);
                asyncCb.SetComputeBufferParam(shader, kIntegratePositions, "_Flags", flags);
                asyncCb.DispatchCompute(shader, kIntegratePositions, (total + 255) / 256, 1, 1);

                if (maxDtLevel >= 0) {
                    for (int level = maxDtLevel; level >= 0; level--) {
                        if (!m.TryGetLevelDt(level, out DT dtLevel) || dtLevel == null)
                            continue;

                        int activeCount = (level > 0) ? m.NodeCount(level) : total;
                        if (activeCount < 3) continue;

                        int pingRead = dtLevel.RenderPing ^ (tick & 1);
                        int pingWrite = pingRead ^ 1;

                        asyncCb.SetComputeIntParam(shader, "_ActiveCount", activeCount);
                        asyncCb.SetComputeIntParam(shader, "_FineCount", (level > 1) ? m.NodeCount(level - 1) : total);
                        asyncCb.SetComputeIntParam(shader, "_DtNeighborCount", dtLevel.NeighborCount);

                        if (updateDtPositionsForRender) {
                            asyncCb.SetComputeBufferParam(shader, kUpdateDtPositions, "_Pos", pos);
                            asyncCb.SetComputeBufferParam(shader, kUpdateDtPositions, "_DtPositions", dtLevel.GetPositionsBuffer(pingWrite));
                            asyncCb.SetComputeVectorParam(shader, "_DtNormCenter", new Vector4(m.DtNormCenter.x, m.DtNormCenter.y, 0f, 0f));
                            asyncCb.SetComputeFloatParam(shader, "_DtNormInvHalfExtent", m.DtNormInvHalfExtent);
                            asyncCb.DispatchCompute(shader, kUpdateDtPositions, (activeCount + 255) / 256, 1, 1);
                        }

                        if (runDtMaintain) {
                            dtLevel.EnqueueMaintain(asyncCb, dtLevel.GetPositionsBuffer(pingWrite), pingWrite, dtFixIterations, dtLegalizeIterations);
                        }
                    }
                }
            }

            if (maxDtLevel >= 0) {
                int pingXor = tickCount & 1;
                for (int level = maxDtLevel; level >= 0; level--) {
                    if (!m.TryGetLevelDt(level, out DT dtLevel) || dtLevel == null)
                        continue;

                    dtLevel.SetPendingRenderPing(dtLevel.RenderPing ^ pingXor);
                }
            }

            var fence = asyncCb.CreateGraphicsFence(GraphicsFenceType.AsyncQueueSynchronisation, SynchronisationStageFlags.ComputeProcessing);
            Graphics.ExecuteCommandBufferAsync(asyncCb, queueType);
            return fence;
        }



        public void CompleteAsyncBatch(Meshless m, int dtSwapMaxLevel) {
            if (m == null)
                return;

            for (int level = m.maxLayer; level >= 0; level--) {
                if (!m.TryGetLevelDt(level, out DT dtLevel) || dtLevel == null)
                    continue;

                dtLevel.CommitPendingRenderPing();
            }
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

            EnsureKernelsCached();

            int total = m.nodes.Count;
            if (total == 0) return;

            EnsureCapacity(total);
            if (!initialized || initializedCount != total)
                InitializeFromMeshless(m);

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
                if (!m.TryGetLevelDt(level, out DT dtLevel) || dtLevel == null)
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
                if (!m.TryGetLevelDt(level, out DT dtLevel) || dtLevel == null)
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

            if (asyncCb != null) {
                asyncCb.Dispose();
                asyncCb = null;
            }

            kernelsCached = false;
        }
    }
}
