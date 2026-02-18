using System;
using GPU.Delaunay;
using Unity.Mathematics;
using UnityEngine;

namespace GPU.Solver {
    public sealed class XPBISolver : IDisposable {
        const uint FixedFlag = 1u;

        const int ColoringMaxColors = 64;
        const int ColoringBatchRounds = 8;
        const int ColoringMaxBatches = 32;

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

        ComputeBuffer[] colorOrders;

        ComputeBuffer coloringColor;
        ComputeBuffer coloringProposed;
        ComputeBuffer coloringPrio;

        ComputeBuffer coloringCountsGpu;
        ComputeBuffer coloringStartsGpu;
        ComputeBuffer coloringWriteGpu;
        ComputeBuffer coloringStatsGpu;

        int[] coloringCountsCpu;
        int[] coloringStartsCpu;
        int[] coloringStatsCpu;

        float2[] posCpu;
        float2[] velCpu;
        float[] invMassCpu;
        uint[] flagsCpu;
        float[] restVolumeCpu;
        int[] parentIndexCpu;
        float4[] FCpu;
        float4[] FpCpu;

        int[][] cachedColorCountsByLevel;
        int[][] cachedColorStartsByLevel;
        int[] cachedColorCountByLevel;
        int[] cachedActiveCountByLevel;
        uint[] cachedAdjacencyVersionByLevel;

        int capacity;

        bool loggedKernelError;

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

            coloringCountsGpu = new ComputeBuffer(ColoringMaxColors, sizeof(int), ComputeBufferType.Structured);
            coloringStartsGpu = new ComputeBuffer(ColoringMaxColors, sizeof(int), ComputeBufferType.Structured);
            coloringWriteGpu = new ComputeBuffer(ColoringMaxColors, sizeof(int), ComputeBufferType.Structured);
            coloringStatsGpu = new ComputeBuffer(2, sizeof(int), ComputeBufferType.Structured);

            coloringCountsCpu = new int[ColoringMaxColors];
            coloringStartsCpu = new int[ColoringMaxColors];
            coloringStatsCpu = new int[2];

            posCpu = new float2[capacity];
            velCpu = new float2[capacity];
            invMassCpu = new float[capacity];
            flagsCpu = new uint[capacity];
            restVolumeCpu = new float[capacity];
            parentIndexCpu = new int[capacity];
            FCpu = new float4[capacity];
            FpCpu = new float4[capacity];

            colorOrders = null;

            cachedColorCountsByLevel = null;
            cachedColorStartsByLevel = null;
            cachedColorCountByLevel = null;
            cachedActiveCountByLevel = null;
            cachedAdjacencyVersionByLevel = null;
        }

        public void UploadFromMeshless(Meshless m) {
            int n = m.nodes.Count;
            EnsureCapacity(n);

            for (int i = 0; i < n; i++) {
                var node = m.nodes[i];
                posCpu[i] = node.pos;
                velCpu[i] = node.vel;
                invMassCpu[i] = node.invMass;
                flagsCpu[i] = node.isFixed || node.invMass <= 0f ? FixedFlag : 0u;
                restVolumeCpu[i] = node.restVolume;
                parentIndexCpu[i] = node.parentIndex;
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
        }

        public void DownloadToMeshless(Meshless m) {
            int n = m.nodes.Count;

            vel.GetData(velCpu, 0, 0, n);
            F.GetData(FCpu, 0, 0, n);
            Fp.GetData(FpCpu, 0, 0, n);

            for (int i = 0; i < n; i++) {
                m.nodes[i].vel = velCpu[i];

                var f = FCpu[i];
                m.nodes[i].F = new float2x2(f.xy, f.zw);

                var fp = FpCpu[i];
                m.nodes[i].Fp = new float2x2(fp.xy, fp.zw);
            }
        }

        void EnsurePerLevelCaches(int levelCount) {
            if (levelCount <= 0) return;

            if (colorOrders == null || colorOrders.Length < levelCount) {
                var next = new ComputeBuffer[levelCount];
                if (colorOrders != null) {
                    Array.Copy(colorOrders, next, colorOrders.Length);
                }
                colorOrders = next;
            }

            if (cachedColorCountsByLevel == null || cachedColorCountsByLevel.Length < levelCount) {
                cachedColorCountsByLevel = new int[levelCount][];
                cachedColorStartsByLevel = new int[levelCount][];
                cachedColorCountByLevel = new int[levelCount];
                cachedActiveCountByLevel = new int[levelCount];
                cachedAdjacencyVersionByLevel = new uint[levelCount];
                for (int i = 0; i < levelCount; i++) {
                    cachedColorCountByLevel[i] = 0;
                    cachedActiveCountByLevel[i] = 0;
                    cachedAdjacencyVersionByLevel[i] = 0;
                }
            } else if (cachedColorCountsByLevel.Length < levelCount) {
                Array.Resize(ref cachedColorCountsByLevel, levelCount);
                Array.Resize(ref cachedColorStartsByLevel, levelCount);
                Array.Resize(ref cachedColorCountByLevel, levelCount);
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

        bool HasAllKernels() {
            return
                shader.HasKernel("ClearCurrentVolume") &&
                shader.HasKernel("CacheVolumesHierarchical") &&
                shader.HasKernel("CacheKernelH") &&
                shader.HasKernel("ComputeCorrectionL") &&
                shader.HasKernel("CacheF0AndResetLambda") &&
                shader.HasKernel("SaveVelPrefix") &&
                shader.HasKernel("ClearVelDelta") &&
                shader.HasKernel("RelaxScatter") &&
                shader.HasKernel("RelaxColored") &&
                shader.HasKernel("ApplyVelDelta") &&
                shader.HasKernel("Prolongate") &&
                shader.HasKernel("CommitDeformation") &&
                shader.HasKernel("ExternalForces") &&
                shader.HasKernel("ColoringInit") &&
                shader.HasKernel("ColoringClearChanged") &&
                shader.HasKernel("ColoringClearUncolored") &&
                shader.HasKernel("ColoringPropose") &&
                shader.HasKernel("ColoringResolve") &&
                shader.HasKernel("ColoringCountUncolored") &&
                shader.HasKernel("ColoringClearMeta") &&
                shader.HasKernel("ColoringBuildCounts") &&
                shader.HasKernel("ColoringBuildStarts") &&
                shader.HasKernel("ColoringScatterOrder");
        }

        public void SolveHierarchical(Meshless m, float dt, bool useHierarchical) {
            if (!HasAllKernels()) {
                if (!loggedKernelError) {
                    loggedKernelError = true;
                    Debug.LogError(
                        $"XPBIGpuSolver: Compute shader '{shader.name}' is missing kernels or failed to compile. " +
                        "Check Console for shader compilation errors from XPBISolver.compute (often bad #include path / HLSL compile error).");
                }
                return;
            }

            int kClearCurrentVolume = shader.FindKernel("ClearCurrentVolume");
            int kCacheVolumesHierarchical = shader.FindKernel("CacheVolumesHierarchical");
            int kCacheKernelH = shader.FindKernel("CacheKernelH");
            int kComputeCorrectionL = shader.FindKernel("ComputeCorrectionL");
            int kCacheF0AndResetLambda = shader.FindKernel("CacheF0AndResetLambda");
            int kSaveVelPrefix = shader.FindKernel("SaveVelPrefix");
            int kClearVelDelta = shader.FindKernel("ClearVelDelta");
            int kRelaxScatter = shader.FindKernel("RelaxScatter");
            int kRelaxColored = shader.FindKernel("RelaxColored");
            int kApplyVelDelta = shader.FindKernel("ApplyVelDelta");
            int kProlongate = shader.FindKernel("Prolongate");
            int kCommitDeformation = shader.FindKernel("CommitDeformation");
            int kExternalForces = shader.FindKernel("ExternalForces");

            int kColoringInit = shader.FindKernel("ColoringInit");
            int kColoringClearChanged = shader.FindKernel("ColoringClearChanged");
            int kColoringClearUncolored = shader.FindKernel("ColoringClearUncolored");
            int kColoringPropose = shader.FindKernel("ColoringPropose");
            int kColoringResolve = shader.FindKernel("ColoringResolve");
            int kColoringCountUncolored = shader.FindKernel("ColoringCountUncolored");
            int kColoringClearMeta = shader.FindKernel("ColoringClearMeta");
            int kColoringBuildCounts = shader.FindKernel("ColoringBuildCounts");
            int kColoringBuildStarts = shader.FindKernel("ColoringBuildStarts");
            int kColoringScatterOrder = shader.FindKernel("ColoringScatterOrder");

            int total = m.nodes.Count;
            if (total == 0) return;

            shader.SetBuffer(kExternalForces, "_Pos", pos);
            shader.SetBuffer(kExternalForces, "_Vel", vel);
            shader.SetBuffer(kExternalForces, "_InvMass", invMass);
            shader.SetBuffer(kExternalForces, "_Flags", flags);

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

            shader.SetBuffer(kRelaxScatter, "_Pos", pos);
            shader.SetBuffer(kRelaxScatter, "_Vel", vel);
            shader.SetBuffer(kRelaxScatter, "_InvMass", invMass);
            shader.SetBuffer(kRelaxScatter, "_Flags", flags);
            shader.SetBuffer(kRelaxScatter, "_RestVolume", restVolume);
            shader.SetBuffer(kRelaxScatter, "_F0", F0);
            shader.SetBuffer(kRelaxScatter, "_L", L);
            shader.SetBuffer(kRelaxScatter, "_KernelH", kernelH);
            shader.SetBuffer(kRelaxScatter, "_CurrentVolumeBits", currentVolumeBits);
            shader.SetBuffer(kRelaxScatter, "_Lambda", lambda);
            shader.SetBuffer(kRelaxScatter, "_VelDeltaBits", velDeltaBits);

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

            shader.SetBuffer(kApplyVelDelta, "_Vel", vel);
            shader.SetBuffer(kApplyVelDelta, "_VelDeltaBits", velDeltaBits);

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

            shader.SetBuffer(kColoringInit, "_ColoringColor", coloringColor);
            shader.SetBuffer(kColoringInit, "_ColoringProposed", coloringProposed);
            shader.SetBuffer(kColoringInit, "_ColoringPrio", coloringPrio);

            shader.SetBuffer(kColoringClearChanged, "_ColoringStats", coloringStatsGpu);
            shader.SetBuffer(kColoringClearUncolored, "_ColoringStats", coloringStatsGpu);

            shader.SetBuffer(kColoringPropose, "_ColoringColor", coloringColor);
            shader.SetBuffer(kColoringPropose, "_ColoringProposed", coloringProposed);
            shader.SetBuffer(kColoringPropose, "_ColoringPrio", coloringPrio);

            shader.SetBuffer(kColoringResolve, "_ColoringColor", coloringColor);
            shader.SetBuffer(kColoringResolve, "_ColoringProposed", coloringProposed);
            shader.SetBuffer(kColoringResolve, "_ColoringPrio", coloringPrio);
            shader.SetBuffer(kColoringResolve, "_ColoringStats", coloringStatsGpu);

            shader.SetBuffer(kColoringCountUncolored, "_ColoringColor", coloringColor);
            shader.SetBuffer(kColoringCountUncolored, "_ColoringStats", coloringStatsGpu);

            shader.SetBuffer(kColoringClearMeta, "_ColoringCounts", coloringCountsGpu);
            shader.SetBuffer(kColoringClearMeta, "_ColoringStarts", coloringStartsGpu);
            shader.SetBuffer(kColoringClearMeta, "_ColoringWrite", coloringWriteGpu);

            shader.SetBuffer(kColoringBuildCounts, "_ColoringColor", coloringColor);
            shader.SetBuffer(kColoringBuildCounts, "_ColoringCounts", coloringCountsGpu);

            shader.SetBuffer(kColoringBuildStarts, "_ColoringCounts", coloringCountsGpu);
            shader.SetBuffer(kColoringBuildStarts, "_ColoringStarts", coloringStartsGpu);
            shader.SetBuffer(kColoringBuildStarts, "_ColoringWrite", coloringWriteGpu);

            shader.SetBuffer(kColoringScatterOrder, "_ColoringColor", coloringColor);
            shader.SetBuffer(kColoringScatterOrder, "_ColoringWrite", coloringWriteGpu);

            shader.SetInt("_Base", 0);
            shader.SetInt("_TotalCount", total);
            shader.SetFloat("_Dt", dt);
            shader.SetFloat("_Gravity", m.gravity);
            shader.SetFloat("_Compliance", m.compliance);

            shader.Dispatch(kExternalForces, (total + 255) / 256, 1, 1);

            int maxLevel = useHierarchical && m.levelEndIndex != null ? m.maxLayer : 0;
            EnsurePerLevelCaches(maxLevel + 1);

            for (int level = maxLevel; level >= 0; level--) {
                if (!m.TryGetLevelDt(level, out DT dtLevel))
                    continue;

                int activeCount = level > 0 ? m.NodeCount(level) : total;
                if (activeCount < 3) continue;

                int fineCount = level > 1 ? m.NodeCount(level - 1) : total;

                shader.SetInt("_ActiveCount", activeCount);
                shader.SetInt("_FineCount", fineCount);

                int neighborCount = dtLevel.NeighborCount;
                shader.SetInt("_DtNeighborCount", neighborCount);

                shader.SetBuffer(kCacheKernelH, "_DtNeighbors", dtLevel.NeighborsBuffer);
                shader.SetBuffer(kCacheKernelH, "_DtNeighborCounts", dtLevel.NeighborCountsBuffer);

                shader.SetBuffer(kComputeCorrectionL, "_DtNeighbors", dtLevel.NeighborsBuffer);
                shader.SetBuffer(kComputeCorrectionL, "_DtNeighborCounts", dtLevel.NeighborCountsBuffer);

                shader.SetBuffer(kRelaxScatter, "_DtNeighbors", dtLevel.NeighborsBuffer);
                shader.SetBuffer(kRelaxScatter, "_DtNeighborCounts", dtLevel.NeighborCountsBuffer);

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
                    cachedColorCountByLevel[level] <= 0 ||
                    cachedActiveCountByLevel[level] != activeCount ||
                    cachedAdjacencyVersionByLevel[level] != adjacencyVersion ||
                    cachedColorCountsByLevel[level] == null ||
                    cachedColorStartsByLevel[level] == null;

                if (rebuildColors) {
                    var orderBuf = EnsureColorOrderBufferForLevel(level);

                    if (cachedColorCountsByLevel[level] == null || cachedColorCountsByLevel[level].Length != ColoringMaxColors)
                        cachedColorCountsByLevel[level] = new int[ColoringMaxColors];

                    if (cachedColorStartsByLevel[level] == null || cachedColorStartsByLevel[level].Length != ColoringMaxColors)
                        cachedColorStartsByLevel[level] = new int[ColoringMaxColors];

                    uint seed = adjacencyVersion ^ (uint)(level * 2654435761);

                    shader.SetInt("_ColoringActiveCount", activeCount);
                    shader.SetInt("_ColoringDtNeighborCount", neighborCount);
                    shader.SetInt("_ColoringMaxColors", ColoringMaxColors);
                    shader.SetInt("_ColoringSeed", unchecked((int)seed));

                    shader.SetBuffer(kColoringInit, "_ColoringDtNeighbors", dtLevel.NeighborsBuffer);
                    shader.SetBuffer(kColoringInit, "_ColoringDtNeighborCounts", dtLevel.NeighborCountsBuffer);

                    shader.SetBuffer(kColoringPropose, "_ColoringDtNeighbors", dtLevel.NeighborsBuffer);
                    shader.SetBuffer(kColoringPropose, "_ColoringDtNeighborCounts", dtLevel.NeighborCountsBuffer);

                    shader.SetBuffer(kColoringResolve, "_ColoringDtNeighbors", dtLevel.NeighborsBuffer);
                    shader.SetBuffer(kColoringResolve, "_ColoringDtNeighborCounts", dtLevel.NeighborCountsBuffer);

                    shader.SetBuffer(kColoringScatterOrder, "_ColoringOrderOut", orderBuf);

                    shader.Dispatch(kColoringInit, (activeCount + 255) / 256, 1, 1);

                    bool coloredOk = false;

                    for (int batch = 0; batch < ColoringMaxBatches; batch++) {
                        for (int r = 0; r < ColoringBatchRounds; r++) {
                            shader.Dispatch(kColoringClearChanged, 1, 1, 1);
                            shader.Dispatch(kColoringPropose, (activeCount + 255) / 256, 1, 1);
                            shader.Dispatch(kColoringResolve, (activeCount + 255) / 256, 1, 1);
                        }

                        shader.Dispatch(kColoringClearUncolored, 1, 1, 1);
                        shader.Dispatch(kColoringCountUncolored, (activeCount + 255) / 256, 1, 1);

                        coloringStatsGpu.GetData(coloringStatsCpu, 0, 0, 2);
                        int uncolored = coloringStatsCpu[0];
                        int changed = coloringStatsCpu[1];

                        if (uncolored == 0) {
                            coloredOk = true;
                            break;
                        }

                        if (changed == 0) {
                            break;
                        }
                    }

                    if (!coloredOk) {
                        Debug.LogError($"XPBISolver: GPU 2-hop coloring did not converge (level={level}, activeCount={activeCount}, adjacencyVersion={adjacencyVersion}).");
                        cachedColorCountByLevel[level] = 0;
                        cachedActiveCountByLevel[level] = activeCount;
                        cachedAdjacencyVersionByLevel[level] = adjacencyVersion;
                        continue;
                    }

                    shader.Dispatch(kColoringClearMeta, 1, 1, 1);
                    shader.Dispatch(kColoringBuildCounts, (activeCount + 255) / 256, 1, 1);
                    shader.Dispatch(kColoringBuildStarts, 1, 1, 1);
                    shader.Dispatch(kColoringScatterOrder, (activeCount + 255) / 256, 1, 1);

                    coloringCountsGpu.GetData(coloringCountsCpu, 0, 0, ColoringMaxColors);
                    coloringStartsGpu.GetData(coloringStartsCpu, 0, 0, ColoringMaxColors);

                    int colorCount = 0;
                    for (int c = 0; c < ColoringMaxColors; c++) {
                        if (coloringCountsCpu[c] > 0) colorCount = c + 1;
                    }

                    if (colorCount <= 0) {
                        cachedColorCountByLevel[level] = 0;
                        cachedActiveCountByLevel[level] = activeCount;
                        cachedAdjacencyVersionByLevel[level] = adjacencyVersion;
                        continue;
                    }

                    Array.Copy(coloringCountsCpu, 0, cachedColorCountsByLevel[level], 0, ColoringMaxColors);
                    Array.Copy(coloringStartsCpu, 0, cachedColorStartsByLevel[level], 0, ColoringMaxColors);

                    cachedColorCountByLevel[level] = colorCount;
                    cachedActiveCountByLevel[level] = activeCount;
                    cachedAdjacencyVersionByLevel[level] = adjacencyVersion;
                }

                int cachedColorCount = cachedColorCountByLevel[level];
                if (cachedColorCount <= 0) continue;

                var colorOrderBuf = EnsureColorOrderBufferForLevel(level);
                shader.SetBuffer(kRelaxColored, "_ColorOrder", colorOrderBuf);

                int iterations = level == 0 ? Const.Iterations : Const.HPBDIterations;
                int[] levelColorCounts2 = cachedColorCountsByLevel[level];
                int[] levelColorStarts2 = cachedColorStartsByLevel[level];

                for (int iter = 0; iter < iterations; iter++) {
                    for (int c = 0; c < cachedColorCount; c++) {
                        int count = levelColorCounts2[c];
                        if (count <= 0) continue;

                        shader.SetInt("_ColorStart", levelColorStarts2[c]);
                        shader.SetInt("_ColorCount", count);

                        shader.Dispatch(kRelaxColored, (count + 255) / 256, 1, 1);
                    }
                }

                if (level > 0 && fineCount > activeCount) {
                    shader.Dispatch(kProlongate, ((fineCount - activeCount) + 255) / 256, 1, 1);
                }

                if (level == 0) {
                    shader.Dispatch(kCommitDeformation, (activeCount + 255) / 256, 1, 1);
                }
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

            coloringCountsGpu?.Dispose(); coloringCountsGpu = null;
            coloringStartsGpu?.Dispose(); coloringStartsGpu = null;
            coloringWriteGpu?.Dispose(); coloringWriteGpu = null;
            coloringStatsGpu?.Dispose(); coloringStatsGpu = null;

            if (colorOrders != null) {
                for (int i = 0; i < colorOrders.Length; i++) {
                    colorOrders[i]?.Dispose();
                    colorOrders[i] = null;
                }
                colorOrders = null;
            }
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

            coloringCountsCpu = null;
            coloringStartsCpu = null;
            coloringStatsCpu = null;

            cachedColorCountsByLevel = null;
            cachedColorStartsByLevel = null;
            cachedColorCountByLevel = null;
            cachedActiveCountByLevel = null;
            cachedAdjacencyVersionByLevel = null;
        }
    }
}
