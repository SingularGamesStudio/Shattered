using System;
using GPU.Delaunay;
using Unity.Mathematics;
using UnityEngine;

namespace GPU.Solver {
    public sealed class XPBIGpuSolver : IDisposable {
        const uint FixedFlag = 1u;

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

        float2[] posCpu;
        float2[] velCpu;
        float[] invMassCpu;
        uint[] flagsCpu;
        float[] restVolumeCpu;
        int[] parentIndexCpu;
        float4[] FCpu;
        float4[] FpCpu;

        int[] colorsCpu;
        int[] colorOrderCpu;

        int[] colorCounts;
        int[] colorStarts;
        int[] colorWrite;

        int[][] cachedColorCountsByLevel;
        int[][] cachedColorStartsByLevel;
        int[] cachedColorCountByLevel;
        int[] cachedActiveCountByLevel;
        uint[] cachedAdjacencyVersionByLevel;

        int capacity;
        int coloringCapacity;

        bool loggedKernelError;

        public XPBIGpuSolver(ComputeShader shader) {
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

            posCpu = new float2[capacity];
            velCpu = new float2[capacity];
            invMassCpu = new float[capacity];
            flagsCpu = new uint[capacity];
            restVolumeCpu = new float[capacity];
            parentIndexCpu = new int[capacity];
            FCpu = new float4[capacity];
            FpCpu = new float4[capacity];

            coloringCapacity = 0;
            colorsCpu = null;
            colorOrderCpu = null;
            colorCounts = null;
            colorStarts = null;
            colorWrite = null;

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

        void EnsureColoringCapacity(int activeCount) {
            if (activeCount <= 0) return;

            if (coloringCapacity < activeCount) {
                coloringCapacity = math.max(256, coloringCapacity);
                while (coloringCapacity < activeCount) coloringCapacity *= 2;

                colorsCpu = new int[coloringCapacity];
                colorOrderCpu = new int[coloringCapacity];

                colorCounts = new int[coloringCapacity];
                colorStarts = new int[coloringCapacity];
                colorWrite = new int[coloringCapacity];
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

        int ClampNeighborCount(int n, int neighborCount) {
            if (n < 0) return 0;
            if (n > neighborCount) return neighborCount;
            return n;
        }

        void Build2HopColoring(int[] neighborsCpu, int[] neighborCountsCpu, int activeCount, int neighborCount, out int colorCount) {
            for (int i = 0; i < activeCount; i++) colorsCpu[i] = -1;

            int usedColors = 0;

            for (int i = 0; i < activeCount; i++) {
                int stamp = i + 1;

                int baseI = i * neighborCount;
                int nI = ClampNeighborCount(neighborCountsCpu[i], neighborCount);

                for (int a = 0; a < nI; a++) {
                    int na = neighborsCpu[baseI + a];
                    if ((uint)na >= (uint)activeCount) continue;

                    int ci = colorsCpu[na];
                    if ((uint)ci < (uint)usedColors) colorCounts[ci] = stamp;
                }

                for (int a = 0; a < nI; a++) {
                    int na = neighborsCpu[baseI + a];
                    if ((uint)na >= (uint)activeCount) continue;

                    int baseA = na * neighborCount;
                    int nA = ClampNeighborCount(neighborCountsCpu[na], neighborCount);

                    for (int b = 0; b < nA; b++) {
                        int nb = neighborsCpu[baseA + b];
                        if ((uint)nb >= (uint)activeCount) continue;

                        int ci = colorsCpu[nb];
                        if ((uint)ci < (uint)usedColors) colorCounts[ci] = stamp;
                    }
                }

                int assigned = -1;
                for (int c = 0; c < usedColors; c++) {
                    if (colorCounts[c] != stamp) { assigned = c; break; }
                }
                if (assigned < 0) assigned = usedColors++;

                colorsCpu[i] = assigned;
            }

            Array.Clear(colorCounts, 0, usedColors);

            for (int i = 0; i < activeCount; i++) colorCounts[colorsCpu[i]]++;

            int sum = 0;
            for (int c = 0; c < usedColors; c++) {
                colorStarts[c] = sum;
                sum += colorCounts[c];
                colorWrite[c] = colorStarts[c];
            }

            for (int i = 0; i < activeCount; i++) {
                int c = colorsCpu[i];
                colorOrderCpu[colorWrite[c]++] = i;
            }

            colorCount = usedColors;
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
                shader.HasKernel("ExternalForces");
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

            shader.SetInt("_Base", 0);
            shader.SetInt("_TotalCount", total);
            shader.SetFloat("_Dt", dt);
            shader.SetFloat("_Gravity", m.gravity);
            shader.SetFloat("_Compliance", m.compliance);

            shader.Dispatch(kExternalForces, (total + 255) / 256, 1, 1);

            int maxLevel = useHierarchical && m.levelEndIndex != null ? m.maxLayer : 0;
            EnsurePerLevelCaches(maxLevel + 1);

            for (int level = maxLevel; level >= 0; level--) {
                if (!m.TryGetLevelDt(level, out DelaunayGpu dtLevel))
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

                EnsureColoringCapacity(activeCount);

                uint adjacencyVersion = dtLevel.AdjacencyVersion;
                bool rebuildColors =
                    cachedColorCountByLevel[level] <= 0 ||
                    cachedActiveCountByLevel[level] != activeCount ||
                    cachedAdjacencyVersionByLevel[level] != adjacencyVersion ||
                    cachedColorCountsByLevel[level] == null ||
                    cachedColorStartsByLevel[level] == null;

                if (rebuildColors) {
                    int[] neigh = dtLevel.NeighborsCpu;
                    int[] counts = dtLevel.NeighborCountsCpu;
                    if (neigh == null || counts == null) {
                        continue;
                    }

                    Build2HopColoring(neigh, counts, activeCount, neighborCount, out int colorCount);

                    var levelCounts = cachedColorCountsByLevel[level];
                    if (levelCounts == null || levelCounts.Length != activeCount)
                        levelCounts = cachedColorCountsByLevel[level] = new int[activeCount];

                    var levelStarts = cachedColorStartsByLevel[level];
                    if (levelStarts == null || levelStarts.Length != activeCount)
                        levelStarts = cachedColorStartsByLevel[level] = new int[activeCount];

                    Array.Copy(colorCounts, 0, levelCounts, 0, colorCount);
                    Array.Copy(colorStarts, 0, levelStarts, 0, colorCount);

                    cachedColorCountByLevel[level] = colorCount;
                    cachedActiveCountByLevel[level] = activeCount;
                    cachedAdjacencyVersionByLevel[level] = adjacencyVersion;

                    var orderBuf = EnsureColorOrderBufferForLevel(level);
                    orderBuf.SetData(colorOrderCpu, 0, 0, activeCount);
                }

                int cachedColorCount = cachedColorCountByLevel[level];
                if (cachedColorCount <= 0) continue;

                var colorBuf2 = EnsureColorOrderBufferForLevel(level);
                shader.SetBuffer(kRelaxColored, "_ColorOrder", colorBuf2);

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

            coloringCapacity = 0;
            colorsCpu = null;
            colorOrderCpu = null;
            colorCounts = null;
            colorStarts = null;
            colorWrite = null;

            cachedColorCountsByLevel = null;
            cachedColorStartsByLevel = null;
            cachedColorCountByLevel = null;
            cachedActiveCountByLevel = null;
            cachedAdjacencyVersionByLevel = null;
        }
    }
}
