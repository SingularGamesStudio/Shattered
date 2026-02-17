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

        ComputeBuffer colorOrder;

        float2[] posCpu;
        float2[] velCpu;
        float[] invMassCpu;
        uint[] flagsCpu;
        float[] restVolumeCpu;
        int[] parentIndexCpu;
        float4[] FCpu;
        float4[] FpCpu;

        int[] dtNeighborCountsCpu;
        int[] dtNeighborsCpu;

        int[] colorsCpu;
        int[] colorOrderCpu;

        int[] colorCounts;
        int[] colorStarts;
        int[] colorWrite;

        int capacity;
        int coloringCapacity;
        int coloringNeighborCapacity;

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

            colorOrder = new ComputeBuffer(capacity, sizeof(int), ComputeBufferType.Structured);

            posCpu = new float2[capacity];
            velCpu = new float2[capacity];
            invMassCpu = new float[capacity];
            flagsCpu = new uint[capacity];
            restVolumeCpu = new float[capacity];
            parentIndexCpu = new int[capacity];
            FCpu = new float4[capacity];
            FpCpu = new float4[capacity];

            coloringCapacity = 0;
            coloringNeighborCapacity = 0;
            dtNeighborCountsCpu = null;
            dtNeighborsCpu = null;
            colorsCpu = null;
            colorOrderCpu = null;
            colorCounts = null;
            colorStarts = null;
            colorWrite = null;
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

        void EnsureColoringCapacity(int activeCount, int neighborCount) {
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

            if (coloringNeighborCapacity < neighborCount) {
                coloringNeighborCapacity = math.max(8, coloringNeighborCapacity);
                while (coloringNeighborCapacity < neighborCount) coloringNeighborCapacity *= 2;
            }

            int countsLen = coloringCapacity;
            if (dtNeighborCountsCpu == null || dtNeighborCountsCpu.Length < countsLen)
                dtNeighborCountsCpu = new int[countsLen];

            int neighLen = coloringCapacity * coloringNeighborCapacity;
            if (dtNeighborsCpu == null || dtNeighborsCpu.Length < neighLen)
                dtNeighborsCpu = new int[neighLen];
        }

        int ClampNeighborCount(int n, int neighborCount) {
            if (n < 0) return 0;
            if (n > neighborCount) return neighborCount;
            return n;
        }

        void Build2HopColoring(DelaunayGpu dtLevel, int activeCount, int neighborCount, out int colorCount) {
            EnsureColoringCapacity(activeCount, neighborCount);

            dtLevel.NeighborCountsBuffer.GetData(dtNeighborCountsCpu, 0, 0, activeCount);
            dtLevel.NeighborsBuffer.GetData(dtNeighborsCpu, 0, 0, activeCount * neighborCount);

            for (int i = 0; i < activeCount; i++) colorsCpu[i] = -1;

            int usedColors = 0;

            for (int i = 0; i < activeCount; i++) {
                int stamp = i + 1;

                int baseI = i * neighborCount;
                int nI = ClampNeighborCount(dtNeighborCountsCpu[i], neighborCount);

                for (int a = 0; a < nI; a++) {
                    int na = dtNeighborsCpu[baseI + a];
                    if ((uint)na >= (uint)activeCount) continue;

                    int ci = colorsCpu[na];
                    if ((uint)ci < (uint)usedColors) colorCounts[ci] = stamp;
                }

                for (int a = 0; a < nI; a++) {
                    int na = dtNeighborsCpu[baseI + a];
                    if ((uint)na >= (uint)activeCount) continue;

                    int baseA = na * neighborCount;
                    int nA = ClampNeighborCount(dtNeighborCountsCpu[na], neighborCount);

                    for (int b = 0; b < nA; b++) {
                        int nb = dtNeighborsCpu[baseA + b];
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

            colorOrder.SetData(colorOrderCpu, 0, 0, activeCount);

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

            shader.SetBuffer(kRelaxColored, "_ColorOrder", colorOrder);
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

                Build2HopColoring(dtLevel, activeCount, neighborCount, out int colorCount);

                int iterations = level == 0 ? Const.Iterations : Const.HPBDIterations;
                for (int iter = 0; iter < iterations; iter++) {
                    for (int c = 0; c < colorCount; c++) {
                        int start = colorStarts[c];
                        int count = colorCounts[c];
                        if (count <= 0) continue;

                        shader.SetInt("_ColorStart", start);
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

            colorOrder?.Dispose(); colorOrder = null;
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
            coloringNeighborCapacity = 0;
            dtNeighborCountsCpu = null;
            dtNeighborsCpu = null;
            colorsCpu = null;
            colorOrderCpu = null;
            colorCounts = null;
            colorStarts = null;
            colorWrite = null;
        }
    }
}
