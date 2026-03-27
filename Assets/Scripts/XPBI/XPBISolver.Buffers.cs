using Unity.Mathematics;
using UnityEngine;

namespace GPU.Solver {
    public sealed partial class XPBISolver {
        internal ComputeBuffer pos;
        internal ComputeBuffer vel;
        internal ComputeBuffer materialIds;
        internal ComputeBuffer invMass;
        internal ComputeBuffer restVolume;
        internal ComputeBuffer parentIndex;
        internal ComputeBuffer parentIndices;
        internal ComputeBuffer parentWeights;
        internal ComputeBuffer F;
        internal ComputeBuffer Fp;


        // CPU mirror arrays.
        private float2[] posCpu;
        private float2[] velCpu;
        private int[] materialIdsCpu;
        private float[] invMassCpu;
        private float[] restVolumeCpu;
        private int[] parentIndexCpu;
        private int[] parentIndicesCpu;
        private float[] parentWeightsCpu;
        private float4[] FCpu;
        private float4[] FpCpu;

        // Capacity and event counts.
        private int capacity;

        internal bool kernelsCached;

        internal void EnsureKernelsCached() {
            if (kernelsCached) return;

            gameplayForce.CacheKernels();
            layerCacheRuntime.CacheRuntimeKernels();
            layerSolveRuntime.CacheRuntimeKernels();
            hierarchySync.CacheRuntimeKernels();
            collisionEvent.CacheRuntimeKernels();
            solverDebug.CacheRuntimeKernels();

            kernelsCached = true;
        }

        internal void InitializeFromMeshless(System.Collections.Generic.List<MeshRange> ranges, int totalCount) {
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
                    int parentBase = gi * Const.ParentKNearest;
                    for (int k = 0; k < Const.ParentKNearest; k++) {
                        parentIndicesCpu[parentBase + k] = -1;
                        parentWeightsCpu[parentBase + k] = 0f;
                    }
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
            parentIndices.SetData(parentIndicesCpu, 0, 0, totalCount * Const.ParentKNearest);
            parentWeights.SetData(parentWeightsCpu, 0, 0, totalCount * Const.ParentKNearest);
            F.SetData(FCpu, 0, 0, totalCount);
            Fp.SetData(FpCpu, 0, 0, totalCount);

            float[] zeroDamage = new float[totalCount];
            layerSolveRuntime.Damage.SetData(zeroDamage, 0, 0, totalCount);
            layerSolveRuntime.DamageKappa.SetData(zeroDamage, 0, 0, totalCount);

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
            parentIndices = new ComputeBuffer(capacity * Const.ParentKNearest, sizeof(int), ComputeBufferType.Structured);
            parentWeights = new ComputeBuffer(capacity * Const.ParentKNearest, sizeof(float), ComputeBufferType.Structured);
            F = new ComputeBuffer(capacity, sizeof(float) * 4, ComputeBufferType.Structured);
            Fp = new ComputeBuffer(capacity, sizeof(float) * 4, ComputeBufferType.Structured);

            posCpu = new float2[capacity];
            velCpu = new float2[capacity];
            materialIdsCpu = new int[capacity];
            invMassCpu = new float[capacity];
            restVolumeCpu = new float[capacity];
            parentIndexCpu = new int[capacity];
            parentIndicesCpu = new int[capacity * Const.ParentKNearest];
            parentWeightsCpu = new float[capacity * Const.ParentKNearest];
            FCpu = new float4[capacity];
            FpCpu = new float4[capacity];

            layerSolveRuntime.AllocateRuntimeBuffers(capacity);
            collisionEvent.AllocateRuntimeBuffers(capacity);
            coloring.AllocateRuntimeBuffers(capacity);

            gameplayForce.EnsureCapacity(64);

            layerMappingCache.AllocateLayerMappingBuffers();
            solverDebug.AllocateRuntimeBuffers();

            initializedCount = -1;
        }

        internal void SetCommonShaderParams(float dt, int total, int baseIndex, float maxStep, float dtClamp = -1f) {
            float dtClampValue = dtClamp > 0f ? dtClamp : dt;
            float prolongationScale = Const.ProlongationScale;
            float postProlongSmoothing = math.max(0f, Const.PostProlongSmoothing);
            float xsphC = math.max(0f, Const.XsphC);
            float positionCorrectionMaxFraction = math.max(0f, Const.PositionCorrectionMaxFraction);

            for (int i = 0; i < commonParamShaders.Length; i++) {
                ComputeShader target = commonParamShaders[i];
                asyncCb.SetComputeFloatParam(target, "_Dt", dt);
                asyncCb.SetComputeFloatParam(target, "_DtClamp", dtClampValue);
                asyncCb.SetComputeFloatParam(target, "_Gravity", Const.Gravity);
                asyncCb.SetComputeFloatParam(target, "_Compliance", Const.Compliance);
                asyncCb.SetComputeFloatParam(target, "_MaxSpeed", Const.MaxVelocity);
                asyncCb.SetComputeFloatParam(target, "_MaxStep", maxStep);
                asyncCb.SetComputeIntParam(target, "_TotalCount", total);
                asyncCb.SetComputeIntParam(target, "_Base", baseIndex);
                asyncCb.SetComputeFloatParam(target, "_ProlongationScale", prolongationScale);
                asyncCb.SetComputeFloatParam(target, "_PostProlongSmoothing", postProlongSmoothing);
                asyncCb.SetComputeFloatParam(target, "_WendlandSupport", Const.WendlandSupport);
                asyncCb.SetComputeFloatParam(target, "_XsphC", xsphC);
                asyncCb.SetComputeIntParam(target, "_EnablePositionCorrection", Const.EnablePositionCorrection ? 1 : 0);
                asyncCb.SetComputeIntParam(target, "_PositionCorrectionIterations", Mathf.Max(0, Const.PositionCorrectionIterations));
                asyncCb.SetComputeFloatParam(target, "_PositionCorrectionGapRatio", Const.PositionCorrectionGapRatio);
                asyncCb.SetComputeFloatParam(target, "_PositionCorrectionCompliance", Const.PositionCorrectionCompliance);
                asyncCb.SetComputeFloatParam(target, "_PositionCorrectionMaxFraction", positionCorrectionMaxFraction);
                asyncCb.SetComputeFloatParam(target, "_CollisionSupportScale", Const.CollisionSupportScale);
                asyncCb.SetComputeFloatParam(target, "_CollisionSkinScale", Const.CollisionSkinScale);
                asyncCb.SetComputeFloatParam(target, "_CollisionCompliance", Const.CollisionCompliance);
                asyncCb.SetComputeFloatParam(target, "_CollisionSlop", Const.CollisionSlop);
                asyncCb.SetComputeFloatParam(target, "_CollisionPenBias", Const.CollisionPenBias);
                asyncCb.SetComputeFloatParam(target, "_CollisionMaxBias", Const.CollisionMaxBias);
                asyncCb.SetComputeFloatParam(target, "_CollisionMaxPush", Const.CollisionMaxPush);
                asyncCb.SetComputeFloatParam(target, "_CollisionRelaxation", Const.CollisionRelaxation);
                asyncCb.SetComputeFloatParam(target, "_CollisionMaxDv", Const.CollisionMaxDv);
                asyncCb.SetComputeFloatParam(target, "_CollisionFriction", Const.CollisionFriction);
                asyncCb.SetComputeFloatParam(target, "_CollisionRestitution", Const.CollisionRestitution);
                asyncCb.SetComputeFloatParam(target, "_CollisionRestitutionThreshold", Const.CollisionRestitutionThreshold);
                asyncCb.SetComputeFloatParam(target, "_CohesiveDamping", Const.CohesiveDamping);
                asyncCb.SetComputeFloatParam(target, "_CohesiveOnsetRatio", Const.CohesiveOnsetRatio);
                asyncCb.SetComputeFloatParam(target, "_CohesivePeakRatio", Const.CohesivePeakRatio);
                asyncCb.SetComputeFloatParam(target, "_DamageOnset", Const.DamageOnset);
                asyncCb.SetComputeFloatParam(target, "_DamageSoftening", Const.DamageSoftening);
                asyncCb.SetComputeFloatParam(target, "_DamageResidualStiffness", Const.DamageResidualStiffness);
                asyncCb.SetComputeFloatParam(target, "_DamageEnergyWeight", Const.DamageEnergyWeight);
                asyncCb.SetComputeFloatParam(target, "_DamageShellWeight", Const.DamageShellWeight);
                asyncCb.SetComputeFloatParam(target, "_DamageMax", Const.DamageMax);
                asyncCb.SetComputeFloatParam(target, "_CohesivePairScale", Const.CohesivePairScale);
                asyncCb.SetComputeIntParam(target, "_UseAffineProlongation", Const.UseAffineProlongation ? 1 : 0);
                asyncCb.SetComputeIntParam(target, "_ParentKNearest", 2);
                asyncCb.SetComputeFloatParam(target, "_ParentWeightEpsilon", math.max(Const.ParentWeightEpsilon, 1e-6f));
            }
        }

        void ReleaseBuffers() {
            pos?.Dispose(); pos = null;
            vel?.Dispose(); vel = null;
            materialIds?.Dispose(); materialIds = null;
            invMass?.Dispose(); invMass = null;
            restVolume?.Dispose(); restVolume = null;
            parentIndex?.Dispose(); parentIndex = null;
            parentIndices?.Dispose(); parentIndices = null;
            parentWeights?.Dispose(); parentWeights = null;
            F?.Dispose(); F = null;
            Fp?.Dispose(); Fp = null;

            layerSolveRuntime.ReleaseRuntimeBuffers();
            collisionEvent.ReleaseRuntimeBuffers();
            coloring.ReleaseRuntimeBuffers();
            solverDebug.ReleaseRuntimeBuffers();
            layerMappingCache.ReleaseLayerMappingBuffers();

            initializedCount = -1;
        }
    }
}
