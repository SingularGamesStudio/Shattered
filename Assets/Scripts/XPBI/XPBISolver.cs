using System;
using GPU.Delaunay;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace GPU.Solver {
    public sealed partial class XPBISolver : IDisposable {

        private bool initialized;
        private int initializedCount = -1;
        private bool parentsBuilt;
        private bool loggedKernelError;
        private Meshless meshless;

        private CommandBuffer asyncCb;

        public ComputeBuffer PositionBuffer => pos;

        private const uint FixedFlag = 1u;
        private const int ColoringMaxColors = 64;
        private const int ColoringConflictRounds = 24;
        readonly ComputeShader shader;

        [System.Runtime.InteropServices.StructLayout(System.Runtime.InteropServices.LayoutKind.Sequential)]
        public struct ForceEvent {
            public uint node;
            public float2 force;
        }

        public XPBISolver(ComputeShader shader) {
            this.shader = shader ? shader : throw new ArgumentNullException(nameof(shader));
        }

        private delegate void DispatchAction(ComputeShader shader, int kernel, int x, int y, int z);
        private delegate void DispatchActionIndirect(ComputeShader shader, int kernel, ComputeBuffer args, uint offset);

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
        // Public entry point for asynchronous simulation
        public GraphicsFence SubmitSolve(
            Meshless m,
            float dtPerTick,
            int tickCount,
            bool useHierarchical,
            int dtFixIterations,
            int dtLegalizeIterations,
            bool rebuildParents,
            bool updateDtPositionsForRender,
            ComputeQueueType queueType) {
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

            // Create or clear command buffer
            if (asyncCb == null)
                asyncCb = new CommandBuffer { name = "XPBI Async Batch" };
            asyncCb.Clear();
            asyncCb.SetExecutionFlags(CommandBufferExecutionFlags.AsyncCompute);

            // Async dispatch wrappers
            DispatchAction dispatch = (shader, kernel, x, y, z) =>
                asyncCb.DispatchCompute(shader, kernel, x, y, z);
            DispatchActionIndirect dispatchIndirect = (shader, kernel, args, offset) =>
                asyncCb.DispatchCompute(shader, kernel, args, offset);

            int maxSolveLevel = (useHierarchical && m.levelEndIndex != null) ? m.maxLayer : 0;
            EnsurePerLevelCaches(maxSolveLevel + 1);

            int maxDtLevel = m.maxLayer;

            for (int tick = 0; tick < tickCount; tick++) {
                // Set per‑tick global parameters
                SetCommonShaderParams(dtPerTick, m.gravity, m.compliance, total);

                // Rebuild parent hierarchy if requested (only on first tick)
                bool rebuildAllParents = useHierarchical && m.maxLayer > 0 && (rebuildParents || !parentsBuilt);
                if (tick == 0 && rebuildAllParents)
                    RebuildAllParents(m, total, dispatch);

                // Apply gameplay and external forces
                ApplyForces(total, dispatch);

                // Process all levels
                for (int level = maxSolveLevel; level >= 0; level--) {
                    if (!m.TryGetLevelDt(level, out DT dtLevel) || dtLevel == null)
                        continue;

                    int activeCount = level > 0 ? m.NodeCount(level) : total;
                    if (activeCount < 3) continue;

                    ProcessLevel(level, dtLevel, total, tick, dispatch, dispatchIndirect);
                }

                // Final position integration
                IntegratePositions(total, dispatch);

                // Update DT
                if (maxDtLevel >= 0) {
                    for (int level = maxDtLevel; level >= 0; level--) {
                        if (!m.TryGetLevelDt(level, out DT dtLevel) || dtLevel == null)
                            continue;

                        int activeCount = (level > 0) ? m.NodeCount(level) : total;
                        if (activeCount < 3) continue;

                        int pingRead = dtLevel.RenderPing ^ (tick & 1);
                        int pingWrite = pingRead ^ 1;

                        if (updateDtPositionsForRender) {
                            UpdateDtPositionsForLevel(level, dtLevel, activeCount, m, pingWrite, dispatch);
                        }

                        dtLevel.EnqueueMaintain(asyncCb, dtLevel.GetPositionsBuffer(pingWrite),
                            pingWrite, dtFixIterations, dtLegalizeIterations);
                    }
                }
            }

            // After all ticks, mark pending render ping for each DT level
            if (maxDtLevel >= 0) {
                int pingXor = tickCount & 1;
                for (int level = maxDtLevel; level >= 0; level--) {
                    if (m.TryGetLevelDt(level, out DT dtLevel) && dtLevel != null)
                        dtLevel.SetPendingRenderPing(dtLevel.RenderPing ^ pingXor);
                }
            }

            var fence = asyncCb.CreateGraphicsFence(GraphicsFenceType.AsyncQueueSynchronisation,
                SynchronisationStageFlags.ComputeProcessing);
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



        private void RebuildAllParents(Meshless m, int total, DispatchAction dispatch) {
            for (int level = m.maxLayer; level >= 1; level--) {
                if (!m.TryGetLevelDt(level, out DT dtLevel) || dtLevel == null)
                    continue;

                int activeCount = m.NodeCount(level);
                int fineCount = level > 1 ? m.NodeCount(level - 1) : total;
                if (fineCount <= activeCount)
                    continue;

                PrepareParentRebuildBuffers(dtLevel, activeCount, fineCount);

                dispatch(shader, kRebuildParentsAtLevel, ((fineCount - activeCount) + 255) / 256, 1, 1);
            }
            parentsBuilt = true;
        }

        private void ApplyForces(int total, DispatchAction dispatch) {
            asyncCb.SetComputeBufferParam(shader, kApplyGameplayForces, "_Vel", vel);
            asyncCb.SetComputeBufferParam(shader, kApplyGameplayForces, "_InvMass", invMass);
            asyncCb.SetComputeBufferParam(shader, kApplyGameplayForces, "_Flags", flags);

            asyncCb.SetComputeBufferParam(shader, kExternalForces, "_Vel", vel);
            asyncCb.SetComputeBufferParam(shader, kExternalForces, "_InvMass", invMass);
            asyncCb.SetComputeBufferParam(shader, kExternalForces, "_Flags", flags);

            if (forceEventsCount > 0) {
                asyncCb.SetComputeIntParam(shader, "_ForceEventCount", forceEventsCount);
                asyncCb.SetComputeBufferParam(shader, kApplyGameplayForces, "_ForceEvents", forceEvents);
                dispatch(shader, kApplyGameplayForces, (forceEventsCount + 255) / 256, 1, 1);
                forceEventsCount = 0;
            } else {
                asyncCb.SetComputeIntParam(shader, "_ForceEventCount", 0);
            }

            dispatch(shader, kExternalForces, (total + 255) / 256, 1, 1);
        }

        private void ProcessLevel(int level, DT dtLevel, int total, int tickIndex, DispatchAction dispatch, DispatchActionIndirect dispatchIndirect) {
            int activeCount = level > 0 ? NodeCount(level) : total;
            int fineCount = level > 1 ? NodeCount(level - 1) : total;

            // Set per‑level parameters
            PrepareRelaxBuffers(dtLevel, activeCount, fineCount, tickIndex);

            // Step 1: Save velocities prefix
            dispatch(shader, kSaveVelPrefix, (activeCount + 255) / 256, 1, 1);

            // Step 2: Caching
            dispatch(shader, kClearCurrentVolume, (activeCount + 255) / 256, 1, 1);
            dispatch(shader, kCacheVolumesHierarchical, (total + 255) / 256, 1, 1);
            dispatch(shader, kCacheKernelH, (activeCount + 255) / 256, 1, 1);
            dispatch(shader, kComputeCorrectionL, (activeCount + 255) / 256, 1, 1);
            dispatch(shader, kCacheF0AndResetLambda, (activeCount + 255) / 256, 1, 1);

            // Step 3: Coloring rebuild
            RebuildColoringForLevel(level, dtLevel, activeCount, dtLevel.NeighborCount,
                    tickIndex & 1, dispatch);

            // Step 4: Relaxation iterations
            int iterations = level == 0 ? Const.Iterations : Const.HPBDIterations;
            for (int iter = 0; iter < iterations; iter++) {
                for (int c = 0; c < ColoringMaxColors; c++) {
                    asyncCb.SetComputeIntParam(shader, "_ColorIndex", c);
                    dispatchIndirect(shader, kRelaxColored, relaxArgsByLevel[level], (uint)c * 12);
                }
            }

            // Step 5: Prolongation
            if (level > 0 && fineCount > activeCount)
                dispatch(shader, kProlongate, ((fineCount - activeCount) + 255) / 256, 1, 1);

            // Step 6: Commit deformation (only at finest level)
            if (level == 0)
                dispatch(shader, kCommitDeformation, (activeCount + 255) / 256, 1, 1);
        }

        private void RebuildColoringForLevel(int level, DT dtLevel,
            int activeCount, int neighborCount,
            int pingBufferIndex, DispatchAction dispatch) {

            PrepareColoringBuffers(dtLevel, level, activeCount, neighborCount, pingBufferIndex);

            int groups = (activeCount + 255) / 256;

            dispatch(shader, kColoringInit, groups, 1, 1);

            for (int r = 0; r < ColoringConflictRounds; r++) {
                dispatch(shader, kColoringDetectConflicts, groups, 1, 1);
                dispatch(shader, kColoringApply, groups, 1, 1);
                dispatch(shader, kColoringChoose, groups, 1, 1);
                dispatch(shader, kColoringApply, groups, 1, 1);
            }

            dispatch(shader, kColoringClearMeta, 1, 1, 1);
            dispatch(shader, kColoringBuildCounts, groups, 1, 1);
            dispatch(shader, kColoringBuildStarts, 1, 1, 1);
            dispatch(shader, kColoringScatterOrder, groups, 1, 1);
            dispatch(shader, kColoringBuildRelaxArgs, 1, 1, 1);
        }

        private void IntegratePositions(int total, DispatchAction dispatch) {
            asyncCb.SetComputeBufferParam(shader, kIntegratePositions, "_Pos", pos);
            asyncCb.SetComputeBufferParam(shader, kIntegratePositions, "_Vel", vel);
            asyncCb.SetComputeBufferParam(shader, kIntegratePositions, "_InvMass", invMass);
            asyncCb.SetComputeBufferParam(shader, kIntegratePositions, "_Flags", flags);
            dispatch(shader, kIntegratePositions, (total + 255) / 256, 1, 1);
        }

        private void UpdateDtPositionsForLevel(int level, DT dtLevel, int activeCount,
            Meshless m, int pingWrite, DispatchAction dispatch) {
            asyncCb.SetComputeIntParam(shader, "_ActiveCount", activeCount);
            asyncCb.SetComputeIntParam(shader, "_DtNeighborCount", dtLevel.NeighborCount);
            asyncCb.SetComputeBufferParam(shader, kUpdateDtPositions, "_Pos", pos);
            asyncCb.SetComputeBufferParam(shader, kUpdateDtPositions, "_DtPositions",
                dtLevel.GetPositionsBuffer(pingWrite));
            asyncCb.SetComputeVectorParam(shader, "_DtNormCenter",
                new Vector4(m.DtNormCenter.x, m.DtNormCenter.y, 0f, 0f));
            asyncCb.SetComputeFloatParam(shader, "_DtNormInvHalfExtent", m.DtNormInvHalfExtent);
            dispatch(shader, kUpdateDtPositions, (activeCount + 255) / 256, 1, 1);
        }
        private int NodeCount(int level) {
            return meshless.NodeCount(level);
        }

        public void Dispose() {
            Release();
        }

        private void InitializeFromMeshless(Meshless m) {
            meshless = m;
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

        private void Release() {
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

            if (asyncCb != null) {
                asyncCb.Dispose();
                asyncCb = null;
            }

            kernelsCached = false;
        }
    }
}