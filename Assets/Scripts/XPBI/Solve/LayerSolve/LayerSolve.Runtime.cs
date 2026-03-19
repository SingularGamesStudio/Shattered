using GPU.Delaunay;
using GPU.Neighbors;
using UnityEngine;

namespace GPU.Solver {
    internal sealed partial class LayerSolve {
        private ComputeBuffer currentVolumeBits;
        private ComputeBuffer currentTotalMassBits;
        private ComputeBuffer fixedChildPosBits;
        private ComputeBuffer fixedChildCount;
        private ComputeBuffer L;
        private ComputeBuffer F0;
        private ComputeBuffer lambda;
        private ComputeBuffer durabilityLambda;
        private ComputeBuffer collisionLambda;
        private ComputeBuffer savedVelPrefix;
        private ComputeBuffer velDeltaBits;
        private ComputeBuffer velPrev;
        private ComputeBuffer lambdaPrev;
        private ComputeBuffer jrVelDeltaBits;
        private ComputeBuffer jrLambdaDelta;
        private ComputeBuffer coarseFixed;
        private ComputeBuffer restrictedDeltaVBits;
        private ComputeBuffer restrictedDeltaVCount;
        private ComputeBuffer restrictedDeltaVAvg;

        internal int kClearHierarchicalStats;
        internal int kCacheHierarchicalStats;
        internal int kFinalizeHierarchicalStats;
        internal int kComputeCorrectionL;
        internal int kCacheF0AndResetLambda;
        internal int kSaveVelPrefix;
        internal int kClearVelDelta;
        internal int kResetCollisionLambda;
        internal int kRelaxColored;
        internal int kRelaxColoredPersistentCoarse;
        internal int kJRSavePrevAndClear;
        internal int kJRComputeDeltas;
        internal int kJRApply;
        internal int kProlongate;
        internal int kCommitDeformation;
        internal int kClearRestrictedDeltaV;
        internal int kRestrictGameplayDeltaVFromEvents;
        internal int kRestrictFineVelocityResidualToActive;
        internal int kApplyRestrictedDeltaVToActiveAndPrefix;
        internal int kRemoveRestrictedDeltaVFromActive;
        internal int kSmoothProlongatedFineVel;

        private ComputeBuffer pos => solver.pos;
        private ComputeBuffer vel => solver.vel;
        private ComputeBuffer materialIds => solver.materialIds;
        private ComputeBuffer invMass => solver.invMass;
        private ComputeBuffer restVolume => solver.restVolume;
        private ComputeBuffer parentIndex => solver.parentIndex;
        private ComputeBuffer parentIndices => solver.parentIndices;
        private ComputeBuffer parentWeights => solver.parentWeights;
        private ComputeBuffer F => solver.F;
        private ComputeBuffer Fp => solver.Fp;
        private ComputeBuffer collisionEvents => solver.collisionEvent.CollisionEventsBuffer;
        private ComputeBuffer collisionEventCount => solver.collisionEvent.CollisionEventCountBuffer;
        private ComputeBuffer xferColCount => solver.collisionEvent.XferColCountBuffer;
        private ComputeBuffer xferColNXBits => solver.collisionEvent.XferColNXBitsBuffer;
        private ComputeBuffer xferColNYBits => solver.collisionEvent.XferColNYBitsBuffer;
        private ComputeBuffer xferColPenBits => solver.collisionEvent.XferColPenBitsBuffer;
        private UnityEngine.Rendering.CommandBuffer asyncCb => solver.asyncCb;
        private ComputeShader shader => solver.shader;
        private ComputeBuffer convergenceDebug => solver.solverDebug.ConvergenceDebug;
        private ComputeBuffer convergenceDebugFallback => solver.solverDebug.ConvergenceDebugFallback;
        private ComputeBuffer prolongationConstraintDebug => solver.solverDebug.ProlongationConstraintDebug;
        private ComputeBuffer defaultDtOwnerByLocal => solver.layerMappingCache.DefaultDtOwnerByLocal;
        private int kClearCollisionEventCount => solver.collisionEvent.ClearCollisionEventCountKernel;
        private int kBuildCollisionEventsL0 => solver.collisionEvent.BuildCollisionEventsL0Kernel;
        private int kClearTransferredCollision => solver.collisionEvent.ClearTransferredCollisionKernel;
        private int kRestrictCollisionEventsToActivePairs => solver.collisionEvent.RestrictCollisionEventsToActivePairsKernel;
        private int kClearConvergenceDebugStats => solver.solverDebug.ClearConvergenceDebugStatsKernel;

        private const int UseTransferredCollisionsDisabled = 0;
        private const int UseTransferredCollisionsEnabled = 1;

        private void BindDtGlobalMappingParams(int kernel, bool useDtGlobalNodeMap, int dtLocalBase, ComputeBuffer dtGlobalNodeMap, ComputeBuffer dtGlobalToLayerLocalMap) {
            solver.layerMappingCache.BindDtGlobalMappingParams(kernel, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
        }

        internal void SetForceEventCountParam(int forceCount) {
            solver.asyncCb.SetComputeIntParam(solver.shader, "_ForceEventCount", forceCount);
        }

        internal void BindRestrictedGameplayEventsBuffer() {
            solver.asyncCb.SetComputeBufferParam(solver.shader, kRestrictGameplayDeltaVFromEvents, "_ForceEvents", solver.gameplayForce.ForceEventsBuffer);
        }

        internal void SetUseTransferredCollisionsParam(bool enabled) {
            solver.asyncCb.SetComputeIntParam(solver.shader, "_UseTransferredCollisions", enabled ? UseTransferredCollisionsEnabled : UseTransferredCollisionsDisabled);
        }

        internal void SetRestrictedDeltaVScale(float scale) {
            solver.asyncCb.SetComputeFloatParam(solver.shader, "_RestrictedDeltaVScale", scale);
        }

        internal void BindLayerColoringBuffers(DTColoring layerColoring) {
            solver.asyncCb.SetComputeBufferParam(solver.shader, kRelaxColored, "_ColorOrder", layerColoring.OrderBuffer);
            solver.asyncCb.SetComputeBufferParam(solver.shader, kRelaxColored, "_ColorStarts", layerColoring.StartsBuffer);
            solver.asyncCb.SetComputeBufferParam(solver.shader, kRelaxColored, "_ColorCounts", layerColoring.CountsBuffer);
            solver.asyncCb.SetComputeBufferParam(solver.shader, kRelaxColoredPersistentCoarse, "_ColorOrder", layerColoring.OrderBuffer);
            solver.asyncCb.SetComputeBufferParam(solver.shader, kRelaxColoredPersistentCoarse, "_ColorStarts", layerColoring.StartsBuffer);
            solver.asyncCb.SetComputeBufferParam(solver.shader, kRelaxColoredPersistentCoarse, "_ColorCounts", layerColoring.CountsBuffer);
        }

        internal void SetConvergenceDebugDisabled() {
            solver.asyncCb.SetComputeIntParam(solver.shader, "_ConvergenceDebugEnable", 0);
        }

        internal void SetCollisionEnable(bool enabled) {
            solver.asyncCb.SetComputeIntParam(solver.shader, "_CollisionEnable", enabled ? 1 : 0);
        }

        internal void SetPersistentRelaxParams(int persistentIterations, int baseDebugIter) {
            solver.asyncCb.SetComputeIntParam(solver.shader, "_PersistentIters", persistentIterations);
            solver.asyncCb.SetComputeIntParam(solver.shader, "_PersistentBaseDebugIter", baseDebugIter);
        }

        internal void SetConvergenceDebugIter(int iter) {
            solver.asyncCb.SetComputeIntParam(solver.shader, "_ConvergenceDebugIter", iter);
        }

        internal void SetColorIndex(int color) {
            solver.asyncCb.SetComputeIntParam(solver.shader, "_ColorIndex", color);
        }

        internal void SetJRParams() {
            solver.asyncCb.SetComputeFloatParam(solver.shader, "_JROmegaV", Const.JROmegaV);
            solver.asyncCb.SetComputeFloatParam(solver.shader, "_JROmegaL", Const.JROmegaL);
        }

        public struct DtMappingContext {
            public readonly bool UseDtGlobalNodeMap;
            public readonly int DtLocalBase;
            public readonly ComputeBuffer DtGlobalNodeMap;
            public readonly ComputeBuffer DtGlobalToLayerLocalMap;
            public readonly ComputeBuffer DtOwnerByLocal;

            public DtMappingContext(
                bool useDtGlobalNodeMap,
                int dtLocalBase,
                ComputeBuffer dtGlobalNodeMap,
                ComputeBuffer dtGlobalToLayerLocalMap,
                ComputeBuffer dtOwnerByLocal
            ) {
                UseDtGlobalNodeMap = useDtGlobalNodeMap;
                DtLocalBase = dtLocalBase;
                DtGlobalNodeMap = dtGlobalNodeMap;
                DtGlobalToLayerLocalMap = dtGlobalToLayerLocalMap;
                DtOwnerByLocal = dtOwnerByLocal;
            }
        }

        internal readonly struct RelaxBufferContext {
            public readonly INeighborSearch NeighborSearch;
            public readonly int BaseIndex;
            public readonly int ActiveCount;
            public readonly int FineCount;
            public readonly int TickIndex;
            public readonly float LayerKernelH;
            public readonly DtMappingContext Mapping;

            public RelaxBufferContext(
                INeighborSearch neighborSearch,
                int baseIndex,
                int activeCount,
                int fineCount,
                int tickIndex,
                float layerKernelH,
                DtMappingContext mapping
            ) {
                NeighborSearch = neighborSearch;
                BaseIndex = baseIndex;
                ActiveCount = activeCount;
                FineCount = fineCount;
                TickIndex = tickIndex;
                LayerKernelH = layerKernelH;
                Mapping = mapping;
            }
        }

        internal int KClearHierarchicalStats => kClearHierarchicalStats;
        internal int KCacheHierarchicalStats => kCacheHierarchicalStats;
        internal int KFinalizeHierarchicalStats => kFinalizeHierarchicalStats;
        internal int KComputeCorrectionL => kComputeCorrectionL;
        internal int KCacheF0AndResetLambda => kCacheF0AndResetLambda;
        internal int KSaveVelPrefix => kSaveVelPrefix;
        internal int KClearVelDelta => kClearVelDelta;
        internal int KResetCollisionLambda => kResetCollisionLambda;
        internal int KRelaxColored => kRelaxColored;
        internal int KRelaxColoredPersistentCoarse => kRelaxColoredPersistentCoarse;
        internal int KJRSavePrevAndClear => kJRSavePrevAndClear;
        internal int KJRComputeDeltas => kJRComputeDeltas;
        internal int KJRApply => kJRApply;
        internal int KProlongate => kProlongate;
        internal int KCommitDeformation => kCommitDeformation;
        internal int KClearRestrictedDeltaV => kClearRestrictedDeltaV;
        internal int KRestrictGameplayDeltaVFromEvents => kRestrictGameplayDeltaVFromEvents;
        internal int KRestrictFineVelocityResidualToActive => kRestrictFineVelocityResidualToActive;
        internal int KApplyRestrictedDeltaVToActiveAndPrefix => kApplyRestrictedDeltaVToActiveAndPrefix;
        internal int KRemoveRestrictedDeltaVFromActive => kRemoveRestrictedDeltaVFromActive;
        internal int KSmoothProlongatedFineVel => kSmoothProlongatedFineVel;

        internal void AllocateRuntimeBuffers(int newCapacity) {
            currentVolumeBits = new ComputeBuffer(newCapacity, sizeof(uint), ComputeBufferType.Structured);
            currentTotalMassBits = new ComputeBuffer(newCapacity, sizeof(uint), ComputeBufferType.Structured);
            fixedChildPosBits = new ComputeBuffer(newCapacity * 2, sizeof(uint), ComputeBufferType.Structured);
            fixedChildCount = new ComputeBuffer(newCapacity, sizeof(uint), ComputeBufferType.Structured);
            L = new ComputeBuffer(newCapacity, sizeof(float) * 4, ComputeBufferType.Structured);
            F0 = new ComputeBuffer(newCapacity, sizeof(float) * 4, ComputeBufferType.Structured);
            lambda = new ComputeBuffer(newCapacity, sizeof(float), ComputeBufferType.Structured);
            durabilityLambda = new ComputeBuffer(newCapacity * Const.NeighborCount, sizeof(float), ComputeBufferType.Structured);
            collisionLambda = new ComputeBuffer(newCapacity * Const.NeighborCount, sizeof(float), ComputeBufferType.Structured);
            savedVelPrefix = new ComputeBuffer(newCapacity, sizeof(float) * 2, ComputeBufferType.Structured);
            velDeltaBits = new ComputeBuffer(newCapacity * 2, sizeof(uint), ComputeBufferType.Structured);
            velPrev = new ComputeBuffer(newCapacity, sizeof(float) * 2, ComputeBufferType.Structured);
            lambdaPrev = new ComputeBuffer(newCapacity, sizeof(float), ComputeBufferType.Structured);
            jrVelDeltaBits = new ComputeBuffer(newCapacity * 2, sizeof(uint), ComputeBufferType.Structured);
            jrLambdaDelta = new ComputeBuffer(newCapacity, sizeof(float), ComputeBufferType.Structured);
            coarseFixed = new ComputeBuffer(newCapacity, sizeof(uint), ComputeBufferType.Structured);
            restrictedDeltaVBits = new ComputeBuffer(newCapacity * 2, sizeof(uint), ComputeBufferType.Structured);
            restrictedDeltaVCount = new ComputeBuffer(newCapacity, sizeof(uint), ComputeBufferType.Structured);
            restrictedDeltaVAvg = new ComputeBuffer(newCapacity, sizeof(float) * 2, ComputeBufferType.Structured);
        }

        internal void ReleaseRuntimeBuffers() {
            currentVolumeBits?.Dispose(); currentVolumeBits = null;
            currentTotalMassBits?.Dispose(); currentTotalMassBits = null;
            fixedChildPosBits?.Dispose(); fixedChildPosBits = null;
            fixedChildCount?.Dispose(); fixedChildCount = null;
            L?.Dispose(); L = null;
            F0?.Dispose(); F0 = null;
            lambda?.Dispose(); lambda = null;
            durabilityLambda?.Dispose(); durabilityLambda = null;
            collisionLambda?.Dispose(); collisionLambda = null;
            savedVelPrefix?.Dispose(); savedVelPrefix = null;
            velDeltaBits?.Dispose(); velDeltaBits = null;
            velPrev?.Dispose(); velPrev = null;
            lambdaPrev?.Dispose(); lambdaPrev = null;
            jrVelDeltaBits?.Dispose(); jrVelDeltaBits = null;
            jrLambdaDelta?.Dispose(); jrLambdaDelta = null;
            coarseFixed?.Dispose(); coarseFixed = null;
            restrictedDeltaVBits?.Dispose(); restrictedDeltaVBits = null;
            restrictedDeltaVCount?.Dispose(); restrictedDeltaVCount = null;
            restrictedDeltaVAvg?.Dispose(); restrictedDeltaVAvg = null;
        }

        internal void CacheRuntimeKernels() {
            kClearHierarchicalStats = solver.shader.FindKernel("ClearHierarchicalStats");
            kCacheHierarchicalStats = solver.shader.FindKernel("CacheHierarchicalStats");
            kFinalizeHierarchicalStats = solver.shader.FindKernel("FinalizeHierarchicalStats");
            kComputeCorrectionL = solver.shader.FindKernel("ComputeCorrectionL");
            kCacheF0AndResetLambda = solver.shader.FindKernel("CacheF0AndResetLambda");
            kSaveVelPrefix = solver.shader.FindKernel("SaveVelPrefix");
            kClearVelDelta = solver.shader.FindKernel("ClearVelDelta");
            kResetCollisionLambda = solver.shader.FindKernel("ResetCollisionLambda");
            kRelaxColored = solver.shader.FindKernel("RelaxColored");
            kRelaxColoredPersistentCoarse = solver.shader.FindKernel("RelaxColoredPersistentCoarse");
            kJRSavePrevAndClear = solver.shader.FindKernel("JR_SavePrevAndClear");
            kJRComputeDeltas = solver.shader.FindKernel("JR_ComputeDeltas");
            kJRApply = solver.shader.FindKernel("JR_Apply");
            kProlongate = solver.shader.FindKernel("Prolongate");
            kCommitDeformation = solver.shader.FindKernel("CommitDeformation");
            kClearRestrictedDeltaV = solver.shader.FindKernel("ClearRestrictedDeltaV");
            kRestrictGameplayDeltaVFromEvents = solver.shader.FindKernel("RestrictGameplayDeltaVFromEvents");
            kRestrictFineVelocityResidualToActive = solver.shader.FindKernel("RestrictFineVelocityResidualToActive");
            kApplyRestrictedDeltaVToActiveAndPrefix = solver.shader.FindKernel("ApplyRestrictedDeltaVToActiveAndPrefix");
            kRemoveRestrictedDeltaVFromActive = solver.shader.FindKernel("RemoveRestrictedDeltaVFromActive");
            kSmoothProlongatedFineVel = solver.shader.FindKernel("SmoothProlongatedFineVel");
        }

        internal void PrepareRelaxBuffers(in RelaxBufferContext context) {
            INeighborSearch neighborSearch = context.NeighborSearch;
            int baseIndex = context.BaseIndex;
            int activeCount = context.ActiveCount;
            int fineCount = context.FineCount;
            float layerKernelH = context.LayerKernelH;
            bool useDtGlobalNodeMap = context.Mapping.UseDtGlobalNodeMap;
            int dtLocalBase = context.Mapping.DtLocalBase;
            ComputeBuffer dtGlobalNodeMap = context.Mapping.DtGlobalNodeMap;
            ComputeBuffer dtGlobalToLayerLocalMap = context.Mapping.DtGlobalToLayerLocalMap;
            ComputeBuffer dtOwnerByLocal = context.Mapping.DtOwnerByLocal;
            var matLib = MaterialLibrary.Instance;
            var physicalParams = matLib != null ? matLib.PhysicalParamsBuffer : null;
            int physicalParamCount = (matLib != null && physicalParams != null) ? matLib.MaterialCount : 0;

            solver.asyncCb.SetComputeIntParam(solver.shader, "_Base", baseIndex);
            solver.asyncCb.SetComputeIntParam(solver.shader, "_ActiveCount", activeCount);
            solver.asyncCb.SetComputeIntParam(solver.shader, "_FineCount", fineCount);
            solver.asyncCb.SetComputeIntParam(solver.shader, "_DtNeighborCount", neighborSearch.NeighborCount);
            solver.asyncCb.SetComputeIntParam(solver.shader, "_PhysicalParamCount", physicalParamCount);
            solver.asyncCb.SetComputeFloatParam(solver.shader, "_LayerKernelH", layerKernelH);
            solver.asyncCb.SetComputeIntParam(solver.shader, "_UseDtOwnerFilter", dtOwnerByLocal != null ? 1 : 0);
            solver.asyncCb.SetComputeIntParam(solver.shader, "_CollisionEventCapacity", solver.collisionEvent.CollisionEventsBuffer != null ? solver.collisionEvent.CollisionEventsBuffer.count : 0);

            ComputeBuffer convergenceDebugBinding = solver.solverDebug.ConvergenceDebug ?? solver.solverDebug.ProlongationConstraintDebug ?? solver.solverDebug.ConvergenceDebugFallback;
            solver.asyncCb.SetComputeBufferParam(solver.shader, kRelaxColored, "_ConvergenceDebug", convergenceDebugBinding);
            solver.asyncCb.SetComputeBufferParam(solver.shader, kRelaxColoredPersistentCoarse, "_ConvergenceDebug", convergenceDebugBinding);
            solver.asyncCb.SetComputeBufferParam(solver.shader, kJRComputeDeltas, "_ConvergenceDebug", convergenceDebugBinding);
            solver.asyncCb.SetComputeBufferParam(solver.shader, solver.solverDebug.ClearConvergenceDebugStatsKernel, "_ConvergenceDebug", convergenceDebugBinding);

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
            asyncCb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_ParentIndices", parentIndices);
            asyncCb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_ParentWeights", parentWeights);
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
            asyncCb.SetComputeBufferParam(shader, kResetCollisionLambda, "_DurabilityLambda", durabilityLambda);
            asyncCb.SetComputeBufferParam(shader, kResetCollisionLambda, "_CollisionLambda", collisionLambda);
            asyncCb.SetComputeBufferParam(shader, kClearCollisionEventCount, "_CollisionEventCount", collisionEventCount);
            asyncCb.SetComputeBufferParam(shader, kBuildCollisionEventsL0, "_Pos", pos);
            asyncCb.SetComputeBufferParam(shader, kBuildCollisionEventsL0, "_DtNeighbors", neighborSearch.NeighborsBuffer);
            asyncCb.SetComputeBufferParam(shader, kBuildCollisionEventsL0, "_DtNeighborCounts", neighborSearch.NeighborCountsBuffer);
            asyncCb.SetComputeBufferParam(shader, kBuildCollisionEventsL0, "_DtOwnerByLocal", dtOwnerByLocal ?? defaultDtOwnerByLocal);
            asyncCb.SetComputeBufferParam(shader, kBuildCollisionEventsL0, "_CollisionEvents", collisionEvents);
            asyncCb.SetComputeBufferParam(shader, kBuildCollisionEventsL0, "_CollisionEventCount", collisionEventCount);
            asyncCb.SetComputeBufferParam(shader, kClearTransferredCollision, "_XferColCount", xferColCount);
            asyncCb.SetComputeBufferParam(shader, kClearTransferredCollision, "_XferColNXBits", xferColNXBits);
            asyncCb.SetComputeBufferParam(shader, kClearTransferredCollision, "_XferColNYBits", xferColNYBits);
            asyncCb.SetComputeBufferParam(shader, kClearTransferredCollision, "_XferColPenBits", xferColPenBits);
            asyncCb.SetComputeBufferParam(shader, kRestrictCollisionEventsToActivePairs, "_ParentIndex", parentIndex);
            asyncCb.SetComputeBufferParam(shader, kRestrictCollisionEventsToActivePairs, "_ParentIndices", parentIndices);
            asyncCb.SetComputeBufferParam(shader, kRestrictCollisionEventsToActivePairs, "_ParentWeights", parentWeights);
            asyncCb.SetComputeBufferParam(shader, kRestrictCollisionEventsToActivePairs, "_DtNeighbors", neighborSearch.NeighborsBuffer);
            asyncCb.SetComputeBufferParam(shader, kRestrictCollisionEventsToActivePairs, "_DtNeighborCounts", neighborSearch.NeighborCountsBuffer);
            asyncCb.SetComputeBufferParam(shader, kRestrictCollisionEventsToActivePairs, "_CollisionEvents", collisionEvents);
            asyncCb.SetComputeBufferParam(shader, kRestrictCollisionEventsToActivePairs, "_CollisionEventCount", collisionEventCount);
            asyncCb.SetComputeBufferParam(shader, kRestrictCollisionEventsToActivePairs, "_XferColCount", xferColCount);
            asyncCb.SetComputeBufferParam(shader, kRestrictCollisionEventsToActivePairs, "_XferColNXBits", xferColNXBits);
            asyncCb.SetComputeBufferParam(shader, kRestrictCollisionEventsToActivePairs, "_XferColNYBits", xferColNYBits);
            asyncCb.SetComputeBufferParam(shader, kRestrictCollisionEventsToActivePairs, "_XferColPenBits", xferColPenBits);
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
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_DurabilityLambda", durabilityLambda);
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_CollisionLambda", collisionLambda);
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_XferColCount", xferColCount);
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_XferColNXBits", xferColNXBits);
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_XferColNYBits", xferColNYBits);
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_XferColPenBits", xferColPenBits);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_Pos", pos);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_Vel", vel);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_MaterialIds", materialIds);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_InvMass", invMass);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_RestVolume", restVolume);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_F0", F0);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_L", L);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_CurrentVolumeBits", currentVolumeBits);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_CurrentTotalMassBits", currentTotalMassBits);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_FixedChildPosBits", fixedChildPosBits);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_FixedChildCount", fixedChildCount);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_Lambda", lambda);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_DurabilityLambda", durabilityLambda);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_CollisionLambda", collisionLambda);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_XferColCount", xferColCount);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_XferColNXBits", xferColNXBits);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_XferColNYBits", xferColNYBits);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_XferColPenBits", xferColPenBits);

            asyncCb.SetComputeBufferParam(shader, kProlongate, "_InvMass", invMass);
            asyncCb.SetComputeBufferParam(shader, kProlongate, "_Vel", vel);
            asyncCb.SetComputeBufferParam(shader, kProlongate, "_Pos", pos);
            asyncCb.SetComputeBufferParam(shader, kProlongate, "_MaterialIds", materialIds);
            asyncCb.SetComputeBufferParam(shader, kProlongate, "_RestVolume", restVolume);
            asyncCb.SetComputeBufferParam(shader, kProlongate, "_CurrentVolumeBits", currentVolumeBits);
            asyncCb.SetComputeBufferParam(shader, kProlongate, "_FixedChildPosBits", fixedChildPosBits);
            asyncCb.SetComputeBufferParam(shader, kProlongate, "_FixedChildCount", fixedChildCount);
            asyncCb.SetComputeBufferParam(shader, kProlongate, "_ParentIndex", parentIndex);
            asyncCb.SetComputeBufferParam(shader, kProlongate, "_ParentIndices", parentIndices);
            asyncCb.SetComputeBufferParam(shader, kProlongate, "_ParentWeights", parentWeights);
            asyncCb.SetComputeBufferParam(shader, kProlongate, "_SavedVelPrefix", savedVelPrefix);


            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_Pos", pos);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_Vel", vel);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_MaterialIds", materialIds);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_InvMass", invMass);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_F0", F0);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_L", L);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_CurrentVolumeBits", currentVolumeBits);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_FixedChildPosBits", fixedChildPosBits);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_FixedChildCount", fixedChildCount);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_F", F);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_Fp", Fp);

            asyncCb.SetComputeBufferParam(shader, kComputeCorrectionL, "_DtNeighbors", neighborSearch.NeighborsBuffer);
            asyncCb.SetComputeBufferParam(shader, kComputeCorrectionL, "_DtNeighborCounts", neighborSearch.NeighborCountsBuffer);
            asyncCb.SetComputeBufferParam(shader, kComputeCorrectionL, "_DtOwnerByLocal", dtOwnerByLocal ?? defaultDtOwnerByLocal);
            BindDtGlobalMappingParams(kComputeCorrectionL, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);

            BindDtGlobalMappingParams(kClearHierarchicalStats, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(kCacheHierarchicalStats, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(kFinalizeHierarchicalStats, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(kCacheF0AndResetLambda, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(kSaveVelPrefix, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(kClearVelDelta, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(kResetCollisionLambda, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(kBuildCollisionEventsL0, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(kClearTransferredCollision, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(kRestrictCollisionEventsToActivePairs, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(kClearRestrictedDeltaV, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(kRestrictGameplayDeltaVFromEvents, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(kRestrictFineVelocityResidualToActive, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(kApplyRestrictedDeltaVToActiveAndPrefix, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(kRemoveRestrictedDeltaVFromActive, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(kProlongate, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);

            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_DtNeighbors", neighborSearch.NeighborsBuffer);
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_DtNeighborCounts", neighborSearch.NeighborCountsBuffer);
            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_DtOwnerByLocal", dtOwnerByLocal ?? defaultDtOwnerByLocal);
            BindDtGlobalMappingParams(kRelaxColored, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_DtNeighbors", neighborSearch.NeighborsBuffer);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_DtNeighborCounts", neighborSearch.NeighborCountsBuffer);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_DtOwnerByLocal", dtOwnerByLocal ?? defaultDtOwnerByLocal);
            BindDtGlobalMappingParams(kRelaxColoredPersistentCoarse, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);

            asyncCb.SetComputeBufferParam(shader, kJRSavePrevAndClear, "_Vel", vel);
            asyncCb.SetComputeBufferParam(shader, kJRSavePrevAndClear, "_InvMass", invMass);
            asyncCb.SetComputeBufferParam(shader, kJRSavePrevAndClear, "_CoarseFixed", coarseFixed);
            asyncCb.SetComputeBufferParam(shader, kJRSavePrevAndClear, "_Lambda", lambda);
            asyncCb.SetComputeBufferParam(shader, kJRSavePrevAndClear, "_VelPrev", velPrev);
            asyncCb.SetComputeBufferParam(shader, kJRSavePrevAndClear, "_LambdaPrev", lambdaPrev);
            asyncCb.SetComputeBufferParam(shader, kJRSavePrevAndClear, "_JRVelDeltaBits", jrVelDeltaBits);
            asyncCb.SetComputeBufferParam(shader, kJRSavePrevAndClear, "_JRLambdaDelta", jrLambdaDelta);

            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_Pos", pos);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_MaterialIds", materialIds);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_InvMass", invMass);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_RestVolume", restVolume);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_F0", F0);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_L", L);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_CurrentVolumeBits", currentVolumeBits);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_CurrentTotalMassBits", currentTotalMassBits);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_FixedChildPosBits", fixedChildPosBits);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_FixedChildCount", fixedChildCount);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_VelPrev", velPrev);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_LambdaPrev", lambdaPrev);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_DurabilityLambda", durabilityLambda);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_JRVelDeltaBits", jrVelDeltaBits);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_JRLambdaDelta", jrLambdaDelta);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_DtNeighbors", neighborSearch.NeighborsBuffer);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_DtNeighborCounts", neighborSearch.NeighborCountsBuffer);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_DtOwnerByLocal", dtOwnerByLocal ?? defaultDtOwnerByLocal);
            asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_CoarseFixed", coarseFixed);

            asyncCb.SetComputeBufferParam(shader, kJRApply, "_Vel", vel);
            asyncCb.SetComputeBufferParam(shader, kJRApply, "_Lambda", lambda);
            asyncCb.SetComputeBufferParam(shader, kJRApply, "_Pos", pos);
            asyncCb.SetComputeBufferParam(shader, kJRApply, "_MaterialIds", materialIds);
            asyncCb.SetComputeBufferParam(shader, kJRApply, "_InvMass", invMass);
            asyncCb.SetComputeBufferParam(shader, kJRApply, "_RestVolume", restVolume);
            asyncCb.SetComputeBufferParam(shader, kJRApply, "_CurrentVolumeBits", currentVolumeBits);
            asyncCb.SetComputeBufferParam(shader, kJRApply, "_FixedChildPosBits", fixedChildPosBits);
            asyncCb.SetComputeBufferParam(shader, kJRApply, "_FixedChildCount", fixedChildCount);
            asyncCb.SetComputeBufferParam(shader, kJRApply, "_CoarseFixed", coarseFixed);
            asyncCb.SetComputeBufferParam(shader, kJRApply, "_JRVelDeltaBits", jrVelDeltaBits);
            asyncCb.SetComputeBufferParam(shader, kJRApply, "_JRLambdaDelta", jrLambdaDelta);

            BindDtGlobalMappingParams(kJRSavePrevAndClear, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(kJRComputeDeltas, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(kJRApply, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);

            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_DtNeighbors", neighborSearch.NeighborsBuffer);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_DtNeighborCounts", neighborSearch.NeighborCountsBuffer);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_DtOwnerByLocal", dtOwnerByLocal ?? defaultDtOwnerByLocal);
            BindDtGlobalMappingParams(kCommitDeformation, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);

            asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_CoarseFixed", coarseFixed);
            asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_CoarseFixed", coarseFixed);
            asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_CoarseFixed", coarseFixed);

            asyncCb.SetComputeBufferParam(shader, kClearRestrictedDeltaV, "_RestrictedDeltaVBits", restrictedDeltaVBits);
            asyncCb.SetComputeBufferParam(shader, kClearRestrictedDeltaV, "_RestrictedDeltaVCount", restrictedDeltaVCount);
            asyncCb.SetComputeBufferParam(shader, kClearRestrictedDeltaV, "_RestrictedDeltaVAvg", restrictedDeltaVAvg);

            asyncCb.SetComputeBufferParam(shader, kRestrictGameplayDeltaVFromEvents, "_RestrictedDeltaVBits", restrictedDeltaVBits);
            asyncCb.SetComputeBufferParam(shader, kRestrictGameplayDeltaVFromEvents, "_RestrictedDeltaVCount", restrictedDeltaVCount);
            asyncCb.SetComputeBufferParam(shader, kRestrictGameplayDeltaVFromEvents, "_InvMass", invMass);
            asyncCb.SetComputeBufferParam(shader, kRestrictGameplayDeltaVFromEvents, "_ParentIndex", parentIndex);
            asyncCb.SetComputeBufferParam(shader, kRestrictGameplayDeltaVFromEvents, "_ParentIndices", parentIndices);
            asyncCb.SetComputeBufferParam(shader, kRestrictGameplayDeltaVFromEvents, "_ParentWeights", parentWeights);
            asyncCb.SetComputeBufferParam(shader, kRestrictGameplayDeltaVFromEvents, "_ForceEvents", solver.gameplayForce.ForceEventsBuffer);

            asyncCb.SetComputeBufferParam(shader, kRestrictFineVelocityResidualToActive, "_Vel", vel);
            asyncCb.SetComputeBufferParam(shader, kRestrictFineVelocityResidualToActive, "_InvMass", invMass);
            asyncCb.SetComputeBufferParam(shader, kRestrictFineVelocityResidualToActive, "_ParentIndex", parentIndex);
            asyncCb.SetComputeBufferParam(shader, kRestrictFineVelocityResidualToActive, "_ParentIndices", parentIndices);
            asyncCb.SetComputeBufferParam(shader, kRestrictFineVelocityResidualToActive, "_ParentWeights", parentWeights);
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
            asyncCb.SetComputeBufferParam(shader, kSmoothProlongatedFineVel, "_DtNeighbors", neighborSearch.NeighborsBuffer);
            asyncCb.SetComputeBufferParam(shader, kSmoothProlongatedFineVel, "_DtNeighborCounts", neighborSearch.NeighborCountsBuffer);
            asyncCb.SetComputeBufferParam(shader, kSmoothProlongatedFineVel, "_DtOwnerByLocal", dtOwnerByLocal ?? defaultDtOwnerByLocal);
            BindDtGlobalMappingParams(kSmoothProlongatedFineVel, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);

            if (physicalParams != null) {
                asyncCb.SetComputeBufferParam(shader, kRelaxColored, "_PhysicalParams", physicalParams);
                asyncCb.SetComputeBufferParam(shader, kRelaxColoredPersistentCoarse, "_PhysicalParams", physicalParams);
                asyncCb.SetComputeBufferParam(shader, kJRComputeDeltas, "_PhysicalParams", physicalParams);
                asyncCb.SetComputeBufferParam(shader, kJRApply, "_PhysicalParams", physicalParams);
                asyncCb.SetComputeBufferParam(shader, kProlongate, "_PhysicalParams", physicalParams);
                asyncCb.SetComputeBufferParam(shader, kCommitDeformation, "_PhysicalParams", physicalParams);
            }
        }
    }
}
