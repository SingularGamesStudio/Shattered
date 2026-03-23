using GPU.Neighbors;
using UnityEngine;
using UnityEngine.Rendering;

namespace GPU.Solver {
    internal sealed class LayerCacheRuntime {
        private readonly XPBISolver solver;
        private readonly ComputeShader shader;
        private readonly ComputeShader collisionShader;

        public LayerCacheRuntime(XPBISolver solver) {
            this.solver = solver;
            shader = solver.LayerCacheShader;
            collisionShader = solver.CollisionEventsShader;
        }

        internal XPBISolver Solver => solver;

        internal int kClearHierarchicalStats;
        internal int kCacheHierarchicalStats;
        internal int kFinalizeHierarchicalStats;
        internal int kComputeCorrectionL;
        internal int kCacheF0AndResetLambda;
        internal int kSaveVelPrefix;
        internal int kClearVelDelta;
        internal int kResetCollisionLambda;
        internal int kClearRestrictedDeltaV;
        internal int kRestrictGameplayDeltaVFromEvents;
        internal int kRestrictFineVelocityResidualToActive;
        internal int kApplyRestrictedDeltaVToActiveAndPrefix;
        internal int kRemoveRestrictedDeltaVFromActive;

        private ComputeBuffer pos => solver.pos;
        private ComputeBuffer vel => solver.vel;
        private ComputeBuffer invMass => solver.invMass;
        private ComputeBuffer restVolume => solver.restVolume;
        private ComputeBuffer parentIndex => solver.parentIndex;
        private ComputeBuffer parentIndices => solver.parentIndices;
        private ComputeBuffer parentWeights => solver.parentWeights;
        private ComputeBuffer F => solver.F;
        private ComputeBuffer collisionEvents => solver.collisionEvent.CollisionEventsBuffer;
        private ComputeBuffer collisionEventCount => solver.collisionEvent.CollisionEventCountBuffer;
        private ComputeBuffer xferColCount => solver.collisionEvent.XferColCountBuffer;
        private ComputeBuffer xferColNXBits => solver.collisionEvent.XferColNXBitsBuffer;
        private ComputeBuffer xferColNYBits => solver.collisionEvent.XferColNYBitsBuffer;
        private ComputeBuffer xferColPenBits => solver.collisionEvent.XferColPenBitsBuffer;
        private ComputeBuffer xferColSBits => solver.collisionEvent.XferColSBitsBuffer;
        private ComputeBuffer xferColTBits => solver.collisionEvent.XferColTBitsBuffer;
        private ComputeBuffer xferColQAGi => solver.collisionEvent.XferColQAGiBuffer;
        private ComputeBuffer xferColQBGi => solver.collisionEvent.XferColQBGiBuffer;
        private ComputeBuffer xferColOAGi => solver.collisionEvent.XferColOAGiBuffer;
        private ComputeBuffer xferColOBGi => solver.collisionEvent.XferColOBGiBuffer;
        private ComputeBuffer defaultDtOwnerByLocal => solver.layerMappingCache.DefaultDtOwnerByLocal;
        private ComputeBuffer defaultDtCollisionOwnerByLocal => solver.layerMappingCache.DefaultDtCollisionOwnerByLocal;
        private int kClearCollisionEventCount => solver.collisionEvent.ClearCollisionEventCountKernel;
        private int kBuildCollisionEventsL0 => solver.collisionEvent.BuildCollisionEventsL0Kernel;
        private int kClearTransferredCollision => solver.collisionEvent.ClearTransferredCollisionKernel;
        private int kRestrictCollisionEventsToActivePairs => solver.collisionEvent.RestrictCollisionEventsToActivePairsKernel;
        private LayerSolveRuntime actualRuntime => solver.LayerSolveRuntime;

        private void BindDtGlobalMappingParams(CommandBuffer cb, ComputeShader targetShader, int kernel, bool useDtGlobalNodeMap, int dtLocalBase, ComputeBuffer dtGlobalNodeMap, ComputeBuffer dtGlobalToLayerLocalMap) {
            solver.layerMappingCache.BindDtGlobalMappingParams(cb, targetShader, kernel, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
        }

        internal void SetForceEventCountParam(CommandBuffer cb, int forceCount) {
            cb.SetComputeIntParam(shader, "_ForceEventCount", forceCount);
        }

        internal void BindRestrictedGameplayEventsBuffer(CommandBuffer cb) {
            cb.SetComputeBufferParam(shader, kRestrictGameplayDeltaVFromEvents, "_ForceEvents", solver.gameplayForce.ForceEventsBuffer);
        }

        internal void SetUseTransferredCollisionsParam(CommandBuffer cb, bool enabled) {
            actualRuntime.SetUseTransferredCollisionsParam(cb, enabled);
        }

        internal void SetRestrictedDeltaVScale(CommandBuffer cb, float scale) {
            cb.SetComputeFloatParam(shader, "_RestrictedDeltaVScale", scale);
        }

        public struct DtMappingContext {
            public readonly bool UseDtGlobalNodeMap;
            public readonly int DtLocalBase;
            public readonly ComputeBuffer DtGlobalNodeMap;
            public readonly ComputeBuffer DtGlobalToLayerLocalMap;
            public readonly ComputeBuffer DtOwnerByLocal;
            public readonly ComputeBuffer DtCollisionOwnerByLocal;

            public DtMappingContext(
                bool useDtGlobalNodeMap,
                int dtLocalBase,
                ComputeBuffer dtGlobalNodeMap,
                ComputeBuffer dtGlobalToLayerLocalMap,
                ComputeBuffer dtOwnerByLocal,
                ComputeBuffer dtCollisionOwnerByLocal
            ) {
                UseDtGlobalNodeMap = useDtGlobalNodeMap;
                DtLocalBase = dtLocalBase;
                DtGlobalNodeMap = dtGlobalNodeMap;
                DtGlobalToLayerLocalMap = dtGlobalToLayerLocalMap;
                DtOwnerByLocal = dtOwnerByLocal;
                DtCollisionOwnerByLocal = dtCollisionOwnerByLocal;
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
        internal int KClearRestrictedDeltaV => kClearRestrictedDeltaV;
        internal int KRestrictGameplayDeltaVFromEvents => kRestrictGameplayDeltaVFromEvents;
        internal int KRestrictFineVelocityResidualToActive => kRestrictFineVelocityResidualToActive;
        internal int KApplyRestrictedDeltaVToActiveAndPrefix => kApplyRestrictedDeltaVToActiveAndPrefix;
        internal int KRemoveRestrictedDeltaVFromActive => kRemoveRestrictedDeltaVFromActive;

        internal void CacheRuntimeKernels() {
            kClearHierarchicalStats = shader.FindKernel("ClearHierarchicalStats");
            kCacheHierarchicalStats = shader.FindKernel("CacheHierarchicalStats");
            kFinalizeHierarchicalStats = shader.FindKernel("FinalizeHierarchicalStats");
            kComputeCorrectionL = shader.FindKernel("ComputeCorrectionL");
            kCacheF0AndResetLambda = shader.FindKernel("CacheF0AndResetLambda");
            kSaveVelPrefix = shader.FindKernel("SaveVelPrefix");
            kClearVelDelta = shader.FindKernel("ClearVelDelta");
            kResetCollisionLambda = shader.FindKernel("ResetCollisionLambda");
            kClearRestrictedDeltaV = shader.FindKernel("ClearRestrictedDeltaV");
            kRestrictGameplayDeltaVFromEvents = shader.FindKernel("RestrictGameplayDeltaVFromEvents");
            kRestrictFineVelocityResidualToActive = shader.FindKernel("RestrictFineVelocityResidualToActive");
            kApplyRestrictedDeltaVToActiveAndPrefix = shader.FindKernel("ApplyRestrictedDeltaVToActiveAndPrefix");
            kRemoveRestrictedDeltaVFromActive = shader.FindKernel("RemoveRestrictedDeltaVFromActive");
        }

        internal void PrepareCacheAndRestrictionBuffers(CommandBuffer cb, in RelaxBufferContext context) {
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
            ComputeBuffer dtCollisionOwnerByLocal = context.Mapping.DtCollisionOwnerByLocal;

            cb.SetComputeIntParam(shader, "_Base", baseIndex);
            cb.SetComputeIntParam(shader, "_ActiveCount", activeCount);
            cb.SetComputeIntParam(shader, "_FineCount", fineCount);
            cb.SetComputeIntParam(shader, "_DtNeighborCount", neighborSearch.NeighborCount);
            cb.SetComputeFloatParam(shader, "_LayerKernelH", layerKernelH);
            cb.SetComputeIntParam(shader, "_UseDtOwnerFilter", dtOwnerByLocal != null ? 1 : 0);
            cb.SetComputeIntParam(shader, "_CollisionEventCapacity", solver.collisionEvent.CollisionEventsBuffer != null ? solver.collisionEvent.CollisionEventsBuffer.count : 0);

            cb.SetComputeBufferParam(shader, kClearHierarchicalStats, "_CurrentVolumeBits", actualRuntime.CurrentVolumeBits);
            cb.SetComputeBufferParam(shader, kClearHierarchicalStats, "_CurrentTotalMassBits", actualRuntime.CurrentTotalMassBits);
            cb.SetComputeBufferParam(shader, kClearHierarchicalStats, "_FixedChildPosBits", actualRuntime.FixedChildPosBits);
            cb.SetComputeBufferParam(shader, kClearHierarchicalStats, "_FixedChildCount", actualRuntime.FixedChildCount);
            cb.SetComputeBufferParam(shader, kClearHierarchicalStats, "_CoarseFixed", actualRuntime.CoarseFixed);
            cb.SetComputeBufferParam(shader, kClearHierarchicalStats, "_InvMass", invMass);

            cb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_InvMass", invMass);
            cb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_RestVolume", restVolume);
            cb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_Pos", pos);
            cb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_F", F);
            cb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_ParentIndex", parentIndex);
            cb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_ParentIndices", parentIndices);
            cb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_ParentWeights", parentWeights);
            cb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_CurrentVolumeBits", actualRuntime.CurrentVolumeBits);
            cb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_CurrentTotalMassBits", actualRuntime.CurrentTotalMassBits);
            cb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_FixedChildPosBits", actualRuntime.FixedChildPosBits);
            cb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_FixedChildCount", actualRuntime.FixedChildCount);
            cb.SetComputeBufferParam(shader, kCacheHierarchicalStats, "_CoarseFixed", actualRuntime.CoarseFixed);

            cb.SetComputeBufferParam(shader, kFinalizeHierarchicalStats, "_FixedChildCount", actualRuntime.FixedChildCount);
            cb.SetComputeBufferParam(shader, kFinalizeHierarchicalStats, "_CoarseFixed", actualRuntime.CoarseFixed);

            cb.SetComputeBufferParam(shader, kComputeCorrectionL, "_Pos", pos);
            cb.SetComputeBufferParam(shader, kComputeCorrectionL, "_CurrentVolumeBits", actualRuntime.CurrentVolumeBits);
            cb.SetComputeBufferParam(shader, kComputeCorrectionL, "_L", actualRuntime.CorrectionL);

            cb.SetComputeBufferParam(shader, kCacheF0AndResetLambda, "_F", F);
            cb.SetComputeBufferParam(shader, kCacheF0AndResetLambda, "_F0", actualRuntime.CachedF0);
            cb.SetComputeBufferParam(shader, kCacheF0AndResetLambda, "_Lambda", actualRuntime.Lambda);

            cb.SetComputeBufferParam(shader, kSaveVelPrefix, "_Vel", vel);
            cb.SetComputeBufferParam(shader, kSaveVelPrefix, "_SavedVelPrefix", actualRuntime.SavedVelPrefix);

            cb.SetComputeBufferParam(shader, kClearVelDelta, "_VelDeltaBits", actualRuntime.VelDeltaBits);

            cb.SetComputeBufferParam(shader, kResetCollisionLambda, "_DurabilityLambda", actualRuntime.DurabilityLambda);
            cb.SetComputeBufferParam(shader, kResetCollisionLambda, "_CollisionLambda", actualRuntime.CollisionLambda);

            cb.SetComputeBufferParam(collisionShader, kClearCollisionEventCount, "_CollisionEventCount", collisionEventCount);
            cb.SetComputeBufferParam(collisionShader, kBuildCollisionEventsL0, "_Pos", pos);
            cb.SetComputeBufferParam(collisionShader, kBuildCollisionEventsL0, "_DtNeighbors", neighborSearch.NeighborsBuffer);
            cb.SetComputeBufferParam(collisionShader, kBuildCollisionEventsL0, "_DtNeighborCounts", neighborSearch.NeighborCountsBuffer);
            cb.SetComputeBufferParam(collisionShader, kBuildCollisionEventsL0, "_DtOwnerByLocal", dtOwnerByLocal ?? defaultDtOwnerByLocal);
            cb.SetComputeBufferParam(collisionShader, kBuildCollisionEventsL0, "_DtCollisionOwnerByLocal", dtCollisionOwnerByLocal ?? dtOwnerByLocal ?? defaultDtCollisionOwnerByLocal);
            cb.SetComputeBufferParam(collisionShader, kBuildCollisionEventsL0, "_CollisionEvents", collisionEvents);
            cb.SetComputeBufferParam(collisionShader, kBuildCollisionEventsL0, "_CollisionEventCount", collisionEventCount);

            cb.SetComputeBufferParam(collisionShader, kClearTransferredCollision, "_XferColCount", xferColCount);
            cb.SetComputeBufferParam(collisionShader, kClearTransferredCollision, "_XferColNXBits", xferColNXBits);
            cb.SetComputeBufferParam(collisionShader, kClearTransferredCollision, "_XferColNYBits", xferColNYBits);
            cb.SetComputeBufferParam(collisionShader, kClearTransferredCollision, "_XferColPenBits", xferColPenBits);
            cb.SetComputeBufferParam(collisionShader, kClearTransferredCollision, "_XferColSBits", xferColSBits);
            cb.SetComputeBufferParam(collisionShader, kClearTransferredCollision, "_XferColTBits", xferColTBits);
            cb.SetComputeBufferParam(collisionShader, kClearTransferredCollision, "_XferColQAGi", xferColQAGi);
            cb.SetComputeBufferParam(collisionShader, kClearTransferredCollision, "_XferColQBGi", xferColQBGi);
            cb.SetComputeBufferParam(collisionShader, kClearTransferredCollision, "_XferColOAGi", xferColOAGi);
            cb.SetComputeBufferParam(collisionShader, kClearTransferredCollision, "_XferColOBGi", xferColOBGi);

            cb.SetComputeBufferParam(collisionShader, kRestrictCollisionEventsToActivePairs, "_ParentIndex", parentIndex);
            cb.SetComputeBufferParam(collisionShader, kRestrictCollisionEventsToActivePairs, "_ParentIndices", parentIndices);
            cb.SetComputeBufferParam(collisionShader, kRestrictCollisionEventsToActivePairs, "_ParentWeights", parentWeights);
            cb.SetComputeBufferParam(collisionShader, kRestrictCollisionEventsToActivePairs, "_DtNeighbors", neighborSearch.NeighborsBuffer);
            cb.SetComputeBufferParam(collisionShader, kRestrictCollisionEventsToActivePairs, "_DtNeighborCounts", neighborSearch.NeighborCountsBuffer);
            cb.SetComputeBufferParam(collisionShader, kRestrictCollisionEventsToActivePairs, "_CollisionEvents", collisionEvents);
            cb.SetComputeBufferParam(collisionShader, kRestrictCollisionEventsToActivePairs, "_CollisionEventCount", collisionEventCount);
            cb.SetComputeBufferParam(collisionShader, kRestrictCollisionEventsToActivePairs, "_XferColCount", xferColCount);
            cb.SetComputeBufferParam(collisionShader, kRestrictCollisionEventsToActivePairs, "_XferColNXBits", xferColNXBits);
            cb.SetComputeBufferParam(collisionShader, kRestrictCollisionEventsToActivePairs, "_XferColNYBits", xferColNYBits);
            cb.SetComputeBufferParam(collisionShader, kRestrictCollisionEventsToActivePairs, "_XferColPenBits", xferColPenBits);
            cb.SetComputeBufferParam(collisionShader, kRestrictCollisionEventsToActivePairs, "_XferColSBits", xferColSBits);
            cb.SetComputeBufferParam(collisionShader, kRestrictCollisionEventsToActivePairs, "_XferColTBits", xferColTBits);
            cb.SetComputeBufferParam(collisionShader, kRestrictCollisionEventsToActivePairs, "_XferColQAGi", xferColQAGi);
            cb.SetComputeBufferParam(collisionShader, kRestrictCollisionEventsToActivePairs, "_XferColQBGi", xferColQBGi);
            cb.SetComputeBufferParam(collisionShader, kRestrictCollisionEventsToActivePairs, "_XferColOAGi", xferColOAGi);
            cb.SetComputeBufferParam(collisionShader, kRestrictCollisionEventsToActivePairs, "_XferColOBGi", xferColOBGi);

            cb.SetComputeBufferParam(shader, kClearRestrictedDeltaV, "_RestrictedDeltaVBits", actualRuntime.RestrictedDeltaVBits);
            cb.SetComputeBufferParam(shader, kClearRestrictedDeltaV, "_RestrictedDeltaVCount", actualRuntime.RestrictedDeltaVCount);
            cb.SetComputeBufferParam(shader, kClearRestrictedDeltaV, "_RestrictedDeltaVAvg", actualRuntime.RestrictedDeltaVAvg);

            cb.SetComputeBufferParam(shader, kRestrictGameplayDeltaVFromEvents, "_RestrictedDeltaVBits", actualRuntime.RestrictedDeltaVBits);
            cb.SetComputeBufferParam(shader, kRestrictGameplayDeltaVFromEvents, "_RestrictedDeltaVCount", actualRuntime.RestrictedDeltaVCount);
            cb.SetComputeBufferParam(shader, kRestrictGameplayDeltaVFromEvents, "_InvMass", invMass);
            cb.SetComputeBufferParam(shader, kRestrictGameplayDeltaVFromEvents, "_ParentIndex", parentIndex);
            cb.SetComputeBufferParam(shader, kRestrictGameplayDeltaVFromEvents, "_ParentIndices", parentIndices);
            cb.SetComputeBufferParam(shader, kRestrictGameplayDeltaVFromEvents, "_ParentWeights", parentWeights);
            cb.SetComputeBufferParam(shader, kRestrictGameplayDeltaVFromEvents, "_ForceEvents", solver.gameplayForce.ForceEventsBuffer);

            cb.SetComputeBufferParam(shader, kRestrictFineVelocityResidualToActive, "_Vel", vel);
            cb.SetComputeBufferParam(shader, kRestrictFineVelocityResidualToActive, "_InvMass", invMass);
            cb.SetComputeBufferParam(shader, kRestrictFineVelocityResidualToActive, "_ParentIndex", parentIndex);
            cb.SetComputeBufferParam(shader, kRestrictFineVelocityResidualToActive, "_ParentIndices", parentIndices);
            cb.SetComputeBufferParam(shader, kRestrictFineVelocityResidualToActive, "_ParentWeights", parentWeights);
            cb.SetComputeBufferParam(shader, kRestrictFineVelocityResidualToActive, "_RestrictedDeltaVBits", actualRuntime.RestrictedDeltaVBits);
            cb.SetComputeBufferParam(shader, kRestrictFineVelocityResidualToActive, "_RestrictedDeltaVCount", actualRuntime.RestrictedDeltaVCount);

            cb.SetComputeBufferParam(shader, kApplyRestrictedDeltaVToActiveAndPrefix, "_RestrictedDeltaVBits", actualRuntime.RestrictedDeltaVBits);
            cb.SetComputeBufferParam(shader, kApplyRestrictedDeltaVToActiveAndPrefix, "_RestrictedDeltaVCount", actualRuntime.RestrictedDeltaVCount);
            cb.SetComputeBufferParam(shader, kApplyRestrictedDeltaVToActiveAndPrefix, "_RestrictedDeltaVAvg", actualRuntime.RestrictedDeltaVAvg);
            cb.SetComputeBufferParam(shader, kApplyRestrictedDeltaVToActiveAndPrefix, "_CoarseFixed", actualRuntime.CoarseFixed);
            cb.SetComputeBufferParam(shader, kApplyRestrictedDeltaVToActiveAndPrefix, "_Vel", vel);
            cb.SetComputeBufferParam(shader, kApplyRestrictedDeltaVToActiveAndPrefix, "_SavedVelPrefix", actualRuntime.SavedVelPrefix);

            cb.SetComputeBufferParam(shader, kRemoveRestrictedDeltaVFromActive, "_RestrictedDeltaVAvg", actualRuntime.RestrictedDeltaVAvg);
            cb.SetComputeBufferParam(shader, kRemoveRestrictedDeltaVFromActive, "_Vel", vel);

            BindDtGlobalMappingParams(cb, shader, kClearHierarchicalStats, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(cb, shader, kCacheHierarchicalStats, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(cb, shader, kFinalizeHierarchicalStats, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(cb, shader, kComputeCorrectionL, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(cb, shader, kCacheF0AndResetLambda, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(cb, shader, kSaveVelPrefix, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(cb, shader, kClearVelDelta, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(cb, shader, kResetCollisionLambda, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(cb, shader, kClearRestrictedDeltaV, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(cb, shader, kRestrictGameplayDeltaVFromEvents, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(cb, shader, kRestrictFineVelocityResidualToActive, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(cb, shader, kApplyRestrictedDeltaVToActiveAndPrefix, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(cb, shader, kRemoveRestrictedDeltaVFromActive, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(cb, collisionShader, kBuildCollisionEventsL0, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(cb, collisionShader, kClearTransferredCollision, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);
            BindDtGlobalMappingParams(cb, collisionShader, kRestrictCollisionEventsToActivePairs, useDtGlobalNodeMap, dtLocalBase, dtGlobalNodeMap, dtGlobalToLayerLocalMap);

            cb.SetComputeBufferParam(shader, kComputeCorrectionL, "_DtNeighbors", neighborSearch.NeighborsBuffer);
            cb.SetComputeBufferParam(shader, kComputeCorrectionL, "_DtNeighborCounts", neighborSearch.NeighborCountsBuffer);
            cb.SetComputeBufferParam(shader, kComputeCorrectionL, "_DtOwnerByLocal", dtOwnerByLocal ?? defaultDtOwnerByLocal);
        }
    }
}
