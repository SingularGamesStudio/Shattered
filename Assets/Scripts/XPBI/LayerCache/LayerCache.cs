using UnityEngine;
using UnityEngine.Rendering;
using LayerContext = GPU.Solver.XPBISolver.LayerContext;
using SolveSession = GPU.Solver.XPBISolver.SolveSession;
using TickContext = GPU.Solver.XPBISolver.TickContext;

namespace GPU.Solver {
    internal sealed class LayerCachePass {
        private readonly LayerCacheRuntime runtime;
        private readonly ComputeShader layerCacheShader;

        public LayerCachePass(LayerCacheRuntime runtime) {
            this.runtime = runtime;
            layerCacheShader = runtime.Solver.LayerCacheShader;
        }

        private static void Dispatch(CommandBuffer cb, string marker, ComputeShader shader, int kernel, int groupsX, int groupsY, int groupsZ) {
            cb.BeginSample(marker);
            cb.DispatchCompute(shader, kernel, groupsX, groupsY, groupsZ);
            cb.EndSample(marker);
        }

        internal void RecordCache(
            SolveSession session,
            TickContext tickContext,
            LayerContext layerContext,
            out bool injectRestrictedGameplay,
            out bool injectRestrictedResidual
        ) {
            int layer = layerContext.Layer;
            int activeCount = layerContext.ActiveCount;
            int fineCount = layerContext.FineCount;

            runtime.PrepareCacheAndRestrictionBuffers(session.AsyncCb, new LayerCacheRuntime.RelaxBufferContext(
                layerContext.NeighborSearch,
                0,
                activeCount,
                fineCount,
                tickContext.TickIndex,
                layerContext.KernelH,
                new LayerCacheRuntime.DtMappingContext(
                    layerContext.UseMappedIndices,
                    0,
                    layerContext.GlobalNodeMap,
                    layerContext.GlobalToLocalMap,
                    layerContext.OwnerByLocalBuffer,
                    layerContext.CollisionOwnerByLocalBuffer)));

            Dispatch(session.AsyncCb, "XPBI.ClearHierarchicalStats", layerCacheShader, runtime.KClearHierarchicalStats, XPBISolver.Groups256(activeCount), 1, 1);
            Dispatch(session.AsyncCb, "XPBI.CacheHierarchicalStats", layerCacheShader, runtime.KCacheHierarchicalStats, XPBISolver.Groups256(fineCount), 1, 1);
            Dispatch(session.AsyncCb, "XPBI.FinalizeHierarchicalStats", layerCacheShader, runtime.KFinalizeHierarchicalStats, XPBISolver.Groups256(activeCount), 1, 1);
            Dispatch(session.AsyncCb, "XPBI.SaveVelPrefix", layerCacheShader, runtime.KSaveVelPrefix, XPBISolver.Groups256(activeCount), 1, 1);

            bool useHierarchyTransfer = layer > 0 && fineCount > activeCount;
            injectRestrictedGameplay =
                useHierarchyTransfer &&
                tickContext.ForceCount > 0 &&
                tickContext.HasForceEventsBuffer &&
                Const.RestrictedDeltaVScale > 0f;
            injectRestrictedResidual =
                useHierarchyTransfer &&
                Const.RestrictResidualDeltaVScale > 0f;

            if (injectRestrictedResidual) {
                Dispatch(session.AsyncCb, "XPBI.ClearRestrictedDeltaV", layerCacheShader, runtime.KClearRestrictedDeltaV, XPBISolver.Groups256(activeCount), 1, 1);
                Dispatch(session.AsyncCb, "XPBI.RestrictFineVelocityResidualToActive", layerCacheShader, runtime.KRestrictFineVelocityResidualToActive, XPBISolver.Groups256(fineCount - activeCount), 1, 1);
                runtime.SetRestrictedDeltaVScale(session.AsyncCb, Const.RestrictResidualDeltaVScale);
                Dispatch(session.AsyncCb, "XPBI.ApplyRestrictedDeltaVToActiveAndPrefix", layerCacheShader, runtime.KApplyRestrictedDeltaVToActiveAndPrefix, XPBISolver.Groups256(activeCount), 1, 1);
            }

            if (injectRestrictedGameplay) {
                Dispatch(session.AsyncCb, "XPBI.ClearRestrictedDeltaV", layerCacheShader, runtime.KClearRestrictedDeltaV, XPBISolver.Groups256(activeCount), 1, 1);
                runtime.SetForceEventCountParam(session.AsyncCb, tickContext.ForceCount);
                runtime.BindRestrictedGameplayEventsBuffer(session.AsyncCb);
                Dispatch(session.AsyncCb, "XPBI.RestrictGameplayDeltaVFromEvents", layerCacheShader, runtime.KRestrictGameplayDeltaVFromEvents, XPBISolver.Groups256(tickContext.ForceCount), 1, 1);
                runtime.SetRestrictedDeltaVScale(session.AsyncCb, Const.RestrictedDeltaVScale);
                Dispatch(session.AsyncCb, "XPBI.ApplyRestrictedDeltaVToActiveAndPrefix", layerCacheShader, runtime.KApplyRestrictedDeltaVToActiveAndPrefix, XPBISolver.Groups256(activeCount), 1, 1);
            }

            if (!injectRestrictedGameplay)
                runtime.SetForceEventCountParam(session.AsyncCb, 0);

            Dispatch(session.AsyncCb, "XPBI.ComputeCorrectionL", layerCacheShader, runtime.KComputeCorrectionL, XPBISolver.Groups256(activeCount), 1, 1);
            Dispatch(session.AsyncCb, "XPBI.CacheF0AndResetLambda", layerCacheShader, runtime.KCacheF0AndResetLambda, XPBISolver.Groups256(activeCount), 1, 1);
            Dispatch(session.AsyncCb, "XPBI.ResetCollisionLambda", layerCacheShader, runtime.KResetCollisionLambda, XPBISolver.Groups256(activeCount), 1, 1);
        }

        internal void RecordRestrictionCleanup(CommandBuffer cb, int activeCount, bool injectRestrictedGameplay, bool injectRestrictedResidual) {
            if (injectRestrictedGameplay || injectRestrictedResidual)
                Dispatch(cb, "XPBI.RemoveRestrictedDeltaVFromActive", layerCacheShader, runtime.KRemoveRestrictedDeltaVFromActive, XPBISolver.Groups256(activeCount), 1, 1);
        }
    }
}
