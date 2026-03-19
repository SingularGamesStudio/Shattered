using GPU.Delaunay;
using UnityEngine;
using LayerContext = GPU.Solver.XPBISolver.LayerContext;
using ProlongationConstraintProbe = GPU.Solver.XPBISolver.ProlongationConstraintProbe;
using SolveSession = GPU.Solver.XPBISolver.SolveSession;
using TickContext = GPU.Solver.XPBISolver.TickContext;

namespace GPU.Solver {
    internal sealed partial class LayerSolve {
        private readonly XPBISolver solver;

        public LayerSolve(XPBISolver solver) {
            this.solver = solver;
        }

        /// <summary>
        /// Records all kernels for one layer solve pass.
        /// </summary>
        public void Record(SolveSession session, TickContext tickContext, LayerContext layerContext) {
            int prolongationPreProbeEntry = -1;
            bool hasProlongationTargetProbe = false;
            LayerContext targetLayerContext = null;

            if (session.EnableProlongationConstraintProbeDebug &&
                layerContext.Layer > 0 &&
                session.ProlongationConstraintProbes != null &&
                solver.solverDebug.ProlongationConstraintDebug != null &&
                session.ProlongationProbeCursor + 1 < solver.solverDebug.ProlongationConstraintDebugEntries) {
                if (solver.layerMappingCache.TryBuildLayerContext(session, layerContext.Layer - 1, out targetLayerContext)) {
                    prolongationPreProbeEntry = session.ProlongationProbeCursor++;
                    PrepareRelaxBuffers(new RelaxBufferContext(
                        targetLayerContext.NeighborSearch,
                        0,
                        targetLayerContext.ActiveCount,
                        targetLayerContext.FineCount,
                        tickContext.TickIndex,
                        targetLayerContext.KernelH,
                        new DtMappingContext(
                            targetLayerContext.UseMappedIndices,
                            0,
                            targetLayerContext.GlobalNodeMap,
                            targetLayerContext.GlobalToLocalMap,
                            targetLayerContext.OwnerByLocalBuffer)));
                    solver.solverDebug.CaptureCurrentConstraintErrorSample(
                        prolongationPreProbeEntry,
                        targetLayerContext.ActiveCount,
                        targetLayerContext.ActiveCount,
                        targetLayerContext.OwnerByLocalBuffer != null);
                    hasProlongationTargetProbe = true;
                }
            }

            RecordLayerSolvePass(session, tickContext, layerContext);

            if (hasProlongationTargetProbe && targetLayerContext != null) {
                int prolongationPostProbeEntry = session.ProlongationProbeCursor++;
                PrepareRelaxBuffers(new RelaxBufferContext(
                    targetLayerContext.NeighborSearch,
                    0,
                    targetLayerContext.ActiveCount,
                    targetLayerContext.FineCount,
                    tickContext.TickIndex,
                    targetLayerContext.KernelH,
                    new DtMappingContext(
                        targetLayerContext.UseMappedIndices,
                        0,
                        targetLayerContext.GlobalNodeMap,
                        targetLayerContext.GlobalToLocalMap,
                        targetLayerContext.OwnerByLocalBuffer)));
                solver.solverDebug.CaptureCurrentConstraintErrorSample(
                    prolongationPostProbeEntry,
                    targetLayerContext.ActiveCount,
                    targetLayerContext.ActiveCount,
                    targetLayerContext.OwnerByLocalBuffer != null);
                session.ProlongationConstraintProbes.Add(
                    new ProlongationConstraintProbe(
                        tickContext.TickIndex,
                        layerContext.Layer,
                        prolongationPreProbeEntry,
                        prolongationPostProbeEntry));
            }
        }

        /// <summary>
        /// Records all kernel dispatches for one layer solve pass.
        /// </summary>
        private void RecordLayerSolvePass(SolveSession session, TickContext tickContext, LayerContext layerContext) {
            int layer = layerContext.Layer;
            int activeCount = layerContext.ActiveCount;
            int fineCount = layerContext.FineCount;

            PrepareRelaxBuffers(new RelaxBufferContext(
                layerContext.NeighborSearch,
                0,
                activeCount,
                fineCount,
                tickContext.TickIndex,
                layerContext.KernelH,
                new DtMappingContext(
                    layerContext.UseMappedIndices,
                    0,
                    layerContext.GlobalNodeMap,
                    layerContext.GlobalToLocalMap,
                    layerContext.OwnerByLocalBuffer)));

            solver.Dispatch("XPBI.ClearHierarchicalStats", solver.shader, kClearHierarchicalStats, XPBISolver.Groups256(activeCount), 1, 1);
            solver.Dispatch("XPBI.CacheHierarchicalStats", solver.shader, kCacheHierarchicalStats, XPBISolver.Groups256(fineCount), 1, 1);
            solver.Dispatch("XPBI.FinalizeHierarchicalStats", solver.shader, kFinalizeHierarchicalStats, XPBISolver.Groups256(activeCount), 1, 1);
            solver.Dispatch("XPBI.SaveVelPrefix", solver.shader, kSaveVelPrefix, XPBISolver.Groups256(activeCount), 1, 1);

            bool useHierarchyTransfer = layer > 0 && fineCount > activeCount;
            bool injectRestrictedGameplay =
                useHierarchyTransfer &&
                tickContext.ForceCount > 0 &&
                solver.gameplayForce.HasEventsBuffer &&
                Const.RestrictedDeltaVScale > 0f;
            bool injectRestrictedResidual =
                useHierarchyTransfer &&
                Const.RestrictResidualDeltaVScale > 0f;

            if (injectRestrictedResidual) {
                solver.Dispatch("XPBI.ClearRestrictedDeltaV", solver.shader, kClearRestrictedDeltaV, XPBISolver.Groups256(activeCount), 1, 1);
                solver.Dispatch("XPBI.RestrictFineVelocityResidualToActive", solver.shader, kRestrictFineVelocityResidualToActive, XPBISolver.Groups256(fineCount - activeCount), 1, 1);
                SetRestrictedDeltaVScale(Const.RestrictResidualDeltaVScale);
                solver.Dispatch("XPBI.ApplyRestrictedDeltaVToActiveAndPrefix", solver.shader, kApplyRestrictedDeltaVToActiveAndPrefix, XPBISolver.Groups256(activeCount), 1, 1);
            }

            if (injectRestrictedGameplay) {
                solver.Dispatch("XPBI.ClearRestrictedDeltaV", solver.shader, kClearRestrictedDeltaV, XPBISolver.Groups256(activeCount), 1, 1);
                SetForceEventCountParam(tickContext.ForceCount);
                BindRestrictedGameplayEventsBuffer();
                solver.Dispatch("XPBI.RestrictGameplayDeltaVFromEvents", solver.shader, kRestrictGameplayDeltaVFromEvents, XPBISolver.Groups256(tickContext.ForceCount), 1, 1);
                SetRestrictedDeltaVScale(Const.RestrictedDeltaVScale);
                solver.Dispatch("XPBI.ApplyRestrictedDeltaVToActiveAndPrefix", solver.shader, kApplyRestrictedDeltaVToActiveAndPrefix, XPBISolver.Groups256(activeCount), 1, 1);
            }

            if (!injectRestrictedGameplay)
                SetForceEventCountParam(0);

            solver.Dispatch("XPBI.ComputeCorrectionL", solver.shader, kComputeCorrectionL, XPBISolver.Groups256(activeCount), 1, 1);
            solver.Dispatch("XPBI.CacheF0AndResetLambda", solver.shader, kCacheF0AndResetLambda, XPBISolver.Groups256(activeCount), 1, 1);
            solver.Dispatch("XPBI.ResetCollisionLambda", solver.shader, kResetCollisionLambda, XPBISolver.Groups256(activeCount), 1, 1);

            bool useTransferredCollisions = layer > 0;
            SetUseTransferredCollisionsParam(useTransferredCollisions);
            if (useTransferredCollisions) {
                solver.Dispatch("XPBI.ClearTransferredCollision", solver.shader, solver.collisionEvent.ClearTransferredCollisionKernel, XPBISolver.Groups256(activeCount), 1, 1);
                solver.Dispatch("XPBI.RestrictCollisionEventsToActivePairs", solver.shader, solver.collisionEvent.RestrictCollisionEventsToActivePairsKernel, XPBISolver.Groups256(solver.collisionEvent.CollisionEventsBuffer != null ? solver.collisionEvent.CollisionEventsBuffer.count : 0), 1, 1);
            }

            int jrIterations = GetJRIterationsForLayer(layer, session.MaxSolveLayer);
            int gsIterations = layer == 0 ? Const.GSIterationsL0 : 1;

            DTColoring layerColoring = null;
            if (gsIterations > 0) {
                layerColoring = solver.coloring.RebuildForLayer(layerContext, session.FixedObjectSignature);
                if (layerColoring != null) {
                    if (session.ColoringUpdatedByLayer != null && layer >= 0 && layer < session.ColoringUpdatedByLayer.Length)
                        session.ColoringUpdatedByLayer[layer] = true;
                    BindLayerColoringBuffers(layerColoring);
                }
            }

            int debugIterations = gsIterations + jrIterations;
            bool debugEnabled = solver.solverDebug.ConvergenceDebug != null;
            bool usePersistentCoarseGs =
                gsIterations > 0 &&
                layerColoring != null &&
                activeCount <= Const.PersistentCoarseMaxNodes &&
                !debugEnabled;

            if (debugEnabled && debugIterations > 0) {
                solver.solverDebug.ClearDebugBuffer(layer, debugIterations, tickContext.TickIndex);
            } else {
                SetConvergenceDebugDisabled();
            }

            if (gsIterations > 0 && layerColoring != null) {
                SetCollisionEnable(true);

                if (usePersistentCoarseGs) {
                    SetPersistentRelaxParams(gsIterations, 0);
                    solver.Dispatch("XPBI.RelaxColoredPersistentCoarse", solver.shader, kRelaxColoredPersistentCoarse, 1, 1, 1);
                } else {
                    for (int iter = 0; iter < gsIterations; iter++) {
                        if (debugEnabled)
                            SetConvergenceDebugIter(iter);

                        for (int color = 0; color < 16; color++) {
                            SetColorIndex(color);
                            if (layerColoring.RelaxArgsBuffer != null)
                                solver.DispatchIndirect(
                                    solver.GetRelaxDispatchMarker(layer, color),
                                    solver.shader,
                                    kRelaxColored,
                                    layerColoring.RelaxArgsBuffer,
                                    (uint)color * 12);
                        }
                    }
                }
            }

            if (jrIterations > 0) {
                SetCollisionEnable(false);
                SetJRParams();

                for (int iter = 0; iter < jrIterations; iter++) {
                    if (debugEnabled)
                        SetConvergenceDebugIter(gsIterations + iter);

                    solver.Dispatch("XPBI.JR.SavePrevAndClear", solver.shader, kJRSavePrevAndClear, XPBISolver.Groups256(activeCount), 1, 1);
                    solver.Dispatch("XPBI.JR.ComputeDeltas", solver.shader, kJRComputeDeltas, XPBISolver.Groups256(activeCount), 1, 1);
                    solver.Dispatch("XPBI.JR.Apply", solver.shader, kJRApply, XPBISolver.Groups256(activeCount), 1, 1);
                }
            }

            if (layer > 0 && fineCount > activeCount) {
                solver.Dispatch("XPBI.Prolongate", solver.shader, kProlongate, XPBISolver.Groups256(fineCount), 1, 1);
                if (Const.PostProlongSmoothing > 0f)
                    solver.Dispatch("XPBI.SmoothProlongatedFineVel", solver.shader, kSmoothProlongatedFineVel, XPBISolver.Groups256(fineCount), 1, 1);
            }

            if (injectRestrictedGameplay || injectRestrictedResidual)
                solver.Dispatch("XPBI.RemoveRestrictedDeltaVFromActive", solver.shader, kRemoveRestrictedDeltaVFromActive, XPBISolver.Groups256(activeCount), 1, 1);

            if (layer == 0)
                solver.Dispatch("XPBI.CommitDeformation", solver.shader, kCommitDeformation, XPBISolver.Groups256(activeCount), 1, 1);
        }

        /// <summary>
        /// Returns JR iteration count for the layer according to configured top/mid/layer0 policy.
        /// </summary>
        private static int GetJRIterationsForLayer(int layer, int maxSolveLayer) {
            int iterations = Const.JRIterationsLMid;
            if (layer == maxSolveLayer)
                iterations = Const.JRIterationsLMax;
            if (layer == 0)
                iterations = Const.JRIterationsL0;

            return Mathf.Max(1, iterations);
        }
    }
}
