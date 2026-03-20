using GPU.Delaunay;
using GPU.Neighbors;
using LayerContext = GPU.Solver.XPBISolver.LayerContext;
using ProlongationConstraintProbe = GPU.Solver.XPBISolver.ProlongationConstraintProbe;
using SolveSession = GPU.Solver.XPBISolver.SolveSession;
using TickContext = GPU.Solver.XPBISolver.TickContext;

namespace GPU.Solver {
    internal sealed class LayerSolvePass {
        private readonly LayerSolveRuntime runtime;
        private readonly UnityEngine.ComputeShader shader;
        private XPBISolver solver => runtime.Solver;

        public LayerSolvePass(LayerSolveRuntime runtime) {
            this.runtime = runtime;
            shader = runtime.Solver.LayerSolveShader;
        }

        private static void Dispatch(UnityEngine.Rendering.CommandBuffer cb, string marker, UnityEngine.ComputeShader dispatchShader, int kernel, int groupsX, int groupsY, int groupsZ) {
            cb.BeginSample(marker);
            cb.DispatchCompute(dispatchShader, kernel, groupsX, groupsY, groupsZ);
            cb.EndSample(marker);
        }

        private static void DispatchIndirect(UnityEngine.Rendering.CommandBuffer cb, string marker, UnityEngine.ComputeShader dispatchShader, int kernel, UnityEngine.ComputeBuffer args, uint argsOffset) {
            cb.BeginSample(marker);
            cb.DispatchCompute(dispatchShader, kernel, args, argsOffset);
            cb.EndSample(marker);
        }

        internal void RecordSolve(SolveSession session, TickContext tickContext, LayerContext layerContext) {
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
                    runtime.PrepareSolveBuffers(session.AsyncCb, new LayerSolveRuntime.RelaxBufferContext(
                        targetLayerContext.NeighborSearch,
                        0,
                        targetLayerContext.ActiveCount,
                        targetLayerContext.FineCount,
                        tickContext.TickIndex,
                        targetLayerContext.KernelH,
                        new LayerSolveRuntime.DtMappingContext(
                            targetLayerContext.UseMappedIndices,
                            0,
                            targetLayerContext.GlobalNodeMap,
                            targetLayerContext.GlobalToLocalMap,
                            targetLayerContext.OwnerByLocalBuffer)));
                    solver.solverDebug.CaptureCurrentConstraintErrorSample(
                        session.AsyncCb,
                        prolongationPreProbeEntry,
                        targetLayerContext.ActiveCount,
                        targetLayerContext.ActiveCount,
                        targetLayerContext.OwnerByLocalBuffer != null);
                    hasProlongationTargetProbe = true;
                }
            }

            RecordSolvePass(session, tickContext, layerContext);

            if (hasProlongationTargetProbe && targetLayerContext != null) {
                int prolongationPostProbeEntry = session.ProlongationProbeCursor++;
                runtime.PrepareSolveBuffers(session.AsyncCb, new LayerSolveRuntime.RelaxBufferContext(
                    targetLayerContext.NeighborSearch,
                    0,
                    targetLayerContext.ActiveCount,
                    targetLayerContext.FineCount,
                    tickContext.TickIndex,
                    targetLayerContext.KernelH,
                    new LayerSolveRuntime.DtMappingContext(
                        targetLayerContext.UseMappedIndices,
                        0,
                        targetLayerContext.GlobalNodeMap,
                        targetLayerContext.GlobalToLocalMap,
                        targetLayerContext.OwnerByLocalBuffer)));
                solver.solverDebug.CaptureCurrentConstraintErrorSample(
                    session.AsyncCb,
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

        private void RecordSolvePass(SolveSession session, TickContext tickContext, LayerContext layerContext) {
            int layer = layerContext.Layer;
            int activeCount = layerContext.ActiveCount;
            int fineCount = layerContext.FineCount;

            runtime.PrepareSolveBuffers(session.AsyncCb, new LayerSolveRuntime.RelaxBufferContext(
                layerContext.NeighborSearch,
                0,
                activeCount,
                fineCount,
                tickContext.TickIndex,
                layerContext.KernelH,
                new LayerSolveRuntime.DtMappingContext(
                    layerContext.UseMappedIndices,
                    0,
                    layerContext.GlobalNodeMap,
                    layerContext.GlobalToLocalMap,
                    layerContext.OwnerByLocalBuffer)));

            int jrIterations = GetJRIterationsForLayer(layer, session.MaxSolveLayer);
            int gsIterations = layer == 0 ? Const.GSIterationsL0 : 1;

            NeighborColoring layerColoring = null;
            if (gsIterations > 0) {
                layerColoring = solver.coloring.RebuildForLayer(session.AsyncCb, session, layerContext, session.FixedObjectSignature);
                if (layerColoring != null) {
                    if (session.ColoringUpdatedByLayer != null && layer >= 0 && layer < session.ColoringUpdatedByLayer.Length)
                        session.ColoringUpdatedByLayer[layer] = true;
                    runtime.BindLayerColoringBuffers(session.AsyncCb, layerColoring);
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
                solver.solverDebug.ClearDebugBuffer(session.AsyncCb, layer, debugIterations, tickContext.TickIndex);
            } else {
                runtime.SetConvergenceDebugDisabled(session.AsyncCb);
            }

            if (gsIterations > 0 && layerColoring != null) {
                runtime.SetCollisionEnable(session.AsyncCb, true);

                if (usePersistentCoarseGs) {
                    runtime.SetPersistentRelaxParams(session.AsyncCb, gsIterations, 0);
                    Dispatch(session.AsyncCb, "XPBI.RelaxColoredPersistentCoarse", shader, runtime.KRelaxColoredPersistentCoarse, 1, 1, 1);
                } else {
                    for (int iter = 0; iter < gsIterations; iter++) {
                        if (debugEnabled)
                            runtime.SetConvergenceDebugIter(session.AsyncCb, iter);

                        for (int color = 0; color < 16; color++) {
                            runtime.SetColorIndex(session.AsyncCb, color);
                            if (layerColoring.RelaxArgsBuffer != null)
                                DispatchIndirect(
                                    session.AsyncCb,
                                    solver.GetRelaxDispatchMarker(layer, color),
                                    shader,
                                    runtime.KRelaxColored,
                                    layerColoring.RelaxArgsBuffer,
                                    (uint)color * 12);
                        }
                    }
                }
            }

            if (jrIterations > 0) {
                runtime.SetCollisionEnable(session.AsyncCb, false);
                runtime.SetJRParams(session.AsyncCb);

                for (int iter = 0; iter < jrIterations; iter++) {
                    if (debugEnabled)
                        runtime.SetConvergenceDebugIter(session.AsyncCb, gsIterations + iter);

                    Dispatch(session.AsyncCb, "XPBI.JR.SavePrevAndClear", shader, runtime.KJRSavePrevAndClear, XPBISolver.Groups256(activeCount), 1, 1);
                    Dispatch(session.AsyncCb, "XPBI.JR.ComputeDeltas", shader, runtime.KJRComputeDeltas, XPBISolver.Groups256(activeCount), 1, 1);
                    Dispatch(session.AsyncCb, "XPBI.JR.Apply", shader, runtime.KJRApply, XPBISolver.Groups256(activeCount), 1, 1);
                }
            }

            if (layer > 0 && fineCount > activeCount) {
                Dispatch(session.AsyncCb, "XPBI.Prolongate", shader, runtime.KProlongate, XPBISolver.Groups256(fineCount), 1, 1);
                if (Const.PostProlongSmoothing > 0f)
                    Dispatch(session.AsyncCb, "XPBI.SmoothProlongatedFineVel", shader, runtime.KSmoothProlongatedFineVel, XPBISolver.Groups256(fineCount), 1, 1);
            }

            if (layer == 0)
                Dispatch(session.AsyncCb, "XPBI.CommitDeformation", shader, runtime.KCommitDeformation, XPBISolver.Groups256(activeCount), 1, 1);
        }

        private static int GetJRIterationsForLayer(int layer, int maxSolveLayer) {
            if (maxSolveLayer <= 0)
                return 0;

            if (Const.JRIterationsL0 <= 0 && Const.JRIterationsLMid <= 0 && Const.JRIterationsLMax <= 0)
                return 0;

            if (layer == 0)
                return Const.JRIterationsL0;

            if (layer == maxSolveLayer)
                return Const.JRIterationsLMax;

            return Const.JRIterationsLMid;
        }
    }
}
