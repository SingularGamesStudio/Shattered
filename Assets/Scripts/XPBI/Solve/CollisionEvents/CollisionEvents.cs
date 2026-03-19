using GPU.Neighbors;
using UnityEngine;
using LayerContext = GPU.Solver.XPBISolver.LayerContext;
using SolveSession = GPU.Solver.XPBISolver.SolveSession;
using TickContext = GPU.Solver.XPBISolver.TickContext;

namespace GPU.Solver {
    internal sealed partial class CollisionEvents {
        private const int UseTransferredCollisionsDisabled = 0;

        private readonly XPBISolver solver;

        public CollisionEvents(XPBISolver solver) {
            this.solver = solver;
        }

        /// <summary>
        /// Records optional override-neighbor build for layer 0 and collision-event generation.
        /// </summary>
        public void RecordLayer0Build(SolveSession session, TickContext tickContext) {
            if (session.UseOverrideLayer0NeighborSearch) {
                if (session.Request.GlobalDTHierarchy.TryGetLayerExecutionContext(0, out int activeCount, out _, out float layerKernelH)) {
                    float supportRadius = Mathf.Max(1e-5f, Const.WendlandSupport * layerKernelH);
                    float cellSize = supportRadius;
                    session.Request.Layer0NeighborSearch.EnqueueBuild(
                        solver.asyncCb,
                        solver.pos,
                        activeCount,
                        cellSize,
                        supportRadius,
                        session.Request.Layer0NeighborBoundsMin,
                        session.Request.Layer0NeighborBoundsMax,
                        session.Request.ReadSlot,
                        session.Request.WriteSlot,
                        Const.DTFixIterations,
                        Const.DTLegalizeIterations,
                        true);
                }
            }

            if (!solver.layerMappingCache.TryBuildLayerContext(session, 0, out LayerContext layer0))
                return;

            RecordBuildLayer0CollisionEventsPerTick(
                layer0.NeighborSearch,
                layer0.ActiveCount,
                layer0.FineCount,
                layer0.KernelH,
                tickContext.TickIndex,
                layer0.UseMappedIndices,
                layer0.GlobalNodeMap,
                layer0.GlobalToLocalMap,
                layer0.OwnerByLocalBuffer);
        }

        /// <summary>
        /// Records layer-0 collision event build for the current tick.
        /// </summary>
        private void RecordBuildLayer0CollisionEventsPerTick(
            INeighborSearch layer0NeighborSearch,
            int layer0ActiveCount,
            int layer0FineCount,
            float layer0KernelH,
            int tickIndex,
            bool useDtGlobalNodeMap,
            ComputeBuffer dtGlobalNodeMap,
            ComputeBuffer dtGlobalToLayerLocalMap,
            ComputeBuffer dtOwnerByLocal
        ) {
            if (layer0NeighborSearch == null || layer0ActiveCount < 3)
                return;

            solver.LayerSolve.PrepareRelaxBuffers(new LayerSolve.RelaxBufferContext(
                layer0NeighborSearch,
                0,
                layer0ActiveCount,
                layer0FineCount,
                tickIndex,
                layer0KernelH,
                new LayerSolve.DtMappingContext(
                    useDtGlobalNodeMap,
                    0,
                    dtGlobalNodeMap,
                    dtGlobalToLayerLocalMap,
                    dtOwnerByLocal)));
            solver.asyncCb.SetComputeIntParam(solver.shader, "_UseTransferredCollisions", UseTransferredCollisionsDisabled);
            solver.Dispatch("XPBI.ClearCollisionEventCount", solver.shader, kClearCollisionEventCount, 1, 1, 1);
            solver.Dispatch("XPBI.BuildCollisionEventsL0", solver.shader, kBuildCollisionEventsL0, XPBISolver.Groups256(layer0ActiveCount), 1, 1);
        }
    }
}
