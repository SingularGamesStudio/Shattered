using System.Collections.Generic;
using UnityEngine;
using SolveRequest = GPU.Solver.XPBISolver.SolveRequest;
using SolveSession = GPU.Solver.XPBISolver.SolveSession;

namespace GPU.Solver {
    internal sealed partial class SolveLayout {
        private readonly XPBISolver solver;

        public SolveLayout(XPBISolver solver) {
            this.solver = solver;
        }

        /// <summary>
        /// Validates request input, ensures capacities, and builds a complete solve session.
        /// </summary>
        public bool TryBuildSession(in SolveRequest request, out SolveSession session) {
            session = null;

            if (request.TickCount <= 0)
                return false;

            if (request.GlobalDTHierarchy == null || request.GlobalDTHierarchy.MaxLayer < 0)
                return false;

            if (!EnsureLayout(request.Meshes, out int totalCount, out int maxSolveLayer))
                return false;

            maxSolveLayer = Mathf.Max(maxSolveLayer, request.GlobalDTHierarchy.MaxLayer);

            bool useOverrideLayer0NeighborSearch = request.Layer0NeighborSearch != null;
            bool useHierarchical = request.UseHierarchical && !useOverrideLayer0NeighborSearch;
            if (!useHierarchical)
                maxSolveLayer = 0;

            solver.EnsureCapacity(totalCount);
            if (!solver.layoutInitialized)
                solver.InitializeFromMeshless(solver.solveRanges, totalCount);

            int convergenceDebugMaxLayer = useHierarchical ? maxSolveLayer : 0;
            int convergenceDebugLayerCount = convergenceDebugMaxLayer + 1;
            int convergenceDebugMaxIterations = Mathf.Max(
                Const.GSIterationsL0 + Const.JRIterationsL0,
                1 + Mathf.Max(Const.JRIterationsLMax, Const.JRIterationsLMid));

            session = new SolveSession {
                Request = request,
                TotalCount = totalCount,
                MaxSolveLayer = maxSolveLayer,
                UseHierarchical = useHierarchical,
                UseOverrideLayer0NeighborSearch = useOverrideLayer0NeighborSearch,
                EnableProlongationConstraintProbeDebug = Const.ProlongationConstraintDebugEnabled,
                FixedObjectSignature = ComputeFixedObjectSignature(),
                ConvergenceDebugMaxLayer = convergenceDebugMaxLayer,
                ConvergenceDebugLayerCount = convergenceDebugLayerCount,
                ConvergenceDebugMaxIterations = convergenceDebugMaxIterations,
            };

            session.MaxProlongationProbeSamples = session.EnableProlongationConstraintProbeDebug && session.UseHierarchical
                ? Mathf.Max(0, request.TickCount * Mathf.Max(0, maxSolveLayer) * 2)
                : 0;

            return true;
        }

        /// <summary>
        /// Builds or validates the active solve layout and derives aggregate node/layer counts.
        /// </summary>
        private bool EnsureLayout(IReadOnlyList<Meshless> meshes, out int totalCount, out int maxSolveLayer) {
            totalCount = 0;
            maxSolveLayer = 0;

            if (meshes == null || meshes.Count == 0)
                return false;

            bool changed = false;
            int validIndex = 0;

            for (int i = 0; i < meshes.Count; i++) {
                Meshless meshless = meshes[i];
                if (meshless == null || meshless.nodes == null || meshless.nodes.Count <= 0)
                    continue;

                int count = meshless.nodes.Count;
                if (validIndex >= solver.solveRanges.Count) {
                    changed = true;
                } else {
                    XPBISolver.MeshRange existing = solver.solveRanges[validIndex];
                    if (existing.meshless != meshless || existing.baseIndex != totalCount || existing.totalCount != count)
                        changed = true;
                }

                totalCount += count;
                maxSolveLayer = Mathf.Max(maxSolveLayer, meshless.maxLayer);
                validIndex++;
            }

            if (validIndex == 0 || totalCount <= 0)
                return false;

            if (validIndex != solver.solveRanges.Count)
                changed = true;

            if (changed) {
                solver.solveRanges.Clear();

                int baseIndex = 0;
                for (int i = 0; i < meshes.Count; i++) {
                    Meshless meshless = meshes[i];
                    if (meshless == null || meshless.nodes == null || meshless.nodes.Count <= 0)
                        continue;

                    int count = meshless.nodes.Count;
                    solver.solveRanges.Add(new XPBISolver.MeshRange {
                        meshless = meshless,
                        baseIndex = baseIndex,
                        totalCount = count,
                    });

                    baseIndex += count;
                }

                solver.layoutInitialized = false;
            }

            return true;
        }

        /// <summary>
        /// Computes a stable signature for fixed-object ownership used by coloring cache invalidation.
        /// </summary>
        private int ComputeFixedObjectSignature() {
            int signature = 17;
            for (int i = 0; i < solver.solveRanges.Count; i++) {
                Meshless meshless = solver.solveRanges[i].meshless;
                signature = unchecked(signature * 31 + ((meshless != null && meshless.fixedObject) ? 1 : 0));
            }

            return signature;
        }

    }
}
