using System.Collections.Generic;
using GPU.Delaunay;
using GPU.Neighbors;
using UnityEngine;
using LayerContext = GPU.Solver.XPBISolver.LayerContext;

namespace GPU.Solver {
    internal sealed partial class Coloring {
        private readonly XPBISolver solver;
        private readonly Dictionary<ulong, DTColoring> coloringByMeshLayer = new Dictionary<ulong, DTColoring>(128);
        private readonly Dictionary<int, LayerColoringMaskCache> coloringMaskCacheByLayer = new Dictionary<int, LayerColoringMaskCache>(8);

        private sealed class LayerColoringMaskCache {
            public uint[] mask;
            public int[] ownerRef;
            public int activeCount;
            public int fixedSignature;
        }

        public Coloring(XPBISolver solver) {
            this.solver = solver;
        }

        /// <summary>
        /// Rebuilds global coloring for the layer with current ownership/fixed-object filters.
        /// </summary>
        public DTColoring RebuildForLayer(LayerContext layerContext, int fixedObjectSignature) {
            return RebuildGlobalColoringForLayer(
                layerContext.Layer,
                layerContext.NeighborSearch,
                layerContext.OwnerBodyByLocal,
                fixedObjectSignature,
                layerContext.ActiveCount,
                layerContext.KernelH);
        }

        public void RequestConflictHistoryDebugLog(int maxSolveLayer, bool[] coloringUpdatedByLayer) {
            for (int layer = maxSolveLayer; layer >= 0; layer--) {
                if (coloringUpdatedByLayer == null || layer < 0 || layer >= coloringUpdatedByLayer.Length || !coloringUpdatedByLayer[layer])
                    continue;

                ulong key = 0xFFFFFFFF00000000UL | (uint)layer;
                if (!coloringByMeshLayer.TryGetValue(key, out var layerColoring) || layerColoring == null)
                    continue;

                int recordedIterations = layerColoring.GetRecordedConflictIterationCount();
                if (recordedIterations <= 0)
                    continue;

                int capturedLayer = layer;
                layerColoring.ReadConflictHistoryAsync(conflicts => {
                    if (conflicts == null || conflicts.Length == 0) {
                        Debug.LogError($"Coloring conflicts per iteration L{capturedLayer}: unavailable");
                        return;
                    }

                    var line = new System.Text.StringBuilder();
                    line.Append("Coloring conflicts per iteration L")
                        .Append(capturedLayer)
                        .Append(": ");

                    for (int i = 0; i < conflicts.Length; i++) {
                        if (i > 0)
                            line.Append(", ");
                        line.Append("i").Append(i).Append("=").Append(conflicts[i]);
                    }

                    Debug.LogError(line.ToString());
                });
            }
        }

        /// <summary>
        /// Builds or reuses the per-layer active node mask used by coloring rebuild.
        /// </summary>
        private uint[] BuildLayerColoringActiveMask(int layer, int[] ownerBodyByLocal, int fixedObjectSignature, int activeCount) {
            if (!coloringMaskCacheByLayer.TryGetValue(layer, out LayerColoringMaskCache cache) || cache == null) {
                cache = new LayerColoringMaskCache();
                coloringMaskCacheByLayer[layer] = cache;
            }

            if (cache.mask == null || cache.mask.Length < activeCount)
                cache.mask = new uint[Mathf.Max(1, activeCount)];

            bool needsRebuild =
                cache.ownerRef != ownerBodyByLocal ||
                cache.activeCount != activeCount ||
                cache.fixedSignature != fixedObjectSignature;

            if (needsRebuild) {
                bool hasOwners = ownerBodyByLocal != null && ownerBodyByLocal.Length >= activeCount;
                for (int i = 0; i < activeCount; i++) {
                    bool include = true;

                    if (hasOwners) {
                        int owner = ownerBodyByLocal[i];
                        if (owner >= 0 && owner < solver.solveRanges.Count) {
                            Meshless ownerMesh = solver.solveRanges[owner].meshless;
                            include = ownerMesh != null && !ownerMesh.fixedObject;
                        }
                    }

                    cache.mask[i] = include ? 1u : 0u;
                }

                cache.ownerRef = ownerBodyByLocal;
                cache.activeCount = activeCount;
                cache.fixedSignature = fixedObjectSignature;
            }

            return cache.mask;
        }

        /// <summary>
        /// Rebuilds and updates global coloring data for one layer based on current neighborhood and activity mask.
        /// </summary>
        private DTColoring RebuildGlobalColoringForLayer(int layer, INeighborSearch neighborSearch, int[] ownerBodyByLocal, int fixedObjectSignature, int activeCount, float layerCellSize) {
            if (solver.coloringShader == null) {
                Debug.LogError("XPBISolver: No coloring shader provided. Cannot rebuild global coloring.");
                return null;
            }

            ulong key = 0xFFFFFFFF00000000UL | (uint)layer;
            if (!coloringByMeshLayer.TryGetValue(key, out DTColoring layerColoring) || layerColoring == null) {
                layerColoring = new DTColoring(solver.coloringShader);
                coloringByMeshLayer[key] = layerColoring;
            }

            uint[] activeMask = BuildLayerColoringActiveMask(layer, ownerBodyByLocal, fixedObjectSignature, activeCount);

            uint seed = 12345u + (uint)layer;
            if (layerColoring.ColorBuffer == null) {
                layerColoring.Init(activeCount, neighborSearch.NeighborCount, seed);
                layerColoring.UpdateActiveMask(activeMask, activeCount);
                layerColoring.EnqueueInitTriGrid(solver.asyncCb, solver.pos, layerCellSize);
                neighborSearch.MarkAllDirty(solver.asyncCb);
                layerColoring.EnqueueUpdateAfterMaintain(solver.asyncCb, solver.pos, neighborSearch, layerCellSize, Const.InitColoringConflictRounds);
            } else {
                layerColoring.UpdateActiveMask(activeMask, activeCount);
                layerColoring.EnqueueUpdateAfterMaintain(solver.asyncCb, solver.pos, neighborSearch, layerCellSize, Const.ColoringConflictRounds);
            }

            layerColoring.EnqueueRebuildOrderAndArgs(solver.asyncCb);
            return layerColoring;
        }

        public void Release() {
            foreach (var kv in coloringByMeshLayer)
                kv.Value?.Dispose();
            coloringByMeshLayer.Clear();
            coloringMaskCacheByLayer.Clear();
        }
    }
}
