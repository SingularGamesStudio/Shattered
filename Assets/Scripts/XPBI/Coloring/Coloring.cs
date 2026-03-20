using System.Collections.Generic;
using GPU.Neighbors;
using UnityEngine;
using UnityEngine.Rendering;
using LayerContext = GPU.Solver.XPBISolver.LayerContext;
using SolveSession = GPU.Solver.XPBISolver.SolveSession;

namespace GPU.Solver {
    internal sealed partial class Coloring {
        private readonly ComputeShader shader;
        private readonly Dictionary<ulong, NeighborColoring> coloringByMeshLayer = new Dictionary<ulong, NeighborColoring>(128);
        private readonly Dictionary<int, LayerColoringMaskCache> coloringMaskCacheByLayer = new Dictionary<int, LayerColoringMaskCache>(8);

        private sealed class LayerColoringMaskCache {
            public uint[] mask;
            public int[] ownerRef;
            public int activeCount;
            public int fixedSignature;
        }

        public Coloring(XPBISolver solver) {
            shader = solver.coloringShader;
        }

        /// <summary>
        /// Rebuilds global coloring for the layer with current ownership/fixed-object filters.
        /// </summary>
        public NeighborColoring RebuildForLayer(CommandBuffer cb, SolveSession session, LayerContext layerContext, int fixedObjectSignature) {
            return RebuildGlobalColoringForLayer(
            cb,
                layerContext.Layer,
                layerContext.NeighborSearch,
                layerContext.OwnerBodyByLocal,
                fixedObjectSignature,
                layerContext.ActiveCount,
            layerContext.KernelH,
            session.Pos,
            session.SolveRanges);
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
        private static uint[] BuildLayerColoringActiveMask(LayerColoringMaskCache cache, IReadOnlyList<XPBISolver.MeshRange> solveRanges, int[] ownerBodyByLocal, int fixedObjectSignature, int activeCount) {
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
                        if (owner >= 0 && owner < solveRanges.Count) {
                            Meshless ownerMesh = solveRanges[owner].meshless;
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
        private NeighborColoring RebuildGlobalColoringForLayer(
            CommandBuffer cb,
            int layer,
            INeighborSearch neighborSearch,
            int[] ownerBodyByLocal,
            int fixedObjectSignature,
            int activeCount,
            float layerCellSize,
            ComputeBuffer pos,
            IReadOnlyList<XPBISolver.MeshRange> solveRanges
        ) {
            if (shader == null) {
                Debug.LogError("XPBISolver: No coloring shader provided. Cannot rebuild global coloring.");
                return null;
            }

            if (pos == null || solveRanges == null)
                return null;

            ulong key = 0xFFFFFFFF00000000UL | (uint)layer;
            if (!coloringByMeshLayer.TryGetValue(key, out NeighborColoring layerColoring) || layerColoring == null) {
                layerColoring = new NeighborColoring(shader);
                coloringByMeshLayer[key] = layerColoring;
            }

            if (!coloringMaskCacheByLayer.TryGetValue(layer, out LayerColoringMaskCache cache) || cache == null) {
                cache = new LayerColoringMaskCache();
                coloringMaskCacheByLayer[layer] = cache;
            }

            uint[] activeMask = BuildLayerColoringActiveMask(cache, solveRanges, ownerBodyByLocal, fixedObjectSignature, activeCount);

            uint seed = 12345u + (uint)layer;
            if (layerColoring.ColorBuffer == null) {
                layerColoring.Init(activeCount, neighborSearch.NeighborCount, seed);
                layerColoring.UpdateActiveMask(activeMask, activeCount);
                layerColoring.EnqueueInitTriGrid(cb, pos, layerCellSize);
                neighborSearch.MarkAllDirty(cb);
                layerColoring.EnqueueUpdateAfterMaintain(cb, pos, neighborSearch, layerCellSize, Const.InitColoringConflictRounds);
            } else {
                layerColoring.UpdateActiveMask(activeMask, activeCount);
                layerColoring.EnqueueUpdateAfterMaintain(cb, pos, neighborSearch, layerCellSize, Const.ColoringConflictRounds);
            }

            layerColoring.EnqueueRebuildOrderAndArgs(cb);
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
