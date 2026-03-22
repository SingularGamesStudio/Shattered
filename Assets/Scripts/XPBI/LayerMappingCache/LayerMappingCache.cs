using System;
using System.Collections.Generic;
using GPU.Delaunay;
using GPU.Neighbors;
using Unity.Mathematics;
using UnityEngine;
using LayerContext = GPU.Solver.XPBISolver.LayerContext;
using SolveSession = GPU.Solver.XPBISolver.SolveSession;

namespace GPU.Solver {
    internal sealed partial class LayerMappingCache {

        /// <summary>
        /// Builds execution context for one layer and materializes mapped-index buffers when needed.
        /// </summary>
        public bool TryBuildLayerContext(SolveSession session, int layer, out LayerContext context) {
            context = null;

            if (!session.Request.GlobalDTHierarchy.TryGetLayerDt(layer, out DT globalLayerDt) || globalLayerDt == null)
                return false;

            INeighborSearch layerNeighborSearch = session.UseOverrideLayer0NeighborSearch && layer == 0
                ? session.Request.Layer0NeighborSearch
                : globalLayerDt;
            if (layerNeighborSearch == null)
                return false;

            if (!session.Request.GlobalDTHierarchy.TryGetLayerMappings(layer, out int[] ownerBodyByLocal, out _, out int[] globalFineNodeByLocal, out int globalActiveCount, out int globalFineCount))
                return false;
            if (globalActiveCount < 3)
                return false;

            int[] collisionOwnerByLocal = ownerBodyByLocal;
            if (session.Request.GlobalDTHierarchy.TryGetLayerCollisionOwnerMapping(layer, out int[] collisionOwners) && collisionOwners != null)
                collisionOwnerByLocal = collisionOwners;

            if (!session.Request.GlobalDTHierarchy.TryGetLayerExecutionContext(layer, out int execActiveCount, out int execFineCount, out float layerKernelH))
                return false;

            bool useMappedIndices = !XPBISolver.IsIdentityMapping(globalFineNodeByLocal, globalFineCount);
            ComputeBuffer globalNodeMap = null;
            ComputeBuffer globalToLocalMap = null;
            if (useMappedIndices) {
                globalNodeMap = EnsureGlobalLayerNodeMapBuffer(layer, globalFineNodeByLocal, globalFineCount);
                globalToLocalMap = EnsureGlobalLayerGlobalToLocalBufferCached(layer, globalFineNodeByLocal, globalFineCount, session.TotalCount);
            }

            ComputeBuffer ownerByLocalBuffer = null;
            if (ownerBodyByLocal != null && ownerBodyByLocal.Length >= globalActiveCount)
                ownerByLocalBuffer = EnsureGlobalLayerOwnerByLocalBuffer(layer, ownerBodyByLocal, globalActiveCount);

            ComputeBuffer collisionOwnerByLocalBuffer = ownerByLocalBuffer;
            if (collisionOwnerByLocal != null && collisionOwnerByLocal.Length >= globalActiveCount)
                collisionOwnerByLocalBuffer = EnsureGlobalLayerCollisionOwnerByLocalBuffer(layer, collisionOwnerByLocal, globalActiveCount);

            context = new LayerContext {
                Layer = layer,
                NeighborSearch = layerNeighborSearch,
                OwnerBodyByLocal = ownerBodyByLocal,
                CollisionOwnerByLocal = collisionOwnerByLocal,
                ActiveCount = execActiveCount,
                FineCount = execFineCount,
                KernelH = layerKernelH,
                UseMappedIndices = useMappedIndices,
                GlobalNodeMap = globalNodeMap,
                GlobalToLocalMap = globalToLocalMap,
                OwnerByLocalBuffer = ownerByLocalBuffer,
                CollisionOwnerByLocalBuffer = collisionOwnerByLocalBuffer,
            };

            return true;
        }

    }
}

