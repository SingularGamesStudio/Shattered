using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;

namespace GPU.Solver {
    internal sealed partial class LayerMappingCache {
        private readonly Dictionary<int, ComputeBuffer> globalLayerNodeMapBuffers = new Dictionary<int, ComputeBuffer>(16);
        private readonly Dictionary<int, ComputeBuffer> globalLayerGlobalToLocalBuffers = new Dictionary<int, ComputeBuffer>(16);
        private readonly Dictionary<int, ComputeBuffer> globalLayerOwnerByLocalBuffers = new Dictionary<int, ComputeBuffer>(16);
        private readonly Dictionary<int, ComputeBuffer> globalLayerCollisionOwnerByLocalBuffers = new Dictionary<int, ComputeBuffer>(16);
        private readonly Dictionary<int, int[]> globalLayerGlobalToLocalCpu = new Dictionary<int, int[]>(16);
        private readonly Dictionary<int, int[]> globalLayerNodeMapRefs = new Dictionary<int, int[]>(16);
        private readonly Dictionary<int, int[]> globalLayerGlobalToLocalRefs = new Dictionary<int, int[]>(16);
        private readonly Dictionary<int, int[]> globalLayerOwnerByLocalRefs = new Dictionary<int, int[]>(16);
        private readonly Dictionary<int, int[]> globalLayerCollisionOwnerByLocalRefs = new Dictionary<int, int[]>(16);
        private readonly Dictionary<int, int> globalLayerNodeMapRefCounts = new Dictionary<int, int>(16);
        private readonly Dictionary<int, int> globalLayerGlobalToLocalRefCounts = new Dictionary<int, int>(16);
        private readonly Dictionary<int, int> globalLayerOwnerByLocalRefCounts = new Dictionary<int, int>(16);
        private readonly Dictionary<int, int> globalLayerCollisionOwnerByLocalRefCounts = new Dictionary<int, int>(16);
        private readonly Dictionary<int, int> globalLayerNodeMapHashes = new Dictionary<int, int>(16);
        private readonly Dictionary<int, int> globalLayerGlobalToLocalHashes = new Dictionary<int, int>(16);
        private readonly Dictionary<int, int> globalLayerOwnerByLocalHashes = new Dictionary<int, int>(16);
        private readonly Dictionary<int, int> globalLayerCollisionOwnerByLocalHashes = new Dictionary<int, int>(16);
        private readonly Dictionary<int, int> globalLayerGlobalToLocalTotals = new Dictionary<int, int>(16);
        private ComputeBuffer defaultDtGlobalNodeMap;
        private ComputeBuffer defaultDtGlobalToLocalMap;
        private ComputeBuffer defaultDtOwnerByLocal;
        internal ComputeBuffer DefaultDtOwnerByLocal => defaultDtOwnerByLocal;
        internal ComputeBuffer DefaultDtCollisionOwnerByLocal => defaultDtOwnerByLocal;

        internal void AllocateLayerMappingBuffers() {
            defaultDtGlobalNodeMap = new ComputeBuffer(1, sizeof(int), ComputeBufferType.Structured);
            defaultDtGlobalNodeMap.SetData(new[] { 0 });
            defaultDtGlobalToLocalMap = new ComputeBuffer(1, sizeof(int), ComputeBufferType.Structured);
            defaultDtGlobalToLocalMap.SetData(new[] { -1 });
            defaultDtOwnerByLocal = new ComputeBuffer(1, sizeof(int), ComputeBufferType.Structured);
            defaultDtOwnerByLocal.SetData(new[] { -1 });
        }

        internal void ReleaseLayerMappingBuffers() {
            foreach (var kv in globalLayerNodeMapBuffers)
                kv.Value?.Dispose();
            globalLayerNodeMapBuffers.Clear();
            globalLayerNodeMapRefs.Clear();
            globalLayerNodeMapRefCounts.Clear();
            globalLayerNodeMapHashes.Clear();

            foreach (var kv in globalLayerGlobalToLocalBuffers)
                kv.Value?.Dispose();
            globalLayerGlobalToLocalBuffers.Clear();
            globalLayerGlobalToLocalCpu.Clear();
            globalLayerGlobalToLocalRefs.Clear();
            globalLayerGlobalToLocalRefCounts.Clear();
            globalLayerGlobalToLocalHashes.Clear();
            globalLayerGlobalToLocalTotals.Clear();

            foreach (var kv in globalLayerOwnerByLocalBuffers)
                kv.Value?.Dispose();
            globalLayerOwnerByLocalBuffers.Clear();
            globalLayerOwnerByLocalRefs.Clear();
            globalLayerOwnerByLocalRefCounts.Clear();
            globalLayerOwnerByLocalHashes.Clear();

            foreach (var kv in globalLayerCollisionOwnerByLocalBuffers)
                kv.Value?.Dispose();
            globalLayerCollisionOwnerByLocalBuffers.Clear();
            globalLayerCollisionOwnerByLocalRefs.Clear();
            globalLayerCollisionOwnerByLocalRefCounts.Clear();
            globalLayerCollisionOwnerByLocalHashes.Clear();

            defaultDtGlobalNodeMap?.Dispose();
            defaultDtGlobalNodeMap = null;
            defaultDtGlobalToLocalMap?.Dispose();
            defaultDtGlobalToLocalMap = null;
            defaultDtOwnerByLocal?.Dispose();
            defaultDtOwnerByLocal = null;
        }

        private static int ComputeMappingHash(int[] mapping, int count) {
            unchecked {
                int hash = 17;
                hash = (hash * 31) ^ count;
                for (int i = 0; i < count; i++)
                    hash = (hash * 31) ^ mapping[i];
                return hash;
            }
        }

        internal ComputeBuffer EnsureGlobalLayerNodeMapBuffer(int layer, int[] globalNodeByLocal, int activeCount) {
            if (globalNodeByLocal == null)
                throw new ArgumentNullException(nameof(globalNodeByLocal));

            if (activeCount < 0 || activeCount > globalNodeByLocal.Length)
                throw new ArgumentOutOfRangeException(nameof(activeCount));

            if (!globalLayerNodeMapBuffers.TryGetValue(layer, out ComputeBuffer mapBuffer) || mapBuffer == null || !mapBuffer.IsValid() || mapBuffer.count != activeCount) {
                mapBuffer?.Dispose();
                mapBuffer = new ComputeBuffer(math.max(1, activeCount), sizeof(int), ComputeBufferType.Structured);
                globalLayerNodeMapBuffers[layer] = mapBuffer;
                globalLayerNodeMapHashes[layer] = int.MinValue;
            }

            if (globalLayerNodeMapRefs.TryGetValue(layer, out int[] previousRef) &&
                ReferenceEquals(previousRef, globalNodeByLocal) &&
                globalLayerNodeMapRefCounts.TryGetValue(layer, out int previousCount) &&
                previousCount == activeCount)
                return mapBuffer;

            int mappingHash = ComputeMappingHash(globalNodeByLocal, activeCount);
            bool shouldUpload = !globalLayerNodeMapHashes.TryGetValue(layer, out int previousHash) || previousHash != mappingHash;

            if (shouldUpload && activeCount > 0)
                mapBuffer.SetData(globalNodeByLocal, 0, 0, activeCount);

            if (shouldUpload)
                globalLayerNodeMapHashes[layer] = mappingHash;

            globalLayerNodeMapRefs[layer] = globalNodeByLocal;
            globalLayerNodeMapRefCounts[layer] = activeCount;

            return mapBuffer;
        }

        internal ComputeBuffer EnsureGlobalLayerGlobalToLocalBufferCached(int layer, int[] globalNodeByLocal, int activeCount, int totalNodeCount) {
            if (globalNodeByLocal == null)
                throw new ArgumentNullException(nameof(globalNodeByLocal));

            int safeTotal = math.max(1, totalNodeCount);

            if (!globalLayerGlobalToLocalBuffers.TryGetValue(layer, out ComputeBuffer mapBuffer) || mapBuffer == null || !mapBuffer.IsValid() || mapBuffer.count != safeTotal) {
                mapBuffer?.Dispose();
                mapBuffer = new ComputeBuffer(safeTotal, sizeof(int), ComputeBufferType.Structured);
                globalLayerGlobalToLocalBuffers[layer] = mapBuffer;
                globalLayerGlobalToLocalHashes[layer] = int.MinValue;
            }

            if (globalLayerGlobalToLocalRefs.TryGetValue(layer, out int[] previousRef) &&
                ReferenceEquals(previousRef, globalNodeByLocal) &&
                globalLayerGlobalToLocalRefCounts.TryGetValue(layer, out int previousCount) &&
                previousCount == activeCount &&
                globalLayerGlobalToLocalTotals.TryGetValue(layer, out int previousTotal) &&
                previousTotal == safeTotal)
                return mapBuffer;

            if (!globalLayerGlobalToLocalCpu.TryGetValue(layer, out int[] globalToLocal) || globalToLocal == null || globalToLocal.Length != safeTotal) {
                globalToLocal = new int[safeTotal];
                globalLayerGlobalToLocalCpu[layer] = globalToLocal;
            }

            int mappingHash = ComputeMappingHash(globalNodeByLocal, activeCount);
            bool totalChanged = !globalLayerGlobalToLocalTotals.TryGetValue(layer, out int prevTotal) || prevTotal != safeTotal;
            bool mappingChanged = !globalLayerGlobalToLocalHashes.TryGetValue(layer, out int prevHash) || prevHash != mappingHash;
            if (!totalChanged && !mappingChanged)
                return mapBuffer;

            for (int i = 0; i < globalToLocal.Length; i++)
                globalToLocal[i] = -1;

            for (int li = 0; li < activeCount; li++) {
                int gi = globalNodeByLocal[li];
                if (gi >= 0 && gi < globalToLocal.Length)
                    globalToLocal[gi] = li;
            }

            mapBuffer.SetData(globalToLocal);
            globalLayerGlobalToLocalHashes[layer] = mappingHash;
            globalLayerGlobalToLocalTotals[layer] = safeTotal;
            globalLayerGlobalToLocalRefs[layer] = globalNodeByLocal;
            globalLayerGlobalToLocalRefCounts[layer] = activeCount;
            return mapBuffer;
        }

        internal ComputeBuffer EnsureGlobalLayerOwnerByLocalBuffer(int layer, int[] ownerBodyByLocal, int activeCount) {
            if (ownerBodyByLocal == null)
                throw new ArgumentNullException(nameof(ownerBodyByLocal));

            if (activeCount < 0 || activeCount > ownerBodyByLocal.Length)
                throw new ArgumentOutOfRangeException(nameof(activeCount));

            if (!globalLayerOwnerByLocalBuffers.TryGetValue(layer, out ComputeBuffer ownerBuffer) || ownerBuffer == null || !ownerBuffer.IsValid() || ownerBuffer.count != math.max(1, activeCount)) {
                ownerBuffer?.Dispose();
                ownerBuffer = new ComputeBuffer(math.max(1, activeCount), sizeof(int), ComputeBufferType.Structured);
                globalLayerOwnerByLocalBuffers[layer] = ownerBuffer;
                globalLayerOwnerByLocalHashes[layer] = int.MinValue;
            }

            if (globalLayerOwnerByLocalRefs.TryGetValue(layer, out int[] previousRef) &&
                ReferenceEquals(previousRef, ownerBodyByLocal) &&
                globalLayerOwnerByLocalRefCounts.TryGetValue(layer, out int previousCount) &&
                previousCount == activeCount)
                return ownerBuffer;

            int ownerHash = ComputeMappingHash(ownerBodyByLocal, activeCount);
            bool shouldUpload = !globalLayerOwnerByLocalHashes.TryGetValue(layer, out int previousHash) || previousHash != ownerHash;
            if (shouldUpload && activeCount > 0)
                ownerBuffer.SetData(ownerBodyByLocal, 0, 0, activeCount);

            if (shouldUpload)
                globalLayerOwnerByLocalHashes[layer] = ownerHash;

            globalLayerOwnerByLocalRefs[layer] = ownerBodyByLocal;
            globalLayerOwnerByLocalRefCounts[layer] = activeCount;
            return ownerBuffer;
        }

        internal ComputeBuffer EnsureGlobalLayerCollisionOwnerByLocalBuffer(int layer, int[] collisionOwnerByLocal, int activeCount) {
            if (collisionOwnerByLocal == null)
                throw new ArgumentNullException(nameof(collisionOwnerByLocal));

            if (activeCount < 0 || activeCount > collisionOwnerByLocal.Length)
                throw new ArgumentOutOfRangeException(nameof(activeCount));

            if (!globalLayerCollisionOwnerByLocalBuffers.TryGetValue(layer, out ComputeBuffer ownerBuffer) || ownerBuffer == null || !ownerBuffer.IsValid() || ownerBuffer.count != math.max(1, activeCount)) {
                ownerBuffer?.Dispose();
                ownerBuffer = new ComputeBuffer(math.max(1, activeCount), sizeof(int), ComputeBufferType.Structured);
                globalLayerCollisionOwnerByLocalBuffers[layer] = ownerBuffer;
                globalLayerCollisionOwnerByLocalHashes[layer] = int.MinValue;
            }

            if (globalLayerCollisionOwnerByLocalRefs.TryGetValue(layer, out int[] previousRef) &&
                ReferenceEquals(previousRef, collisionOwnerByLocal) &&
                globalLayerCollisionOwnerByLocalRefCounts.TryGetValue(layer, out int previousCount) &&
                previousCount == activeCount)
                return ownerBuffer;

            int ownerHash = ComputeMappingHash(collisionOwnerByLocal, activeCount);
            bool shouldUpload = !globalLayerCollisionOwnerByLocalHashes.TryGetValue(layer, out int previousHash) || previousHash != ownerHash;
            if (shouldUpload && activeCount > 0)
                ownerBuffer.SetData(collisionOwnerByLocal, 0, 0, activeCount);

            if (shouldUpload)
                globalLayerCollisionOwnerByLocalHashes[layer] = ownerHash;

            globalLayerCollisionOwnerByLocalRefs[layer] = collisionOwnerByLocal;
            globalLayerCollisionOwnerByLocalRefCounts[layer] = activeCount;
            return ownerBuffer;
        }

        internal void BindDtGlobalMappingParams(CommandBuffer cb, ComputeShader shader, int kernel, bool useDtGlobalNodeMap, int dtLocalBase, ComputeBuffer dtGlobalNodeMap, ComputeBuffer dtGlobalToLayerLocalMap) {
            cb.SetComputeIntParam(shader, "_UseDtGlobalNodeMap", useDtGlobalNodeMap ? 1 : 0);
            cb.SetComputeIntParam(shader, "_DtLocalBase", dtLocalBase);

            cb.SetComputeBufferParam(shader, kernel, "_DtGlobalNodeMap", dtGlobalNodeMap ?? defaultDtGlobalNodeMap);
            cb.SetComputeBufferParam(shader, kernel, "_DtGlobalToLayerLocalMap", dtGlobalToLayerLocalMap ?? defaultDtGlobalToLocalMap);
        }
    }
}
