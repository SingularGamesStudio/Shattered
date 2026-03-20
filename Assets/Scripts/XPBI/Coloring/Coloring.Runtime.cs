using UnityEngine;


namespace GPU.Solver {
    internal sealed partial class Coloring {
        private ComputeBuffer coloringColor;
        private ComputeBuffer coloringProposed;
        private ComputeBuffer coloringPrio;

        internal void AllocateRuntimeBuffers(int newCapacity) {
            coloringColor = new ComputeBuffer(newCapacity, sizeof(int), ComputeBufferType.Structured);
            coloringProposed = new ComputeBuffer(newCapacity, sizeof(int), ComputeBufferType.Structured);
            coloringPrio = new ComputeBuffer(newCapacity, sizeof(uint), ComputeBufferType.Structured);
        }

        internal void ReleaseRuntimeBuffers() {
            coloringColor?.Dispose(); coloringColor = null;
            coloringProposed?.Dispose(); coloringProposed = null;
            coloringPrio?.Dispose(); coloringPrio = null;
        }
    }
}
