using UnityEngine;

namespace GPU.Solver {
    internal sealed partial class GameplayForces {
        private ComputeBuffer forceEvents;
        private XPBISolver.ForceEvent[] forceEventsCpu;
        private int forceEventsCapacity;
        private int forceEventsCount;
        private int kApplyGameplayForces;
        private int kExternalForces;

        public void CacheKernels(ComputeShader shader) {
            kApplyGameplayForces = shader.FindKernel("ApplyGameplayForces");
            kExternalForces = shader.FindKernel("ExternalForces");
        }

        public void EnsureCapacity(int requiredCount) {
            if (requiredCount <= forceEventsCapacity && forceEvents != null && forceEventsCpu != null)
                return;

            int newCapacity = Mathf.Max(64, forceEventsCapacity);
            while (newCapacity < requiredCount)
                newCapacity *= 2;
            forceEventsCapacity = newCapacity;

            forceEvents?.Dispose();
            forceEvents = new ComputeBuffer(forceEventsCapacity, sizeof(int) + sizeof(float) * 2, ComputeBufferType.Structured);
            forceEventsCpu = new XPBISolver.ForceEvent[forceEventsCapacity];
        }

        public void Release() {
            forceEvents?.Dispose();
            forceEvents = null;
            forceEventsCpu = null;
            forceEventsCapacity = 0;
            forceEventsCount = 0;
        }
    }
}
