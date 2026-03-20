using UnityEngine;
using UnityEngine.Rendering;
using SolveSession = GPU.Solver.XPBISolver.SolveSession;
using TickContext = GPU.Solver.XPBISolver.TickContext;

namespace GPU.Solver {
    internal sealed partial class GameplayForces {
        private readonly ComputeShader shader;

        public int EventCount => forceEventsCount;
        public bool HasEventsBuffer => forceEvents != null;
        public ComputeBuffer ForceEventsBuffer => forceEvents;

        public GameplayForces(XPBISolver solver) {
            shader = solver.GameplayForcesShader;
        }

        private static void Dispatch(CommandBuffer cb, string marker, ComputeShader dispatchShader, int kernel, int groupsX, int groupsY, int groupsZ) {
            cb.BeginSample(marker);
            cb.DispatchCompute(dispatchShader, kernel, groupsX, groupsY, groupsZ);
            cb.EndSample(marker);
        }

        /// <summary>
        /// Uploads gameplay forces to GPU buffers.
        /// </summary>
        public void SetForces(XPBISolver.ForceEvent[] events, int count) {
            if (events == null) throw new System.ArgumentNullException(nameof(events));
            if (count < 0 || count > events.Length) throw new System.ArgumentOutOfRangeException(nameof(count));

            forceEventsCount = count;
            if (forceEventsCount <= 0)
                return;

            EnsureCapacity(forceEventsCount);
            System.Array.Copy(events, 0, forceEventsCpu, 0, forceEventsCount);
            forceEvents.SetData(forceEventsCpu, 0, 0, forceEventsCount);
        }

        /// <summary>
        /// Clears gameplay force state for subsequent ticks.
        /// </summary>
        public void ClearForces() {
            forceEventsCount = 0;
        }

        /// <summary>
        /// Records gameplay and external force kernels for a tick.
        /// </summary>
        public void RecordApplyForces(SolveSession session, TickContext tickContext) {
            session.AsyncCb.SetComputeBufferParam(shader, kApplyGameplayForces, "_Vel", session.Vel);
            session.AsyncCb.SetComputeBufferParam(shader, kApplyGameplayForces, "_InvMass", session.InvMass);
            session.AsyncCb.SetComputeBufferParam(shader, kExternalForces, "_Vel", session.Vel);
            session.AsyncCb.SetComputeBufferParam(shader, kExternalForces, "_InvMass", session.InvMass);

            if (tickContext.ForceCount > 0 && forceEvents != null) {
                session.AsyncCb.SetComputeIntParam(shader, "_ForceEventCount", tickContext.ForceCount);
                session.AsyncCb.SetComputeBufferParam(shader, kApplyGameplayForces, "_ForceEvents", forceEvents);
                Dispatch(session.AsyncCb, "XPBI.ApplyGameplayForces", shader, kApplyGameplayForces, XPBISolver.Groups256(tickContext.ForceCount), 1, 1);
            } else {
                session.AsyncCb.SetComputeIntParam(shader, "_ForceEventCount", 0);
            }

            Dispatch(session.AsyncCb, "XPBI.ExternalForces", shader, kExternalForces, XPBISolver.Groups256(session.TotalCount), 1, 1);
        }

    }
}
