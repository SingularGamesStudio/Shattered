using UnityEngine;
using SolveSession = GPU.Solver.XPBISolver.SolveSession;
using TickContext = GPU.Solver.XPBISolver.TickContext;

namespace GPU.Solver {
    internal sealed partial class GameplayForces {
        private readonly XPBISolver solver;

        public int EventCount => forceEventsCount;
        public bool HasEventsBuffer => forceEvents != null;
        public ComputeBuffer ForceEventsBuffer => forceEvents;

        public GameplayForces(XPBISolver solver) {
            this.solver = solver;
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
            solver.asyncCb.SetComputeBufferParam(solver.shader, kApplyGameplayForces, "_Vel", solver.vel);
            solver.asyncCb.SetComputeBufferParam(solver.shader, kApplyGameplayForces, "_InvMass", solver.invMass);
            solver.asyncCb.SetComputeBufferParam(solver.shader, kExternalForces, "_Vel", solver.vel);
            solver.asyncCb.SetComputeBufferParam(solver.shader, kExternalForces, "_InvMass", solver.invMass);

            if (tickContext.ForceCount > 0 && forceEvents != null) {
                solver.asyncCb.SetComputeIntParam(solver.shader, "_ForceEventCount", tickContext.ForceCount);
                solver.asyncCb.SetComputeBufferParam(solver.shader, kApplyGameplayForces, "_ForceEvents", forceEvents);
                solver.Dispatch("XPBI.ApplyGameplayForces", solver.shader, kApplyGameplayForces, XPBISolver.Groups256(tickContext.ForceCount), 1, 1);
            } else {
                solver.asyncCb.SetComputeIntParam(solver.shader, "_ForceEventCount", 0);
            }

            solver.Dispatch("XPBI.ExternalForces", solver.shader, kExternalForces, XPBISolver.Groups256(session.TotalCount), 1, 1);
        }

    }
}
