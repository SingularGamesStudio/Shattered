namespace GPU.Solver {
    public sealed partial class XPBISolver {
        private int kApplyGameplayForces;
        private int kClampVelocities;
        private int kExternalForces;
        private int kClearHierarchicalStats;
        private int kCacheHierarchicalStats;
        private int kFinalizeHierarchicalStats;
        private int kComputeCorrectionL;
        private int kCacheF0AndResetLambda;
        private int kSaveVelPrefix;
        private int kClearVelDelta;
        private int kResetCollisionLambda;
        private int kClearCollisionEventCount;
        private int kBuildCollisionEventsL0;
        private int kClearTransferredCollision;
        private int kRestrictCollisionEventsToActivePairs;
        private int kRelaxColored;
        private int kRelaxColoredPersistentCoarse;
        private int kJRSavePrevAndClear;
        private int kJRComputeDeltas;
        private int kJRApply;
        private int kProlongate;
        private int kCommitDeformation;
        private int kIntegratePositions;
        private int kUpdateDtPositions;
        private int kUpdateDtPositionsMapped;
        private int kRebuildParentsAtLayer;
        private int kClearConvergenceDebugStats;
        private int kClearRestrictedDeltaV;
        private int kRestrictGameplayDeltaVFromEvents;
        private int kRestrictFineVelocityResidualToActive;
        private int kApplyRestrictedDeltaVToActiveAndPrefix;
        private int kRemoveRestrictedDeltaVFromActive;
        private int kSmoothProlongatedFineVel;

        private bool kernelsCached;

        void EnsureKernelsCached() {
            if (kernelsCached) return;

            kApplyGameplayForces = shader.FindKernel("ApplyGameplayForces");
            kClampVelocities = shader.FindKernel("ClampVelocities");
            kExternalForces = shader.FindKernel("ExternalForces");

            kClearHierarchicalStats = shader.FindKernel("ClearHierarchicalStats");
            kCacheHierarchicalStats = shader.FindKernel("CacheHierarchicalStats");
            kFinalizeHierarchicalStats = shader.FindKernel("FinalizeHierarchicalStats");
            kComputeCorrectionL = shader.FindKernel("ComputeCorrectionL");
            kCacheF0AndResetLambda = shader.FindKernel("CacheF0AndResetLambda");
            kSaveVelPrefix = shader.FindKernel("SaveVelPrefix");
            kClearVelDelta = shader.FindKernel("ClearVelDelta");
            kResetCollisionLambda = shader.FindKernel("ResetCollisionLambda");
            kClearCollisionEventCount = shader.FindKernel("ClearCollisionEventCount");
            kBuildCollisionEventsL0 = shader.FindKernel("BuildCollisionEventsL0");
            kClearTransferredCollision = shader.FindKernel("ClearTransferredCollision");
            kRestrictCollisionEventsToActivePairs = shader.FindKernel("RestrictCollisionEventsToActivePairs");
            kRelaxColored = shader.FindKernel("RelaxColored");
            kRelaxColoredPersistentCoarse = shader.FindKernel("RelaxColoredPersistentCoarse");
            kJRSavePrevAndClear = shader.FindKernel("JR_SavePrevAndClear");
            kJRComputeDeltas = shader.FindKernel("JR_ComputeDeltas");
            kJRApply = shader.FindKernel("JR_Apply");
            kProlongate = shader.FindKernel("Prolongate");
            kCommitDeformation = shader.FindKernel("CommitDeformation");

            kIntegratePositions = shader.FindKernel("IntegratePositions");
            kUpdateDtPositions = shader.FindKernel("UpdateDtPositions");
            kUpdateDtPositionsMapped = shader.FindKernel("UpdateDtPositionsMapped");
            kRebuildParentsAtLayer = shader.FindKernel("RebuildParentsAtLayer");
            kClearConvergenceDebugStats = shader.FindKernel("ClearConvergenceDebugStats");

            kClearRestrictedDeltaV = shader.FindKernel("ClearRestrictedDeltaV");
            kRestrictGameplayDeltaVFromEvents = shader.FindKernel("RestrictGameplayDeltaVFromEvents");
            kRestrictFineVelocityResidualToActive = shader.FindKernel("RestrictFineVelocityResidualToActive");
            kApplyRestrictedDeltaVToActiveAndPrefix = shader.FindKernel("ApplyRestrictedDeltaVToActiveAndPrefix");
            kRemoveRestrictedDeltaVFromActive = shader.FindKernel("RemoveRestrictedDeltaVFromActive");
            kSmoothProlongatedFineVel = shader.FindKernel("SmoothProlongatedFineVel");

            kernelsCached = true;
        }
    }
}