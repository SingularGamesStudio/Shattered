namespace GPU.Solver {
    public sealed partial class XPBISolver {
        private int kApplyGameplayForces;
        private int kExternalForces;
        private int kClearHierarchicalStats;
        private int kCacheHierarchicalStats;
        private int kFinalizeHierarchicalStats;
        private int kCacheKernelH;
        private int kComputeCorrectionL;
        private int kCacheF0AndResetLambda;
        private int kSaveVelPrefix;
        private int kClearVelDelta;
        private int kRelaxColored;
        private int kProlongate;
        private int kCommitDeformation;
        private int kIntegratePositions;
        private int kUpdateDtPositions;
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
            kExternalForces = shader.FindKernel("ExternalForces");

            kClearHierarchicalStats = shader.FindKernel("ClearHierarchicalStats");
            kCacheHierarchicalStats = shader.FindKernel("CacheHierarchicalStats");
            kFinalizeHierarchicalStats = shader.FindKernel("FinalizeHierarchicalStats");
            kCacheKernelH = shader.FindKernel("CacheKernelH");
            kComputeCorrectionL = shader.FindKernel("ComputeCorrectionL");
            kCacheF0AndResetLambda = shader.FindKernel("CacheF0AndResetLambda");
            kSaveVelPrefix = shader.FindKernel("SaveVelPrefix");
            kClearVelDelta = shader.FindKernel("ClearVelDelta");
            kRelaxColored = shader.FindKernel("RelaxColored");
            kProlongate = shader.FindKernel("Prolongate");
            kCommitDeformation = shader.FindKernel("CommitDeformation");

            kIntegratePositions = shader.FindKernel("IntegratePositions");
            kUpdateDtPositions = shader.FindKernel("UpdateDtPositions");
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