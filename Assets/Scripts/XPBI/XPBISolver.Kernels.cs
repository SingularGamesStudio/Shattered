namespace GPU.Solver {
    public sealed partial class XPBISolver {
        // Kernel IDs – cached after first successful HasAllKernels().
        private int kApplyGameplayForces;
        private int kExternalForces;
        private int kClearCurrentVolume;
        private int kCacheVolumesHierarchical;
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
        private int kRebuildParentsAtLevel;
        private int kColoringInit;
        private int kColoringDetectConflicts;
        private int kColoringApply;
        private int kColoringChoose;
        private int kColoringClearMeta;
        private int kColoringBuildCounts;
        private int kColoringBuildStarts;
        private int kColoringScatterOrder;
        private int kColoringBuildRelaxArgs;

        private bool kernelsCached;

        bool HasAllKernels() {
            return
                shader.HasKernel("ClearCurrentVolume") &&
                shader.HasKernel("CacheVolumesHierarchical") &&
                shader.HasKernel("CacheKernelH") &&
                shader.HasKernel("ComputeCorrectionL") &&
                shader.HasKernel("CacheF0AndResetLambda") &&
                shader.HasKernel("SaveVelPrefix") &&
                shader.HasKernel("ClearVelDelta") &&
                shader.HasKernel("RelaxColored") &&
                shader.HasKernel("Prolongate") &&
                shader.HasKernel("CommitDeformation") &&
                shader.HasKernel("ExternalForces") &&

                shader.HasKernel("ApplyGameplayForces") &&
                shader.HasKernel("IntegratePositions") &&
                shader.HasKernel("UpdateDtPositions") &&
                shader.HasKernel("RebuildParentsAtLevel") &&

                shader.HasKernel("ColoringInit") &&
                shader.HasKernel("ColoringDetectConflicts") &&
                shader.HasKernel("ColoringApply") &&
                shader.HasKernel("ColoringChoose") &&
                shader.HasKernel("ColoringClearMeta") &&
                shader.HasKernel("ColoringBuildCounts") &&
                shader.HasKernel("ColoringBuildStarts") &&
                shader.HasKernel("ColoringScatterOrder") &&
                shader.HasKernel("ColoringBuildRelaxArgs");
        }

        void EnsureKernelsCached() {
            if (kernelsCached) return;

            kApplyGameplayForces = shader.FindKernel("ApplyGameplayForces");
            kExternalForces = shader.FindKernel("ExternalForces");

            kClearCurrentVolume = shader.FindKernel("ClearCurrentVolume");
            kCacheVolumesHierarchical = shader.FindKernel("CacheVolumesHierarchical");
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
            kRebuildParentsAtLevel = shader.FindKernel("RebuildParentsAtLevel");

            kColoringInit = shader.FindKernel("ColoringInit");
            kColoringDetectConflicts = shader.FindKernel("ColoringDetectConflicts");
            kColoringApply = shader.FindKernel("ColoringApply");
            kColoringChoose = shader.FindKernel("ColoringChoose");
            kColoringClearMeta = shader.FindKernel("ColoringClearMeta");
            kColoringBuildCounts = shader.FindKernel("ColoringBuildCounts");
            kColoringBuildStarts = shader.FindKernel("ColoringBuildStarts");
            kColoringScatterOrder = shader.FindKernel("ColoringScatterOrder");
            kColoringBuildRelaxArgs = shader.FindKernel("ColoringBuildRelaxArgs");

            kernelsCached = true;
        }
    }
}