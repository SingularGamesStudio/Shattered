public static class Const {
    public const int NeighborCount = 16;
    public const float LayerKernelHFromPoissonRadius = 2f;

    public static int IterationsL0 = 25;
    public const int IterationsLMax = 2;
    public const int IterationsLMid = 4;
    public const bool EnableTwoStageGS2 = true;
    public const int TwoStagePreGsItersL0 = 4;
    public const float TwoStageJROmegaV = 0.3f;
    public const float TwoStageJROmegaL = 0.3f;

    public const int ColoringConflictRounds = 24;
    public const float RestrictedDeltaVScale = 0.75f;
    public const int DTFixIterations = 2;
    public const int DTLegalizeIterations = 2;
    public const bool EnablePerTickDTMaintain = true;


    public const float WendlandSupport = 2.0f;
    public const float Gravity = -0.01f;
    public const float Compliance = 1.0f;
    public const float MaxVelocity = 0.0f;
    public const float MaxDisplacementPerTick = 0.0f;

    public const float ProlongationScale = 0.5f;
    public const bool UseAffineProlongation = true;
    public const bool UseResidualVCycle = true;
    public const int HierarchyVCyclesPerTick = 1;
    public const float RestrictResidualDeltaVScale = 0.35f;
    public const float PostProlongSmoothing = 0.2f;

    public const bool EnableCollisionConstraints = true;
    public const float CollisionSupportScale = 0.6f;
    public const float CollisionCompliance = 0.005f;
    public const float CollisionFriction = 0.35f;
    public const float CollisionRestitution = 0.0f;
    public const float CollisionRestitutionThreshold = 0.005f;

    public const bool DebugSupportRadius = false;
}
