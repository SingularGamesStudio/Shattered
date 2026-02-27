public static class Const {
    public const float Gravity = -0.01f;
    public const float Compliance = 0.1f;

    // Neighbors
    public const int NeighborCount = 16;
    public const float LayerKernelHFromPoissonRadius = 2f;
    public const float WendlandSupport = 2.0f;

    // XPBI Solver
    public const int JRIterationsLMax = 2;
    public const int JRIterationsLMid = 4;
    public const int GSIterationsL0 = 4;
    public const int JRIterationsL0 = 8;
    public const float JROmegaV = 0.3f;
    public const float JROmegaL = 0.3f;
    public const int PersistentCoarseMaxNodes = 256;

    // DT maintenance
    public const int ColoringConflictRounds = 3;
    public const int InitColoringConflictRounds = 24;
    public const int DTFixIterations = 2;
    public const int DTLegalizeIterations = 2;

    // Stability limits
    public const float MaxVelocity = 0.0f;
    public const float MaxDisplacementPerTick = 0.0f;

    // Multigrid prolongation
    public const float ProlongationScale = 0.5f;
    public const float RestrictedDeltaVScale = 0.75f;
    public const bool UseAffineProlongation = true;
    public const float RestrictResidualDeltaVScale = 0.35f;
    public const float PostProlongSmoothing = 0.2f;

    // Collisions
    public const float CollisionSupportScale = 0.3f;
    public const float CollisionCompliance = 0.15f;
    public const float CollisionFriction = 0.6f;
    public const float CollisionRestitution = 0.0f;
    public const float CollisionRestitutionThreshold = 0.005f;

}
