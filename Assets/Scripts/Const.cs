public static class Const {
    public static float Gravity => SimulationParamSource.Current.solverCore.gravity;
    public static float Compliance => SimulationParamSource.Current.solverCore.compliance;

    // Global point generation
    public static float Layer0PointDensity => SimulationParamSource.Current.pointCloud.layer0PointDensity;
    public static float LayerDownsampleRatio => SimulationParamSource.Current.pointCloud.layerDownsampleRatio;
    public static int MinVerticesPerLayer => SimulationParamSource.Current.pointCloud.minVerticesPerLayer;
    public static int MaxAutoLayers => SimulationParamSource.Current.pointCloud.maxAutoLayers;
    public static float PoissonRadiusScale => SimulationParamSource.Current.pointCloud.poissonRadiusScale;
    public static int PoissonK => SimulationParamSource.Current.pointCloud.poissonK;

    // Neighbors
    public static int NeighborCount = 16;
    public static int CollisionTransferManifoldSlots = 5;
    public static float LayerKernelHFromPoissonRadius => SimulationParamSource.Current.neighbors.layerKernelHFromPoissonRadius;
    public static float WendlandSupport => SimulationParamSource.Current.neighbors.wendlandSupport;

    // XPBI Solver
    public static int JRIterationsLMax => SimulationParamSource.Current.iterations.jrIterationsLMax;
    public static int JRIterationsLMid => SimulationParamSource.Current.iterations.jrIterationsLMid;
    public static int GSIterationsL0 => SimulationParamSource.Current.iterations.gsIterationsL0;
    public static int JRIterationsL0 => SimulationParamSource.Current.iterations.jrIterationsL0;
    public static float JROmegaV => SimulationParamSource.Current.iterations.jrOmegaV;
    public static float JROmegaL => SimulationParamSource.Current.iterations.jrOmegaL;
    public static int PersistentCoarseMaxNodes = 256;

    // DT maintenance
    public static int ColoringConflictRounds => SimulationParamSource.Current.dtMaintenance.coloringConflictRounds;
    public static int InitColoringConflictRounds => SimulationParamSource.Current.dtMaintenance.initColoringConflictRounds;
    public static int DTFixIterations => SimulationParamSource.Current.dtMaintenance.dtFixIterations;
    public static int DTLegalizeIterations => SimulationParamSource.Current.dtMaintenance.dtLegalizeIterations;

    // Stability limits
    public static float MaxVelocity => SimulationParamSource.Current.stability.maxVelocity;
    public static float MaxDisplacementPerTick => SimulationParamSource.Current.stability.maxDisplacementPerTick;

    // Multigrid prolongation
    public static float ProlongationScale => SimulationParamSource.Current.prolongation.prolongationScale;
    public static float RestrictedDeltaVScale => SimulationParamSource.Current.prolongation.restrictedDeltaVScale;
    public static bool UseAffineProlongation => SimulationParamSource.Current.prolongation.useAffineProlongation;
    public static float RestrictResidualDeltaVScale => SimulationParamSource.Current.prolongation.restrictResidualDeltaVScale;
    public static float PostProlongSmoothing => SimulationParamSource.Current.prolongation.postProlongSmoothing;
    public static int ParentKNearest => 2;
    public static float ParentRelationMaxSupportScale => 2f;
    public static float ParentWeightEpsilon => SimulationParamSource.Current.prolongation.parentWeightEpsilon;

    // XPBI particle regularization
    public static float XsphC => SimulationParamSource.Current.particleRegularization.xsphC;
    public static bool EnablePositionCorrection => SimulationParamSource.Current.particleRegularization.enablePositionCorrection;
    public static int PositionCorrectionIterations => SimulationParamSource.Current.particleRegularization.positionCorrectionIterations;
    public static float PositionCorrectionGapRatio => SimulationParamSource.Current.particleRegularization.positionCorrectionGapRatio;
    public static float PositionCorrectionCompliance => SimulationParamSource.Current.particleRegularization.positionCorrectionCompliance;
    public static float PositionCorrectionMaxFraction => SimulationParamSource.Current.particleRegularization.positionCorrectionMaxFraction;

    // Collisions
    public static float CollisionSupportScale => SimulationParamSource.Current.collision.collisionSupportScale;
    public static float CollisionSkinScale => SimulationParamSource.Current.collision.collisionSkinScale;
    public static float CollisionCompliance => SimulationParamSource.Current.collision.collisionCompliance;
    public static float CollisionSlop => SimulationParamSource.Current.collision.collisionSlop;
    public static float CollisionPenBias => SimulationParamSource.Current.collision.collisionPenBias;
    public static float CollisionMaxBias => SimulationParamSource.Current.collision.collisionMaxBias;
    public static float CollisionMaxPush => SimulationParamSource.Current.collision.collisionMaxPush;
    public static float CollisionRelaxation => SimulationParamSource.Current.collision.collisionRelaxation;
    public static float CollisionMaxDv => SimulationParamSource.Current.collision.collisionMaxDv;
    public static float CollisionFriction => SimulationParamSource.Current.collision.collisionFriction;
    public static float CollisionRestitution => SimulationParamSource.Current.collision.collisionRestitution;
    public static float CollisionRestitutionThreshold => SimulationParamSource.Current.collision.collisionRestitutionThreshold;
    public static float CollisionSdfBandHalfWidthScale => SimulationParamSource.Current.collision.collisionSdfBandHalfWidthScale;
    public static float CollisionSdfVertexMarginScale => SimulationParamSource.Current.collision.collisionSdfVertexMarginScale;
    public static float CollisionSdfFallbackDepthScale => SimulationParamSource.Current.collision.collisionSdfFallbackDepthScale;
    public static int CollisionSdfEdgeRefineIterations => SimulationParamSource.Current.collision.collisionSdfEdgeRefineIterations;

    // Fracture and damage model
    public static float CohesiveDamping => SimulationParamSource.Current.fractureDamage.cohesiveDamping;
    public static float CohesiveOnsetRatio => SimulationParamSource.Current.fractureDamage.cohesiveOnsetRatio;
    public static float CohesivePeakRatio => SimulationParamSource.Current.fractureDamage.cohesivePeakRatio;
    public static float DamageOnset => SimulationParamSource.Current.fractureDamage.damageOnset;
    public static float DamageSoftening => SimulationParamSource.Current.fractureDamage.damageSoftening;
    public static float DamageResidualStiffness => SimulationParamSource.Current.fractureDamage.damageResidualStiffness;
    public static float DamageEnergyWeight => SimulationParamSource.Current.fractureDamage.damageEnergyWeight;
    public static float DamageShellWeight => SimulationParamSource.Current.fractureDamage.damageShellWeight;
    public static float DamageMax => SimulationParamSource.Current.fractureDamage.damageMax;
    public static float CohesivePairScale => SimulationParamSource.Current.fractureDamage.cohesivePairScale;

    // Optional deep-debug probe; when enabled it forces per-batch fence waits + readback.
    public static bool ProlongationConstraintDebugEnabled => SimulationParamSource.Current.uiAndReadback.prolongationConstraintDebugEnabled;

}
