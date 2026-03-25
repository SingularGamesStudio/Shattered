using System;
using UnityEngine;

[Serializable]
public sealed class SimulationParams {
    [InspectorName("01. Point Cloud")]
    [SerializeField] public PointCloudParams pointCloud = new PointCloudParams();

    [InspectorName("02. Runtime Pace")]
    [SerializeField] public RuntimeParams runtime = new RuntimeParams();

    [InspectorName("03. Interaction and Solver Mode")]
    [SerializeField] public InteractionParams interaction = new InteractionParams();

    [InspectorName("04. Solver Core")]
    [SerializeField] public SolverCoreParams solverCore = new SolverCoreParams();

    [InspectorName("05. Neighborhood")]
    [SerializeField] public NeighborParams neighbors = new NeighborParams();

    [InspectorName("06. Iterations")]
    [SerializeField] public IterationParams iterations = new IterationParams();

    [InspectorName("07. Stability")]
    [SerializeField] public StabilityParams stability = new StabilityParams();

    [InspectorName("08. Prolongation")]
    [SerializeField] public ProlongationParams prolongation = new ProlongationParams();

    [InspectorName("09. Particle Regularization")]
    [SerializeField] public ParticleRegularizationParams particleRegularization = new ParticleRegularizationParams();

    [InspectorName("10. Collisions")]
    [SerializeField] public CollisionParams collision = new CollisionParams();

    [InspectorName("11. Durability")]
    [SerializeField] public DurabilityParams durability = new DurabilityParams();

    [InspectorName("12. DT Maintenance")]
    [SerializeField] public DtMaintenanceParams dtMaintenance = new DtMaintenanceParams();

    [InspectorName("13. UI and CPU Readback")]
    [SerializeField] public UiAndReadbackParams uiAndReadback = new UiAndReadbackParams();

    [Serializable]
    public sealed class PointCloudParams {
        [Min(1f)] public float layer0PointDensity = 256f;
        [Range(0.05f, 1f)] public float layerDownsampleRatio = 0.25f;
        [Min(1)] public int minVerticesPerLayer = 10;
        [Min(1)] public int maxAutoLayers = 6;
        [Range(0.1f, 2f)] public float poissonRadiusScale = 0.85f;
        [Min(1)] public int poissonK = 30;
    }

    [Serializable]
    public sealed class SolverCoreParams {
        public float gravity = -0.01f;
        [Min(0f)] public float compliance = 1f;
    }

    [Serializable]
    public sealed class NeighborParams {
        [Min(0.01f)] public float layerKernelHFromPoissonRadius = 2f;
        [Min(0.01f)] public float wendlandSupport = 2f;
    }

    [Serializable]
    public sealed class IterationParams {
        [Min(1)] public int jrIterationsLMax = 2;
        [Min(1)] public int jrIterationsLMid = 4;
        [Min(1)] public int gsIterationsL0 = 4;
        [Min(1)] public int jrIterationsL0 = 16;
        [Range(0f, 1f)] public float jrOmegaV = 0.3f;
        [Range(0f, 1f)] public float jrOmegaL = 0.3f;
    }

    [Serializable]
    public sealed class DtMaintenanceParams {
        [Min(0)] public int coloringConflictRounds = 3;
        [Min(0)] public int initColoringConflictRounds = 24;
        [Min(0)] public int dtFixIterations = 2;
        [Min(0)] public int dtLegalizeIterations = 2;
    }

    [Serializable]
    public sealed class StabilityParams {
        [Min(0f)] public float maxVelocity = 2f;
        [Min(0f)] public float maxDisplacementPerTick = 0.012f;
    }

    [Serializable]
    public sealed class ProlongationParams {
        [Range(0f, 2f)] public float prolongationScale = 0.5f;
        [Range(0f, 2f)] public float restrictedDeltaVScale = 0.75f;
        public bool useAffineProlongation = true;
        [Range(0f, 2f)] public float restrictResidualDeltaVScale = 0.35f;
        [Range(0f, 1f)] public float postProlongSmoothing = 0.2f;
        [Min(1e-6f)] public float parentWeightEpsilon = 1e-3f;
    }

    [Serializable]
    public sealed class ParticleRegularizationParams {
        [Tooltip("XSPH artificial viscosity blending coefficient (XPBI Eq. 20).")]
        [Range(0f, 0.2f)] public float xsphC = 0.01f;

        [Tooltip("Enable pairwise minimum-distance position correction (XPBI Eq. 21 style).")]
        public bool enablePositionCorrection = true;

        [Tooltip("Additional correction passes per layer solve.")]
        [Min(0)] public int positionCorrectionIterations = 1;

        [Tooltip("Gap ratio epsilon/r used by the minimum-distance inequality. 0.25 matches the paper.")]
        [Range(0f, 1f)] public float positionCorrectionGapRatio = 0.25f;

        [Tooltip("XPBD compliance for the pairwise position correction; 0 means hard correction.")]
        [Min(0f)] public float positionCorrectionCompliance = 0f;

        [Tooltip("Per-pass correction clamp as a fraction of the target minimum distance.")]
        [Range(0f, 1f)] public float positionCorrectionMaxFraction = 0.25f;
    }

    [Serializable]
    public sealed class CollisionParams {
        [Range(0f, 2f)] public float collisionSupportScale = 0.3f;
        [Range(0f, 2f)] public float collisionSkinScale = 0.02f;
        [Min(0f)] public float collisionCompliance = 0.1f;
        [Range(0f, 0.2f)] public float collisionSlop = 0.01f;
        [Range(0f, 4f)] public float collisionPenBias = 0.35f;
        [Min(0f)] public float collisionMaxBias = 0.005f;
        [Min(0f)] public float collisionMaxPush = 0.02f;
        [Range(0f, 1f)] public float collisionRelaxation = 0.5f;
        [Min(0f)] public float collisionMaxDv = 1.5f;
        [Range(0f, 1f)] public float collisionFriction = 0.2f;
        [Range(0f, 1f)] public float collisionRestitution = 0f;
        [Min(0f)] public float collisionRestitutionThreshold = 0.005f;
        [Min(0.5f)] public float collisionSdfBandHalfWidthScale = 4f;
        [Min(0.5f)] public float collisionSdfVertexMarginScale = 1.5f;
        [Min(0.5f)] public float collisionSdfFallbackDepthScale = 2.5f;
        [Range(1, 12)] public int collisionSdfEdgeRefineIterations = 6;
    }

    [Serializable]
    public sealed class DurabilityParams {
        [Min(0f)] public float durabilityCompliance = 0.8f;
        [Range(0f, 1f)] public float durabilityMaxDistanceRatio = 0.8f;
    }

    [Serializable]
    public sealed class RuntimeParams {
        [Tooltip("[HIGH IMPACT] Main simulation cadence.")]
        [Min(1f)] public float targetTPS = 1000f;
        [Min(1f)] public float targetFPS = 60f;
        [Min(0f)] public float simulationSpeed = 1f;
        [Min(1)] public int maxTicksPerBatch = 32;
    }

    [Serializable]
    public sealed class InteractionParams {
        [Tooltip("When enabled, simulation advances only via the T key (tap = 1 tick, hold = continuous).")]
        public bool manual = true;
        [Min(0f)] public float holdThreshold = 0.2f;
        public bool useHierarchicalSolver = true;
        public SimulationController.NeighborSearchMode neighborSearchMode = SimulationController.NeighborSearchMode.DelaunayTriangulation;
    }

    [Serializable]
    public sealed class UiAndReadbackParams {
        public bool showTpsOverlay = true;
        public bool convergenceDebugEnabled = true;
        public bool enableContinuousCpuReadback = true;
        [Min(0.001f)] public float cpuReadbackInterval = 0.02f;
        [Tooltip("Forces per-batch fence waits plus readback when enabled.")]
        public bool prolongationConstraintDebugEnabled = false;
    }
}

public static class SimulationParamSource {
    static readonly SimulationParams defaults = new SimulationParams();
    static SimulationParams current;

    public static SimulationParams Current => current ?? defaults;

    public static void Set(SimulationParams parameters) {
        current = parameters;
    }
}
