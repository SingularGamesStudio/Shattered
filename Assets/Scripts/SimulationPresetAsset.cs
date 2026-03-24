using UnityEngine;

[CreateAssetMenu(fileName = "SimulationPreset", menuName = "Shattered/Simulation Preset", order = 10)]
public sealed class SimulationPresetAsset : ScriptableObject {
    [Header("Iterations")]
     public int gsIterationsL0 = 4;
     public int jrIterationsL0 = 16;
     public int jrIterationsLMid = 4;
     public int jrIterationsLMax = 2;

    [Header("Solver Mode")]
    public bool useHierarchicalSolver = true;
    public SimulationController.NeighborSearchMode neighborSearchMode = SimulationController.NeighborSearchMode.DelaunayTriangulation;

    [Header("Compliance")]
    [Min(0f)] public float compliance = 1f;
    [Min(0f)] public float collisionCompliance = 0.1f;
    [Min(0f)] public float durabilityCompliance = 0.8f;
    [Min(0f)] public float positionCorrectionCompliance = 0f;

    [Header("Collision SDF")]
    [Min(0.5f)] public float collisionSdfBandHalfWidthScale = 4f;
    [Min(0.5f)] public float collisionSdfVertexMarginScale = 1.5f;
    [Min(0.5f)] public float collisionSdfFallbackDepthScale = 2.5f;
    [Range(1, 12)] public int collisionSdfEdgeRefineIterations = 6;

    public void ApplyTo(SimulationParams target) {
        if (target == null)
            return;

        target.iterations.gsIterationsL0 = Mathf.Max(1, gsIterationsL0);
        target.iterations.jrIterationsL0 = Mathf.Max(1, jrIterationsL0);
        target.iterations.jrIterationsLMid = Mathf.Max(1, jrIterationsLMid);
        target.iterations.jrIterationsLMax = Mathf.Max(1, jrIterationsLMax);

        target.interaction.useHierarchicalSolver = useHierarchicalSolver;
        target.interaction.neighborSearchMode = neighborSearchMode;

        target.solverCore.compliance = Mathf.Max(0f, compliance);
        target.collision.collisionCompliance = Mathf.Max(0f, collisionCompliance);
        target.collision.collisionSdfBandHalfWidthScale = Mathf.Max(0.5f, collisionSdfBandHalfWidthScale);
        target.collision.collisionSdfVertexMarginScale = Mathf.Max(0.5f, collisionSdfVertexMarginScale);
        target.collision.collisionSdfFallbackDepthScale = Mathf.Max(0.5f, collisionSdfFallbackDepthScale);
        target.collision.collisionSdfEdgeRefineIterations = Mathf.Clamp(collisionSdfEdgeRefineIterations, 1, 12);
        target.durability.durabilityCompliance = Mathf.Max(0f, durabilityCompliance);
        target.particleRegularization.positionCorrectionCompliance = Mathf.Max(0f, positionCorrectionCompliance);
    }

    public void CaptureFrom(SimulationParams source) {
        if (source == null)
            return;

        gsIterationsL0 = Mathf.Max(1, source.iterations.gsIterationsL0);
        jrIterationsL0 = Mathf.Max(1, source.iterations.jrIterationsL0);
        jrIterationsLMid = Mathf.Max(1, source.iterations.jrIterationsLMid);
        jrIterationsLMax = Mathf.Max(1, source.iterations.jrIterationsLMax);

        useHierarchicalSolver = source.interaction.useHierarchicalSolver;
        neighborSearchMode = source.interaction.neighborSearchMode;

        compliance = Mathf.Max(0f, source.solverCore.compliance);
        collisionCompliance = Mathf.Max(0f, source.collision.collisionCompliance);
        collisionSdfBandHalfWidthScale = Mathf.Max(0.5f, source.collision.collisionSdfBandHalfWidthScale);
        collisionSdfVertexMarginScale = Mathf.Max(0.5f, source.collision.collisionSdfVertexMarginScale);
        collisionSdfFallbackDepthScale = Mathf.Max(0.5f, source.collision.collisionSdfFallbackDepthScale);
        collisionSdfEdgeRefineIterations = Mathf.Clamp(source.collision.collisionSdfEdgeRefineIterations, 1, 12);
        durabilityCompliance = Mathf.Max(0f, source.durability.durabilityCompliance);
        positionCorrectionCompliance = Mathf.Max(0f, source.particleRegularization.positionCorrectionCompliance);
    }
}
