using UnityEngine;
using UnityEngine.Serialization;

[CreateAssetMenu(fileName = "SimulationPreset", menuName = "Shattered/Simulation Preset", order = 10)]
public sealed class SimulationPresetAsset : ScriptableObject {
    [Header("Iterations")]
    [Min(1)] public int gsIterationsL0 = 4;
    [Min(1)] public int jrIterationsL0 = 16;
    [Min(1)] public int jrIterationsLMid = 4;
    [Min(1)] public int jrIterationsLMax = 2;

    [Header("Solver Mode")]
    public bool useHierarchicalSolver = true;
    public SimulationController.NeighborSearchMode neighborSearchMode = SimulationController.NeighborSearchMode.DelaunayTriangulation;

    [Header("Compliance")]
    [Min(0f)] public float compliance = 1f;
    [Min(0f)] public float collisionCompliance = 0.1f;
    [Min(0f)] public float durabilityCompliance = 0.8f;

    [Header("Substeps")]
    [FormerlySerializedAs("substeps")]
    [Min(1)] public int sbsteps = 1;

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
        target.durability.durabilityCompliance = Mathf.Max(0f, durabilityCompliance);

        target.runtime.solverSubsteps = Mathf.Max(1, sbsteps);
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
        durabilityCompliance = Mathf.Max(0f, source.durability.durabilityCompliance);

        sbsteps = Mathf.Max(1, source.runtime.solverSubsteps);
    }
}
