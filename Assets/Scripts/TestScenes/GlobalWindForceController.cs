using System.Collections.Generic;
using GPU.Solver;
using Unity.Mathematics;
using UnityEngine;

public sealed class GlobalWindForceController : MeshlessForceControllerBase {
    public Meshless targetMeshless;

    [Header("Base")]
    public Vector2 baseDirection = new Vector2(1f, 0f);
    public float baseStrength = 0.02f;

    [Header("Oscillation")]
    public float oscillationStrength = 0.03f;
    public float oscillationFrequency = 1.2f;
    public float spatialPhase = 0.35f;

    [Header("Drift")]
    public Vector2 driftDirection = new Vector2(0.2f, 0.1f);
    public float driftStrength = 0.003f;

    public override void GatherForceEvents(Meshless target, float dtPerTick, int ticksToRun, List<XPBISolver.ForceEvent> eventsOut) {
        if (target == null || eventsOut == null)
            return;

        if (targetMeshless != null && targetMeshless != target)
            return;

        if (target.nodes == null || target.nodes.Count == 0)
            return;

        float2 baseDir = math.normalizesafe(new float2(baseDirection.x, baseDirection.y), new float2(1f, 0f));
        float2 driftDir = math.normalizesafe(new float2(driftDirection.x, driftDirection.y), new float2(0f, 1f));
        float time = Time.time;

        for (int i = 0; i < target.nodes.Count; i++) {
            var node = target.nodes[i];
            if (node.invMass <= 0f)
                continue;

            float phase = time * oscillationFrequency + (node.pos.x + node.pos.y) * spatialPhase;
            float windAmp = baseStrength + oscillationStrength * math.sin(phase);
            float2 force = baseDir * windAmp + driftDir * driftStrength;

            eventsOut.Add(new XPBISolver.ForceEvent {
                node = (uint)i,
                force = force,
            });
        }
    }
}
