using System.Collections.Generic;
using GPU.Solver;
using Unity.Mathematics;
using UnityEngine;

public sealed class ReadbackMultiForceController : MeshlessForceControllerBase {
    public Meshless targetMeshless;

    [Header("Regional Pull Forces")]
    public float pullStrength = 0.03f;
    public float cornerBoost = 0.02f;
    public float oscillationAmplitude = 0.35f;
    public float oscillationFrequency = 1.0f;

    public override void GatherForceEvents(Meshless target, float dtPerTick, int ticksToRun, List<XPBISolver.ForceEvent> eventsOut) {
        if (target == null || eventsOut == null)
            return;

        if (targetMeshless != null && targetMeshless != target)
            return;

        if (target.nodes == null || target.nodes.Count == 0)
            return;

        float time = Time.time;
        float osc = 1f + oscillationAmplitude * math.sin(time * oscillationFrequency);
        float baseStrength = pullStrength * osc;

        float2 min = new float2(float.MaxValue, float.MaxValue);
        float2 max = new float2(float.MinValue, float.MinValue);

        for (int i = 0; i < target.nodes.Count; i++) {
            float2 p = target.nodes[i].pos;
            min = math.min(min, p);
            max = math.max(max, p);
        }

        float2 center = 0.5f * (min + max);
        float2 size = math.max(max - min, new float2(1e-4f, 1e-4f));

        for (int i = 0; i < target.nodes.Count; i++) {
            var node = target.nodes[i];
            if (node.invMass <= 0f)
                continue;

            float2 p = target.nodes[i].pos;
            float2 rel = (p - center) / size;

            float2 force = 0f;

            if (rel.x <= -0.15f) force += new float2(-1f, 0f) * baseStrength;
            if (rel.x >= 0.15f) force += new float2(1f, 0f) * baseStrength;
            if (rel.y <= -0.15f) force += new float2(0f, -1f) * baseStrength;
            if (rel.y >= 0.15f) force += new float2(0f, 1f) * baseStrength;

            float corner = math.saturate(math.length(rel) * 1.4142f);
            if (corner > 0.65f) {
                float2 radial = math.normalizesafe(rel, 0f);
                force += radial * cornerBoost;
            }

            if (math.lengthsq(force) <= 1e-12f)
                continue;

            eventsOut.Add(new XPBISolver.ForceEvent {
                node = (uint)i,
                force = force,
            });
        }
    }
}
