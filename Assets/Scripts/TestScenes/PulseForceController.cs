using System.Collections.Generic;
using GPU.Solver;
using Unity.Mathematics;
using UnityEngine;

public sealed class PulseForceController : MeshlessForceControllerBase {
    public Meshless targetMeshless;

    [Header("Pulse")]
    public Vector2 worldCenter;
    public float radius = 1.5f;
    public Vector2 direction = new Vector2(1f, 0.5f);
    public float strength = 12f;
    public float startDelay = 0.5f;
    public float duration = 0.08f;
    public bool repeat;
    public float repeatInterval = 2.5f;

    float nextPulseStart;

    protected override void OnEnable() {
        base.OnEnable();
        nextPulseStart = Time.time + Mathf.Max(0f, startDelay);
    }

    public override void GatherForceEvents(Meshless target, float dtPerTick, int ticksToRun, List<XPBISolver.ForceEvent> eventsOut) {
        if (target == null || eventsOut == null)
            return;

        if (targetMeshless != null && targetMeshless != target)
            return;

        float now = Time.time;
        if (repeat && now > nextPulseStart + Mathf.Max(0.0001f, duration))
            nextPulseStart = now + Mathf.Max(0.01f, repeatInterval);

        float pulseEnd = nextPulseStart + Mathf.Max(0.0001f, duration);
        if (now < nextPulseStart || now > pulseEnd)
            return;

        float2 center = new float2(worldCenter.x, worldCenter.y);
        float radiusSq = math.max(1e-6f, radius * radius);
        float2 dir = math.normalizesafe(new float2(direction.x, direction.y), new float2(1f, 0f));

        for (int i = 0; i < target.nodes.Count; i++) {
            var node = target.nodes[i];
            if (node.invMass <= 0f)
                continue;

            float2 offset = node.pos - center;
            float d2 = math.lengthsq(offset);
            if (d2 > radiusSq)
                continue;

            float falloff = 1f - math.saturate(math.sqrt(d2 / radiusSq));
            float2 force = dir * (strength * falloff);
            eventsOut.Add(new XPBISolver.ForceEvent {
                node = (uint)i,
                force = force,
            });
        }
    }
}
