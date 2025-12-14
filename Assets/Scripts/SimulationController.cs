using System.Collections.Generic;
using UnityEngine;

[DefaultExecutionOrder(-1000)]
public sealed class MeshlessSimulationController : MonoBehaviour {
    public static MeshlessSimulationController Instance { get; private set; }

    [Header("Rate")]
    [Min(1f)] public float targetTPS = 1000f;
    public float simulationSpeed = 1f;

    [Header("Mode")]
    [Tooltip("When enabled, simulation advances only via the T key (tap = 1 tick, hold = continuous).")]
    public bool manual = true;

    [Header("Catch-up")]
    [Min(1)] public int maxTicksPerFrame = 8;

    const float holdThreshold = 0.2f;
    public bool useUnscaledTime = false;

    [Header("Debug")]
    [Tooltip("If enabled, never execute more than 1 tick per rendered frame (TPS becomes FPS-limited).")]
    public bool forceTPSToFPS = false;

    readonly List<Meshless> meshless = new List<Meshless>(64);

    float accumulator;
    float keyHeldTime;

    void Awake() {
        if (Instance != null && Instance != this) { Destroy(gameObject); return; }
        Instance = this;
    }

    void OnDestroy() {
        if (Instance == this) Instance = null;
    }

    public void Register(Meshless m) {
        if (m != null && !meshless.Contains(m)) meshless.Add(m);
    }

    public void Unregister(Meshless m) {
        if (m != null) meshless.Remove(m);
    }

    void Update() {
        float frameDt = useUnscaledTime ? Time.unscaledDeltaTime : Time.deltaTime;
        if (targetTPS <= 0f || frameDt <= 0f) return;

        if (manual) ManualUpdate(frameDt);
        else AutoUpdate(frameDt);
    }

    void AutoUpdate(float frameDt) {
        float dt = 1f / targetTPS;
        accumulator += frameDt;

        int ticks = 0;
        while (accumulator >= dt) {
            Tick(dt);
            accumulator -= dt;

            if (++ticks >= maxTicksPerFrame) break;
            if (forceTPSToFPS) break;
        }
    }

    void ManualUpdate(float frameDt) {
        if (Input.GetKeyDown(KeyCode.T)) {
            keyHeldTime = 0f;
            accumulator = 0f;
        }

        if (Input.GetKey(KeyCode.T)) {
            keyHeldTime += frameDt;
            if (keyHeldTime >= holdThreshold) AutoUpdate(frameDt);
        }

        if (Input.GetKeyUp(KeyCode.T)) {
            if (keyHeldTime < holdThreshold) Tick(1f / targetTPS);
            keyHeldTime = 0f;
            accumulator = 0f;
        }
    }

    void Tick(float dt) {
        for (int i = meshless.Count - 1; i >= 0; i--) {
            var m = meshless[i];
            if (m == null || !m.isActiveAndEnabled) { meshless.RemoveAt(i); continue; }
            m.StepSimulation(dt * simulationSpeed);
        }
    }
}
