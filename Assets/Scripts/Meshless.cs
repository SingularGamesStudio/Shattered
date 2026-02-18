using System.Collections.Generic;
using System.Linq;
using Unity.Mathematics;
using UnityEngine;
using GPU.Delaunay;

public class Meshless : MonoBehaviour {
    public static readonly List<Meshless> Active = new List<Meshless>(64);

    public List<Node> nodes = new List<Node>();

    [HideInInspector]
    public int maxLayer = -1;

    [Header("Simulation parameters")]
    public float gravity = -9.81f;

    [Header("XPBI compliance")]
    public float compliance = 0f;

    public int[] levelEndIndex;

    [Header("GPU Delaunay hierarchy")]
    public ComputeShader delaunayShader;
    public int dtFixIterationsPerTick = 1;
    public int dtLegalizeIterationsPerTick = 1;
    public int dtWarmupFixIterations = 64;
    public int dtWarmupLegalizeIterations = 128;
    public float dtNormalizePadding = 2f;

    [Header("DT normalization (runtime)")]
    public bool dtAutoNormalizeAtRuntime = true;
    [Range(1.0f, 2.0f)] public float dtAutoNormalizeGrowFactor = 1.15f;
    [Range(0.0f, 1.0f)] public float dtAutoNormalizeRecenterThreshold = 0.25f;
    public bool dtAutoNormalizeIncludeCamera = true;

    [HideInInspector] public DTHierarchy delaunayHierarchy;

    float2 dtNormCenter;
    float dtNormInvHalfExtent;
    float2 dtBoundsMinWorld;
    float2 dtBoundsMaxWorld;

    readonly float2 dtSuper0 = new float2(0f, 3f);
    readonly float2 dtSuper1 = new float2(-3f, -3f);
    readonly float2 dtSuper2 = new float2(3f, -3f);

    [Header("Material")]
    public MaterialDef baseMaterialDef;

    public float2 DtNormCenter => dtNormCenter;
    public float DtNormInvHalfExtent => dtNormInvHalfExtent;

    public int GetBaseMaterialId() {
        var lib = MaterialLibrary.Instance;
        if (lib == null) return 0;
        return lib.GetMaterialIndex(baseMaterialDef);
    }

    public bool TryGetLevelDt(int level, out DT dt) {
        if (delaunayHierarchy == null) { dt = null; return false; }
        dt = delaunayHierarchy.GetLevelDt(level);
        return dt != null;
    }

    public void FixNode(int nodeIdx) {
        nodes[nodeIdx].isFixed = true;
        nodes[nodeIdx].invMass = 0.0f;
        nodes[nodeIdx].vel = float2.zero;
    }

    public void Add(float2 pos) {
        Node newNode = new Node(pos, this);
        if (newNode.maxLayer > maxLayer) {
            maxLayer = newNode.maxLayer;
        }
        nodes.Add(newNode);
    }

    public void Build() {
        nodes = nodes.OrderByDescending(node => node.maxLayer).ToList();

        maxLayer = -1;
        for (int i = 0; i < nodes.Count; i++) {
            maxLayer = math.max(maxLayer, nodes[i].maxLayer);
            nodes[i].parentIndex = -1;
        }

        BuildLevelEndIndex();

        if (!delaunayShader) throw new System.InvalidOperationException("Meshless: delaunayShader is not assigned.");

        RecomputeDelaunayNormalizationBounds(dtAutoNormalizeIncludeCamera ? Camera.main : null);
        BuildDelaunayHierarchy();

        ComputeRestVolumesFromDelaunayTriangles();
    }

    void BuildLevelEndIndex() {
        if (maxLayer < 0) { levelEndIndex = null; return; }

        levelEndIndex = new int[maxLayer + 1];
        int idx = 0;
        for (int level = maxLayer; level >= 0; level--) {
            for (; idx < nodes.Count && nodes[idx].maxLayer >= level; idx++) { }
            levelEndIndex[level] = idx;
        }
    }

    public int NodeCount(int level) {
        if (levelEndIndex == null || level < 0 || level > maxLayer) return 0;
        return levelEndIndex[level];
    }

    public void UpdateDelaunayAfterIntegration() {
        UpdateDelaunayAfterIntegration(readback: false);
    }

    public void UpdateDelaunayAfterIntegration(bool readback) {
        if (dtAutoNormalizeAtRuntime) {
            UpdateDelaunayNormalizationIfNeeded(dtAutoNormalizeIncludeCamera ? Camera.main : null);
        }

        delaunayHierarchy.UpdatePositionsFromNodesAllLevels(nodes, dtNormCenter, dtNormInvHalfExtent);
        delaunayHierarchy.MaintainAllLevels(dtFixIterationsPerTick, dtLegalizeIterationsPerTick);
    }

    bool UpdateDelaunayNormalizationIfNeeded(Camera cam) {
        if (nodes.Count == 0) return false;

        float2 min = nodes[0].pos;
        float2 max = nodes[0].pos;

        for (int i = 1; i < nodes.Count; i++) {
            float2 p = nodes[i].pos;
            min = math.min(min, p);
            max = math.max(max, p);
        }

        if (cam != null && cam.orthographic) {
            float halfH = cam.orthographicSize;
            float halfW = halfH * cam.aspect;

            float2 c = new float2(cam.transform.position.x, cam.transform.position.y);
            float2 camMin = c - new float2(halfW, halfH);
            float2 camMax = c + new float2(halfW, halfH);

            min = math.min(min, camMin);
            max = math.max(max, camMax);
        }

        float2 targetMin = min - new float2(dtNormalizePadding, dtNormalizePadding);
        float2 targetMax = max + new float2(dtNormalizePadding, dtNormalizePadding);

        float2 targetCenter = 0.5f * (targetMin + targetMax);
        float2 extent = targetMax - targetMin;
        float targetHalf = 0.5f * math.max(extent.x, extent.y);
        targetHalf = math.max(targetHalf, 1e-6f);

        float currentHalf = 1f / math.max(1e-6f, dtNormInvHalfExtent);

        bool needGrow = targetHalf > currentHalf;
        bool needRecenter = math.any(math.abs(targetCenter - dtNormCenter) > currentHalf * dtAutoNormalizeRecenterThreshold);

        if (!needGrow && !needRecenter) return false;

        float newHalf = needGrow ? targetHalf * dtAutoNormalizeGrowFactor : currentHalf;
        dtNormCenter = targetCenter;
        dtNormInvHalfExtent = 1f / newHalf;

        dtBoundsMinWorld = dtNormCenter - new float2(newHalf, newHalf);
        dtBoundsMaxWorld = dtNormCenter + new float2(newHalf, newHalf);

        return true;
    }

    void RecomputeDelaunayNormalizationBounds(Camera cam) {
        if (nodes.Count == 0) return;

        float2 min = nodes[0].pos;
        float2 max = nodes[0].pos;

        for (int i = 1; i < nodes.Count; i++) {
            float2 p = nodes[i].pos;
            min = math.min(min, p);
            max = math.max(max, p);
        }

        if (cam != null && cam.orthographic) {
            float halfH = cam.orthographicSize;
            float halfW = halfH * cam.aspect;

            float2 c = new float2(cam.transform.position.x, cam.transform.position.y);
            float2 camMin = c - new float2(halfW, halfH);
            float2 camMax = c + new float2(halfW, halfH);

            min = math.min(min, camMin);
            max = math.max(max, camMax);
        }

        dtBoundsMinWorld = min - new float2(dtNormalizePadding, dtNormalizePadding);
        dtBoundsMaxWorld = max + new float2(dtNormalizePadding, dtNormalizePadding);

        dtNormCenter = 0.5f * (dtBoundsMinWorld + dtBoundsMaxWorld);

        float2 extent = dtBoundsMaxWorld - dtBoundsMinWorld;
        float half = 0.5f * math.max(extent.x, extent.y);
        dtNormInvHalfExtent = 1f / math.max(1e-6f, half);
    }

    void BuildDelaunayHierarchy() {
        delaunayHierarchy?.Dispose();
        delaunayHierarchy = null;

        delaunayHierarchy = new DTHierarchy(delaunayShader);

        delaunayHierarchy.InitFromMeshlessNodes(
            nodes,
            dtNormCenter,
            dtNormInvHalfExtent,
            dtSuper0,
            dtSuper1,
            dtSuper2,
            Const.NeighborCount,
            dtWarmupFixIterations,
            dtWarmupLegalizeIterations
        );
    }

    void ComputeRestVolumesFromDelaunayTriangles() {
        int n = nodes.Count;
        if (n < 3) return;

        var points = new float2[n];
        for (int i = 0; i < n; i++) points[i] = nodes[i].pos;

        DTBuilder.BuildBowyerWatsonWithSuper(
            points,
            dtBoundsMinWorld,
            dtBoundsMaxWorld,
            2f,
            out float2[] allPoints,
            out List<DTBuilder.Triangle> tris,
            out int realCount
        );

        var vol = new float[n];
        for (int i = 0; i < vol.Length; i++) vol[i] = 0f;

        for (int ti = 0; ti < tris.Count; ti++) {
            var t = tris[ti];

            if ((uint)t.a >= (uint)realCount || (uint)t.b >= (uint)realCount || (uint)t.c >= (uint)realCount)
                continue;

            float2 a = allPoints[t.a];
            float2 b = allPoints[t.b];
            float2 c = allPoints[t.c];

            float area = 0.5f * math.abs((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
            float share = area / 3f;

            vol[t.a] += share;
            vol[t.b] += share;
            vol[t.c] += share;
        }

        for (int i = 0; i < n; i++)
            nodes[i].restVolume = vol[i];
    }

    void OnEnable() {
        SimulationController.Instance?.Register(this);
        if (!Active.Contains(this)) Active.Add(this);
    }

    void OnDisable() {
        SimulationController.Instance?.Unregister(this);
        Active.Remove(this);

        delaunayHierarchy?.Dispose();
        delaunayHierarchy = null;
    }
    void OnDestroy() {
        delaunayHierarchy?.Dispose();
        delaunayHierarchy = null;
    }
}
