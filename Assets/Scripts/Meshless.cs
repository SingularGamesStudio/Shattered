using System.Collections.Generic;
using System.Linq;
using Unity.Mathematics;
using UnityEngine;
using GPU.Delaunay;

public class Meshless : MonoBehaviour {
    public static readonly List<Meshless> Active = new List<Meshless>(64);
    public List<Node> nodes = new List<Node>();
    public int maxLayer = 3;
    [Header("Simulation parameters")]
    public float gravity = -9.81f;
    [Header("XPBI compliance")]
    public float compliance = 0f;
    public int[] levelEndIndex;
    public float dtNormalizePadding = 2f;
    public bool dtAutoNormalizeIncludeCamera = true;
    [HideInInspector] public DTHierarchy delaunayHierarchy;

    float2 dtNormCenter;
    float dtNormInvHalfExtent;
    float2 dtBoundsMinWorld;
    float2 dtBoundsMaxWorld;
    public float[] layerRadii;

    float2 dtSuper0 = new float2(0f, 3f);
    float2 dtSuper1 = new float2(-3f, -3f);
    float2 dtSuper2 = new float2(3f, -3f);

    [Header("Material")]
    public MaterialDef baseMaterialDef;

    public float2 DtNormCenter => dtNormCenter;
    public float DtNormInvHalfExtent => dtNormInvHalfExtent;

    public int GetBaseMaterialId() {
        var lib = MaterialLibrary.Instance;
        return lib == null ? 0 : lib.GetMaterialIndex(baseMaterialDef);
    }

    public bool TryGetLevelDt(int level, out DT dt) {
        dt = delaunayHierarchy?.GetLevelDt(level);
        return dt != null;
    }

    public void FixNode(int nodeIdx) {
        nodes[nodeIdx].isFixed = true;
        nodes[nodeIdx].invMass = 0.0f;
        nodes[nodeIdx].vel = float2.zero;
    }

    public int Add(float2 pos) {
        nodes.Add(new Node(pos, this));
        return nodes.Count - 1;
    }

    public void Build() {
        nodes = nodes.OrderByDescending(node => node.maxLayer).ToList();

        for (int i = 0; i < nodes.Count; i++)
            nodes[i].parentIndex = -1;

        BuildLevelEndIndex();

        RecomputeDelaunayNormalizationBounds(dtAutoNormalizeIncludeCamera ? Camera.main : null);

        BuildDelaunayHierarchy();

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

    void RecomputeDelaunayNormalizationBounds(Camera cam) {
        if (nodes.Count == 0) return;
        float2 min = nodes[0].pos, max = nodes[0].pos;
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
        ComputeSuperTriangle(dtBoundsMinWorld, dtBoundsMaxWorld, 2f, out dtSuper0, out dtSuper1, out dtSuper2);
        delaunayHierarchy?.Dispose();
        delaunayHierarchy = new DTHierarchy(SimulationController.Instance.delaunayShader);
        delaunayHierarchy.InitFromMeshlessNodes(
            nodes,
            levelEndIndex,
            maxLayer,
            dtNormCenter,
            dtNormInvHalfExtent,
            dtSuper0,
            dtSuper1,
            dtSuper2,
            Const.NeighborCount
        );
    }

    static void ComputeSuperTriangle(float2 min, float2 max, float scale, out float2 p0, out float2 p1, out float2 p2) {
        float2 center = 0.5f * (min + max);
        float2 extent = max - min;
        float d = math.max(extent.x, extent.y);
        float s = math.max(1f, scale) * math.max(1e-6f, d);
        p0 = center + new float2(0f, 2f * s);
        p1 = center + new float2(-2f * s, -2f * s);
        p2 = center + new float2(2f * s, -2f * s);
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