using System.Collections.Generic;
using System.Linq;
using Unity.Mathematics;
using Unity.Collections;
using UnityEngine;
using GPU.Delaunay;

public class Meshless : MonoBehaviour {
    public static readonly List<Meshless> Active = new List<Meshless>(64);
    public List<Node> nodes = new List<Node>();
    public int maxLayer = 2;
    [HideInInspector]
    public int[] layerEndIndex;
    const float dtNormalizePadding = 2f;
    const bool dtAutoNormalizeIncludeCamera = true;
    [HideInInspector]
    public DTHierarchy delaunayHierarchy;

    float2 dtNormCenter;
    float dtNormInvHalfExtent;
    float2 dtBoundsMinWorld;
    float2 dtBoundsMaxWorld;
    [HideInInspector]
    public float[] layerRadii;
    [HideInInspector]
    public float[] layerKernelH;

    [Header("Material")]
    public MaterialDef baseMaterialDef;

    public float2 DtNormCenter => dtNormCenter;
    public float DtNormInvHalfExtent => dtNormInvHalfExtent;

    public float GetLayerKernelH(int layer) {
        if (layerKernelH != null && layer >= 0 && layer < layerKernelH.Length)
            return layerKernelH[layer];
        if (layerRadii != null && layer >= 0 && layer < layerRadii.Length)
            return layerRadii[layer];
        return 0f;
    }

    public void ApplyCpuReadbackNormalizedPositions(NativeArray<float2> normalizedPositions, int expectedCount) {
        if (nodes == null || expectedCount <= 0 || normalizedPositions.Length <= 0)
            return;

        float inv = DtNormInvHalfExtent;
        if (inv <= 0f)
            return;

        float invInv = 1f / inv;
        float2 center = DtNormCenter;

        int copyCount = math.min(math.min(expectedCount, normalizedPositions.Length), nodes.Count);
        for (int i = 0; i < copyCount; i++) {
            var node = nodes[i];
            node.pos = normalizedPositions[i] * invInv + center;
            nodes[i] = node;
        }
    }

    public int GetBaseMaterialId() {
        var lib = MaterialLibrary.Instance;
        return lib == null ? 0 : lib.GetMaterialIndex(baseMaterialDef);
    }

    public bool TryGetLayerDt(int layer, out DT dt) {
        dt = delaunayHierarchy?.GetLayerDt(layer);
        return dt != null;
    }

    public void FixNode(int nodeIdx) {
        nodes[nodeIdx].isFixed = true;
        nodes[nodeIdx].invMass = -1f;
    }

    public int Add(float2 pos) {
        nodes.Add(new Node(pos, this));
        return nodes.Count - 1;
    }

    public void Build() {
        nodes = nodes.OrderByDescending(node => node.maxLayer).ToList();

        BuildLayerEndIndex();
        if (Const.DebugSupportRadius) LogGenerationSupportNeighborStats();

        RecomputeDelaunayNormalizationBounds(dtAutoNormalizeIncludeCamera ? Camera.main : null);

        BuildDelaunayHierarchy();

        RecomputeMassFromDensity();

    }

    void LogGenerationSupportNeighborStats() {
        if (nodes == null || nodes.Count == 0 || layerEndIndex == null || maxLayer < 0)
            return;

        for (int layer = 0; layer <= maxLayer; layer++) {
            int activeCount = NodeCount(layer);
            if (activeCount <= 1)
                continue;

            float supportRadius = Const.WendlandSupport * GetLayerKernelH(layer);
            if (supportRadius <= 0f)
                continue;

            float supportRadiusSq = supportRadius * supportRadius;
            double totalNeighbors = 0.0;

            for (int i = 0; i < activeCount; i++) {
                float2 pi = nodes[i].pos;
                int localCount = 0;
                for (int j = 0; j < activeCount; j++) {
                    if (i == j)
                        continue;

                    float2 d = nodes[j].pos - pi;
                    if (math.dot(d, d) <= supportRadiusSq)
                        localCount++;
                }

                totalNeighbors += localCount;
            }

            float avgNeighbors = (float)(totalNeighbors / activeCount);
            Debug.LogError($"Meshless '{name}' layer {layer}: avg neighbors within support ({supportRadius:0.####}) = {avgNeighbors:0.##} across {activeCount} vertices.", this);
        }
    }

    public void RecomputeMassFromDensity() {
        var materialLib = MaterialLibrary.Instance;

        for (int i = 0; i < nodes.Count; i++) {
            var node = nodes[i];

            if (node.isFixed || node.invMass <= 0f) {
                node.isFixed = true;
                node.invMass = -1f;
                nodes[i] = node;
                continue;
            }

            float density = materialLib != null ? materialLib.GetDensityByIndex(node.materialId) : 1f;
            float restVolume = math.max(node.restVolume, 1e-6f);
            float mass = math.max(1e-6f, density * restVolume);

            node.invMass = 1f / mass;
            node.isFixed = false;
            nodes[i] = node;
        }
    }

    void BuildLayerEndIndex() {
        if (maxLayer < 0) { layerEndIndex = null; return; }
        layerEndIndex = new int[maxLayer + 1];
        int idx = 0;
        for (int layer = maxLayer; layer >= 0; layer--) {
            for (; idx < nodes.Count && nodes[idx].maxLayer >= layer; idx++) { }
            layerEndIndex[layer] = idx;
        }
    }

    public int NodeCount(int layer) {
        if (layerEndIndex == null || layer < 0 || layer > maxLayer) return 0;
        return layerEndIndex[layer];
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
        float2 dtSuper0, dtSuper1, dtSuper2;
        ComputeSuperTriangle(dtBoundsMinWorld, dtBoundsMaxWorld, 2f, out dtSuper0, out dtSuper1, out dtSuper2);
        delaunayHierarchy?.Dispose();
        delaunayHierarchy = new DTHierarchy(SimulationController.Instance.delaunayShader);
        delaunayHierarchy.InitFromMeshlessNodes(
            nodes,
            layerEndIndex,
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