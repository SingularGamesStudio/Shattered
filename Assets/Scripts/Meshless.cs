using System.Collections.Generic;
using System.Linq;
using Unity.Mathematics;
using UnityEngine;
using Physics;
using GPU.Delaunay;

public class Meshless : MonoBehaviour {
    public static readonly List<Meshless> Active = new List<Meshless>(64);

    public HNSW hnsw;

    public List<Node> nodes = new List<Node>();

    [HideInInspector]
    public int maxLayer = -1;

    [Header("Simulation parameters")]
    public float gravity = -9.81f;

    [Header("XPBI compliance")]
    public float compliance = 0f;

    public NodeBatch lastBatchDebug;

    public int[] levelEndIndex;

    [Header("GPU Delaunay hierarchy")]
    public bool useDelaunayHierarchy = true;
    public ComputeShader delaunayShader;
    public int dtFixIterationsPerTick = 1;
    public int dtLegalizeIterationsPerTick = 1;
    public int dtWarmupFixIterations = 64;
    public int dtWarmupLegalizeIterations = 128;
    public float dtNormalizePadding = 2f;

    [HideInInspector] public DelaunayHierarchyGpu delaunayHierarchy;

    float2 dtNormCenter;
    float dtNormInvHalfExtent;
    float2 dtBoundsMinWorld;
    float2 dtBoundsMaxWorld;

    readonly float2 dtSuper0 = new float2(0f, 3f);
    readonly float2 dtSuper1 = new float2(-3f, -3f);
    readonly float2 dtSuper2 = new float2(3f, -3f);

    [Header("Material")]
    public MeshlessMaterialDef baseMaterialDef;

    public float2 DtNormCenter => dtNormCenter;
    public float DtNormInvHalfExtent => dtNormInvHalfExtent;

    public int GetBaseMaterialId() {
        var lib = MeshlessMaterialLibrary.Instance;
        if (lib == null) return 0;
        return lib.GetMaterialIndex(baseMaterialDef);
    }

    public bool TryGetLevelDt(int level, out DelaunayGpu dt) {
        dt = null;
        if (!useDelaunayHierarchy || delaunayHierarchy == null) return false;
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
        for (int i = 0; i < nodes.Count; i++)
            maxLayer = math.max(maxLayer, nodes[i].maxLayer);

        BuildLevelEndIndex();

        if (useDelaunayHierarchy) {
            if (!delaunayShader) throw new System.InvalidOperationException("Meshless: delaunayShader is not assigned.");

            RecomputeDelaunayNormalizationBounds();
            BuildDelaunayHierarchy();

            ComputeRestVolumesFromDelaunayTriangles();

            BuildHierarchy();
        } else {
            hnsw = new HNSW(this);

            const int volumeNeighborCount = 6;
            var knnScratch = new List<int>(8);

            for (int i = 0; i < nodes.Count; i++) {
                Node node = nodes[i];

                hnsw.SearchKnn(node.pos, volumeNeighborCount + 1, knnScratch);

                if (knnScratch.Contains(i)) {
                    knnScratch.Remove(i);
                } else if (knnScratch.Count > volumeNeighborCount) {
                    knnScratch.RemoveAt(volumeNeighborCount);
                }

                if (knnScratch.Count < 2) {
                    node.restVolume = 0.0f;
                    continue;
                }

                int nCount = knnScratch.Count;
                float2[] rel = new float2[nCount];
                float[] ang = new float[nCount];

                for (int k = 0; k < nCount; k++) {
                    float2 v = nodes[knnScratch[k]].pos - node.pos;
                    rel[k] = v;
                    ang[k] = math.atan2(v.y, v.x);
                }

                System.Array.Sort(ang, rel);

                float area = 0.0f;
                for (int k = 0; k < nCount; k++) {
                    int next = (k + 1) % nCount;

                    float dTheta = ang[next] - ang[k];
                    if (dTheta < 0.0f) dTheta += 2.0f * math.PI;

                    if (dTheta > math.PI) continue;

                    float2 a = rel[k];
                    float2 b = rel[next];

                    float wedgeArea = 0.5f * math.abs(a.x * b.y - a.y * b.x);
                    area += wedgeArea;
                }

                node.restVolume = area / 3.0f;
            }

            BuildHierarchy();
        }
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

    public void BuildHierarchy() {
        if (maxLayer < 0) return;
        if (levelEndIndex == null || levelEndIndex.Length != maxLayer + 1)
            BuildLevelEndIndex();

        for (int i = 0; i < nodes.Count; i++) {
            BuildParentRelationship(i);
        }
    }

    void BuildParentRelationship(int nodeIdx) {
        Node node = nodes[nodeIdx];
        int parentLevel = node.maxLayer + 1;

        if (parentLevel > maxLayer) {
            node.parentIndex = -1;
            return;
        }

        if (!useDelaunayHierarchy || delaunayHierarchy == null) {
            node.parentIndex = -1;
            return;
        }

        node.parentIndex = delaunayHierarchy.FindNearestCoarseToFine(parentLevel, node.pos, nodes);
    }

    public int NodeCount(int level) {
        if (levelEndIndex == null || level < 0 || level > maxLayer) return 0;
        return levelEndIndex[level];
    }

    public void UpdateDelaunayAfterIntegration() {
        if (!useDelaunayHierarchy || delaunayHierarchy == null) return;

        delaunayHierarchy.UpdatePositionsFromNodesAllLevels(nodes, dtNormCenter, dtNormInvHalfExtent);
        delaunayHierarchy.MaintainAllLevels(dtFixIterationsPerTick, dtLegalizeIterationsPerTick);
        delaunayHierarchy.ReadbackAllLevels();
    }

    public void GetNeighborsForLevel(int level, int nodeIndex, List<int> dst) {
        delaunayHierarchy?.FillNeighbors(level, nodeIndex, dst);
    }

    void RecomputeDelaunayNormalizationBounds() {
        if (nodes.Count == 0) return;

        float2 min = nodes[0].pos;
        float2 max = nodes[0].pos;

        for (int i = 1; i < nodes.Count; i++) {
            float2 p = nodes[i].pos;
            min = math.min(min, p);
            max = math.max(max, p);
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
        delaunayHierarchy = new DelaunayHierarchyGpu(delaunayShader);

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
    }
}
