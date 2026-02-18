using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using GPU.Delaunay;

[DefaultExecutionOrder(1000)]
public sealed class Renderer : MonoBehaviour {
    [Header("Shaders")]
    public Shader fillShader;
    public Shader wireShader;

    [Header("UV (rest normalized DT space)")]
    public float uvScale = 0.25f;

    [Header("Wireframe")]
    public bool showWireframe = true;
    public Color wireColorLevel0 = new Color(0.15f, 0.35f, 1f, 1f);
    public Color wireColorMaxLevel = new Color(1f, 0.2f, 0.2f, 1f);
    [Range(0.5f, 10f)] public float wireWidthPixels = 1.5f;

    [Header("Levels")]
    public bool drawLevel0Fill = true;
    public bool drawCoarseLevels = true;

    [Header("Culling bounds")]
    public bool preferGpuSnapshotBounds = true;

    Material fillMaterial;
    Material wireMaterial;
    MaterialPropertyBlock mpb;

    sealed class MeshlessState {
        public ComputeBuffer materialIds;
        public int[] materialIdsCpu;

        public ComputeBuffer restNorm;
        public Vector2[] restNormCpu;

        public int capacity;
        public float2 lastCenter;
        public float lastInvHalfExtent;
    }

    readonly Dictionary<Meshless, MeshlessState> states = new Dictionary<Meshless, MeshlessState>(64);

    void Awake() {
        mpb ??= new MaterialPropertyBlock();
    }

    void OnDisable() {
        foreach (var kv in states) {
            kv.Value.materialIds?.Dispose();
            kv.Value.restNorm?.Dispose();
        }
        states.Clear();

        if (fillMaterial != null) Destroy(fillMaterial);
        if (wireMaterial != null) Destroy(wireMaterial);
        fillMaterial = null;
        wireMaterial = null;

        mpb = null;
    }

    void LateUpdate() {
        if (!fillShader || !wireShader) return;

        mpb ??= new MaterialPropertyBlock();

        var lib = MaterialLibrary.Instance;
        if (lib == null || lib.AlbedoArray == null) return;

        fillMaterial ??= new Material(fillShader);
        wireMaterial ??= new Material(wireShader);

        var list = Meshless.Active;
        for (int mi = 0; mi < list.Count; mi++) {
            var m = list[mi];
            if (m == null || !m.isActiveAndEnabled) continue;
            if (m.nodes == null || m.nodes.Count < 3) continue;

            EnsurePerNodeBuffers(m);

            int maxLevel = m.maxLayer;
            Bounds bounds = ComputeBounds(m);

            // Fill: only level 0.
            if (drawLevel0Fill && m.TryGetLevelDt(0, out var dt0) && dt0 != null && dt0.TriCount > 0) {
                SetupCommon(m, dt0, lib, m.NodeCount(0));

                mpb.SetFloat("_UvScale", uvScale);
                Graphics.DrawProcedural(fillMaterial, bounds, MeshTopology.Triangles, dt0.TriCount * 3, 1, null, mpb);
            }

            if (!showWireframe) continue;

            for (int level = 0; level <= maxLevel; level++) {
                if (level != 0 && !drawCoarseLevels) continue;

                if (!m.TryGetLevelDt(level, out var dt) || dt == null) continue;
                if (dt.TriCount <= 0) continue;

                int realCount = m.NodeCount(level);
                if (realCount <= 0) continue;

                float t = maxLevel <= 0 ? 0f : (float)level / maxLevel;
                Color wireColor = Color.Lerp(wireColorLevel0, wireColorMaxLevel, t);

                SetupCommon(m, dt, lib, realCount);

                mpb.SetColor("_WireColor", wireColor);
                mpb.SetFloat("_WireWidthPx", wireWidthPixels);

                Graphics.DrawProcedural(wireMaterial, bounds, MeshTopology.Triangles, dt.TriCount * 3, 1, null, mpb);
            }
        }
    }

    Bounds ComputeBounds(Meshless m) {
        if (preferGpuSnapshotBounds) {
            var sc = SimulationController.Instance;
            if (sc != null && sc.TryGetLatestPositionsSnapshot(m, out float2[] positions, out int count, out _)) {
                if (positions != null && count >= 3)
                    return ComputeBoundsFromPositions(positions, count);
            }
        }

        return ComputeBoundsFromNodes(m.nodes);
    }

    void EnsurePerNodeBuffers(Meshless m) {
        if (!states.TryGetValue(m, out var st)) {
            st = new MeshlessState();
            states[m] = st;
        }

        int n = m.nodes.Count;
        if (st.capacity != n || st.materialIds == null || st.restNorm == null) {
            st.materialIds?.Dispose();
            st.restNorm?.Dispose();

            st.materialIds = new ComputeBuffer(n, sizeof(int), ComputeBufferType.Structured);
            st.materialIdsCpu = new int[n];

            st.restNorm = new ComputeBuffer(n, sizeof(float) * 2, ComputeBufferType.Structured);
            st.restNormCpu = new Vector2[n];

            st.capacity = n;
            st.lastCenter = new float2(float.NaN, float.NaN);
            st.lastInvHalfExtent = float.NaN;
        }

        for (int i = 0; i < n; i++)
            st.materialIdsCpu[i] = m.nodes[i].materialId;
        st.materialIds.SetData(st.materialIdsCpu);

        // Rest in normalized DT space.
        float2 center = m.DtNormCenter;
        float inv = m.DtNormInvHalfExtent;
        if (!math.all(st.lastCenter == center) || st.lastInvHalfExtent != inv) {
            for (int i = 0; i < n; i++) {
                float2 p = m.nodes[i].originalPos;
                float2 norm = (p - center) * inv;
                st.restNormCpu[i] = new Vector2(norm.x, norm.y);
            }
            st.restNorm.SetData(st.restNormCpu);

            st.lastCenter = center;
            st.lastInvHalfExtent = inv;
        }
    }

    void SetupCommon(Meshless m, DT dt, MaterialLibrary lib, int realPointCount) {
        if (!states.TryGetValue(m, out var st) || st.materialIds == null || st.restNorm == null) return;

        mpb.Clear();
        mpb.SetTexture("_AlbedoArray", lib.AlbedoArray);

        mpb.SetBuffer("_Positions", dt.PositionsBuffer);
        mpb.SetBuffer("_HalfEdges", dt.HalfEdgesBuffer);
        mpb.SetBuffer("_TriToHE", dt.TriToHEBuffer);

        mpb.SetBuffer("_MaterialIds", st.materialIds);
        mpb.SetBuffer("_RestNormPositions", st.restNorm);

        mpb.SetInt("_MaterialCount", lib.MaterialCount);
        mpb.SetInt("_RealPointCount", realPointCount);

        mpb.SetVector("_NormCenter", new Vector4(m.DtNormCenter.x, m.DtNormCenter.y, 0f, 0f));
        mpb.SetFloat("_NormInvHalfExtent", m.DtNormInvHalfExtent);
    }

    static Bounds ComputeBoundsFromNodes(List<Node> nodes) {
        float2 min = nodes[0].pos;
        float2 max = nodes[0].pos;

        for (int i = 1; i < nodes.Count; i++) {
            float2 p = nodes[i].pos;
            min = math.min(min, p);
            max = math.max(max, p);
        }

        float2 c2 = 0.5f * (min + max);
        float2 e2 = 0.5f * (max - min);

        float pad = 50f;
        Vector3 center = new Vector3(c2.x, c2.y, 0f);
        Vector3 size = new Vector3(e2.x * 2f + pad, e2.y * 2f + pad, 10f);
        return new Bounds(center, size);
    }

    static Bounds ComputeBoundsFromPositions(float2[] positions, int count) {
        float2 min = positions[0];
        float2 max = positions[0];

        for (int i = 1; i < count; i++) {
            float2 p = positions[i];
            min = math.min(min, p);
            max = math.max(max, p);
        }

        float2 c2 = 0.5f * (min + max);
        float2 e2 = 0.5f * (max - min);

        float pad = 50f;
        Vector3 center = new Vector3(c2.x, c2.y, 0f);
        Vector3 size = new Vector3(e2.x * 2f + pad, e2.y * 2f + pad, 10f);
        return new Bounds(center, size);
    }
}
