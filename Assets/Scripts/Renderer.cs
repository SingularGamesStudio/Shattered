using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using GPU.Delaunay;

[DefaultExecutionOrder(1000)]
public sealed class Renderer : MonoBehaviour {
    [Header("Shaders")]
    public Shader fillShader;
    public Shader wireShader;

    [Header("UV")]
    public float uvScale = 0.25f;

    [Header("Wireframe")]
    public bool showWireframe = true;
    public bool drawCoarseLayers = true;
    [Range(0.5f, 10f)] public float wireWidthPixels = 1.5f;
    Color wireColorLayer0 = new Color(0.15f, 0.35f, 1f, 1f);
    Color wireColorMaxLayer = new Color(1f, 0.2f, 0.2f, 1f);

    [Header("Layers")]
    public bool drawLayer0Fill = true;

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

            int maxLayer = m.maxLayer;
            Bounds bounds = ComputeBoundsFromNorm(m);

            // Fill: layer 0 only.
            if (drawLayer0Fill && m.TryGetLayerDt(0, out var dt0) && dt0 != null && dt0.TriCount > 0) {
                SetupCommon(m, dt0, lib, m.NodeCount(0));

                mpb.SetFloat("_UvScale", uvScale);
                Graphics.DrawProcedural(fillMaterial, bounds, MeshTopology.Triangles, dt0.TriCount * 3, 1, null, mpb);
            }

            if (!showWireframe || m.NodeCount(0) > 1000) continue;

            // Wireframe: edges only, still per-layer to keep coloring and layer control.
            for (int layer = 0; layer <= maxLayer; layer++) {
                if (layer != 0 && !drawCoarseLayers) continue;

                if (!m.TryGetLayerDt(layer, out var dt) || dt == null) continue;
                if (dt.TriCount <= 0) continue;

                int realCount = m.NodeCount(layer);
                if (realCount <= 0) continue;

                float t = maxLayer <= 0 ? 0f : (float)layer / maxLayer;
                Color wireColor = Color.Lerp(wireColorLayer0, wireColorMaxLayer, t);

                SetupCommon(m, dt, lib, realCount);

                mpb.SetColor("_WireColor", wireColor);
                mpb.SetFloat("_WireWidthPx", wireWidthPixels);

                // 3 edges per triangle, each edge is a quad (2 triangles = 6 vertices).
                int vertexCount = dt.TriCount * 3 * 6;
                Graphics.DrawProcedural(wireMaterial, bounds, MeshTopology.Triangles, vertexCount, 1, null, mpb);
            }
        }
    }

    void EnsurePerNodeBuffers(Meshless m) {
        if (!states.TryGetValue(m, out var st)) {
            st = new MeshlessState();
            states[m] = st;
        }

        int n = m.nodes.Count;
        bool reallocated = st.capacity != n || st.materialIds == null || st.restNorm == null;
        if (reallocated) {
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

        bool materialIdsChanged = reallocated;
        for (int i = 0; i < n; i++) {
            int id = m.nodes[i].materialId;
            if (st.materialIdsCpu[i] != id) {
                st.materialIdsCpu[i] = id;
                materialIdsChanged = true;
            }
        }
        if (materialIdsChanged)
            st.materialIds.SetData(st.materialIdsCpu);

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

        float alpha = 0f;
        var sc = SimulationController.Instance;
        if (sc != null) alpha = sc.RenderAlpha;

        mpb.Clear();
        mpb.SetTexture("_AlbedoArray", lib.AlbedoArray);

        mpb.SetBuffer("_PositionsPrev", dt.PositionsBuffer);
        mpb.SetBuffer("_PositionsCurr", dt.PositionsBuffer);
        mpb.SetFloat("_RenderAlpha", alpha);

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

    static Bounds ComputeBoundsFromNorm(Meshless m) {
        float inv = m.DtNormInvHalfExtent;
        if (!(inv > 0f) || float.IsNaN(inv) || float.IsInfinity(inv))
            return ComputeBoundsFromNodes(m.nodes);

        float halfExtent = 1f / inv;

        float pad = 50f;
        Vector3 center = new Vector3(m.DtNormCenter.x, m.DtNormCenter.y, 0f);
        Vector3 size = new Vector3(halfExtent * 2f + pad, halfExtent * 2f + pad, 10f);
        return new Bounds(center, size);
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
}
