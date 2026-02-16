using System.Collections.Generic;
using UnityEngine;
using GPU.Delaunay;

[DefaultExecutionOrder(1000)]
public sealed class MeshlessTriangulationRenderer : MonoBehaviour {
    [Header("Shader")]
    public Shader triangulationShader;

    [Header("UV")]
    public float uvScale = 0.25f;

    [Header("Wireframe")]
    public bool showWireframe = true;
    public Color wireColorLevel0 = new Color(0.15f, 0.35f, 1f, 1f);
    public Color wireColorMaxLevel = new Color(1f, 0.2f, 0.2f, 1f);
    [Range(0.25f, 8f)] public float wireWidth = 1.5f;

    [Header("Levels")]
    public bool drawLevel0Fill = true;
    public bool drawCoarseLevels = true;
    public bool coarseWireOnly = true;

    Material material;
    MaterialPropertyBlock mpb;

    sealed class MeshlessState {
        public ComputeBuffer materialIds;
        public int[] materialIdsCpu;
        public int capacity;
    }

    readonly Dictionary<Meshless, MeshlessState> states = new Dictionary<Meshless, MeshlessState>(64);

    void Awake() {
        mpb ??= new MaterialPropertyBlock();
    }

    void OnDisable() {
        foreach (var kv in states) kv.Value.materialIds?.Dispose();
        states.Clear();

        if (material != null) Destroy(material);
        material = null;
        mpb = null;
    }

    void LateUpdate() {
        if (!triangulationShader) return;

        mpb ??= new MaterialPropertyBlock();

        var lib = MeshlessMaterialLibrary.Instance;
        if (lib == null || lib.AlbedoArray == null) return;

        material ??= new Material(triangulationShader);

        var list = Meshless.Active;
        for (int mi = 0; mi < list.Count; mi++) {
            var m = list[mi];
            if (m == null || !m.isActiveAndEnabled) continue;
            if (m.nodes == null || m.nodes.Count < 3) continue;
            if (!m.useDelaunayHierarchy || m.delaunayHierarchy == null) continue;

            EnsureMaterialIds(m);

            int maxLevel = m.maxLayer;

            // Fill: only level 0.
            bool drewFill0 = false;
            if (drawLevel0Fill && m.TryGetLevelDt(0, out var dt0) && dt0 != null && dt0.TriCount > 0) {
                SetupCommon(m, dt0, lib);
                mpb.SetFloat("_UvScale", uvScale);
                var bounds = new Bounds(Vector3.zero, new Vector3(1e6f, 1e6f, 10f));
                Graphics.DrawProcedural(material, bounds, MeshTopology.Triangles, dt0.TriCount * 3, 1, null, mpb, 0);
                drewFill0 = true;
            }

            // Wire overlay.
            if (!showWireframe) continue;

            for (int level = 0; level <= maxLevel; level++) {
                if (level != 0 && !drawCoarseLevels) continue;

                // Avoid thick outline: if we drew fill for level 0, skip level-0 wire.
                if (level == 0 && drewFill0) continue;

                if (!m.TryGetLevelDt(level, out var dt) || dt == null) continue;
                if (dt.TriCount <= 0) continue;

                float t = maxLevel <= 0 ? 0f : (float)level / maxLevel;
                Color wireColor = Color.Lerp(wireColorLevel0, wireColorMaxLevel, t);

                SetupCommon(m, dt, lib);
                mpb.SetColor("_WireColor", wireColor);
                mpb.SetFloat("_WireWidth", wireWidth);

                var bounds = new Bounds(Vector3.zero, new Vector3(1e6f, 1e6f, 10f));
                Graphics.DrawProcedural(material, bounds, MeshTopology.Triangles, dt.TriCount * 3, 1, null, mpb, layer: 1);
            }
        }
    }

    void SetupCommon(Meshless m, DelaunayGpu dt, MeshlessMaterialLibrary lib) {
        if (!states.TryGetValue(m, out var st) || st.materialIds == null) return;

        mpb.Clear();
        mpb.SetTexture("_AlbedoArray", lib.AlbedoArray);

        mpb.SetBuffer("_Positions", dt.PositionsBuffer);
        mpb.SetBuffer("_HalfEdges", dt.HalfEdgesBuffer);
        mpb.SetBuffer("_TriToHE", dt.TriToHEBuffer);
        mpb.SetBuffer("_MaterialIds", st.materialIds);

        mpb.SetInt("_MaterialCount", lib.MaterialCount);
        mpb.SetVector("_NormCenter", new Vector4(m.DtNormCenter.x, m.DtNormCenter.y, 0f, 0f));
        mpb.SetFloat("_NormInvHalfExtent", m.DtNormInvHalfExtent);
    }

    void EnsureMaterialIds(Meshless m) {
        if (!states.TryGetValue(m, out var st)) {
            st = new MeshlessState();
            states[m] = st;
        }

        int n = m.nodes.Count;
        if (st.materialIds == null || st.capacity != n) {
            st.materialIds?.Dispose();
            st.materialIds = new ComputeBuffer(n, sizeof(int), ComputeBufferType.Structured);
            st.materialIdsCpu = new int[n];
            st.capacity = n;
        }

        for (int i = 0; i < n; i++) st.materialIdsCpu[i] = m.nodes[i].materialId;
        st.materialIds.SetData(st.materialIdsCpu);
    }
}
