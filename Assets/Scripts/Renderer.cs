using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Rendering;
using GPU.Delaunay;

[DefaultExecutionOrder(1000)]
public sealed class Renderer : MonoBehaviour {
    public enum SdfSmoothMode {
        None = 0,
        Blur = 1,
        CurvatureFlow = 2
    }

    [Header("Shaders")]
    public Shader wireShader;

    [Header("Postprocess")]
    public Shader accumShader;
    public Shader compositeShader;
    public ComputeShader sdfCompute;

    [Header("UV")]
    public float uvScale = 0.25f;

    [Header("SDF")]
    [Min(1)] public int sdfDownsample = 1;
    [Range(-20f, 20f)] public float roundPixels = 1.5f;

    [Header("SDF smoothing")]
    public SdfSmoothMode smoothMode = SdfSmoothMode.CurvatureFlow;
    [Range(0, 32)] public int blurRadiusPixels = 6;
    [Range(0, 16)] public int blurIterations = 2;
    [Range(0, 200)] public int curvatureIterations = 40;
    [Range(0.0f, 1.0f)] public float curvatureDt = 0.2f;

    [Header("Debug")]
    public bool debugShowAccum = false;
    public bool debugShowSdf = false;

    [Header("Wireframe")]
    public bool showWireframe = true;
    public bool drawCoarseLayers = true;
    [Range(0.5f, 10f)] public float wireWidthPixels = 1.5f;
    Color wireColorLayer0 = new Color(0.15f, 0.35f, 1f, 1f);
    Color wireColorMaxLayer = new Color(1f, 0.2f, 0.2f, 1f);

    [Header("Layers")]
    public bool drawLayer0Fill = true;

    Material accumMaterial;
    Material compositeMaterial;
    Material wireMaterial;
    MaterialPropertyBlock mpb;

    Camera cam;

    sealed class MeshlessState {
        public ComputeBuffer materialIds;
        public int[] materialIdsCpu;

        public ComputeBuffer restVolumes;
        public float[] restVolumesCpu;

        public ComputeBuffer restNorm;
        public Vector2[] restNormCpu;

        public int capacity;
        public float2 lastCenter;
        public float lastInvHalfExtent;
    }

    sealed class CameraResources {
        public int w, h;

        public RenderTexture accum;   // ARGBHalf: rgb=sum(color), a=sum(weight)
        public RenderTexture seedA;   // ARGBHalf UAV
        public RenderTexture seedB;   // ARGBHalf UAV
        public RenderTexture sdf;     // ARGBHalf UAV (base sdf in .r)
        public RenderTexture tmp;     // ARGBHalf UAV (smoothing ping-pong)

        public CommandBuffer fillCmd; // BeforeImageEffects
        public CommandBuffer wireCmd; // AfterImageEffects (overlay)

        public void Release(Camera owner) {
            if (owner != null) {
                if (fillCmd != null) owner.RemoveCommandBuffer(CameraEvent.BeforeImageEffects, fillCmd);
                if (wireCmd != null) owner.RemoveCommandBuffer(CameraEvent.AfterImageEffects, wireCmd);
            }

            fillCmd?.Release();
            wireCmd?.Release();
            fillCmd = null;
            wireCmd = null;

            accum?.Release();
            seedA?.Release();
            seedB?.Release();
            sdf?.Release();
            tmp?.Release();

            accum = null;
            seedA = null;
            seedB = null;
            sdf = null;
            tmp = null;
            w = h = 0;
        }
    }

    readonly Dictionary<Meshless, MeshlessState> states = new Dictionary<Meshless, MeshlessState>(64);
    CameraResources cr;

    int kInitKernel = -1;
    int kJfaKernel = -1;
    int kFinalizeKernel = -1;
    int kBlurHKernel = -1;
    int kBlurVKernel = -1;
    int kCurvatureKernel = -1;

    static readonly int ID_AlbedoArray = Shader.PropertyToID("_AlbedoArray");
    static readonly int ID_UvScale = Shader.PropertyToID("_UvScale");

    static readonly int ID_PositionsPrev = Shader.PropertyToID("_PositionsPrev");
    static readonly int ID_PositionsCurr = Shader.PropertyToID("_PositionsCurr");
    static readonly int ID_RenderAlpha = Shader.PropertyToID("_RenderAlpha");
    static readonly int ID_Positions = Shader.PropertyToID("_Positions");

    static readonly int ID_HalfEdges = Shader.PropertyToID("_HalfEdges");
    static readonly int ID_TriToHE = Shader.PropertyToID("_TriToHE");

    static readonly int ID_MaterialIds = Shader.PropertyToID("_MaterialIds");
    static readonly int ID_MaterialCount = Shader.PropertyToID("_MaterialCount");
    static readonly int ID_RestVolumes = Shader.PropertyToID("_RestVolumes");

    static readonly int ID_RestNormPositions = Shader.PropertyToID("_RestNormPositions");
    static readonly int ID_RealPointCount = Shader.PropertyToID("_RealPointCount");
    static readonly int ID_LayerKernelH = Shader.PropertyToID("_LayerKernelH");
    static readonly int ID_WendlandSupportScale = Shader.PropertyToID("_WendlandSupportScale");

    static readonly int ID_NormCenter = Shader.PropertyToID("_NormCenter");
    static readonly int ID_NormInvHalfExtent = Shader.PropertyToID("_NormInvHalfExtent");

    static readonly int ID_AccumTex = Shader.PropertyToID("_AccumTex");
    static readonly int ID_SdfTex = Shader.PropertyToID("_SdfTex");
    static readonly int ID_RoundPixels = Shader.PropertyToID("_RoundPixels");

    // Compute
    static readonly int ID_CSTexAccum = Shader.PropertyToID("_Accum");
    static readonly int ID_CSTexSeedIn = Shader.PropertyToID("_SeedIn");
    static readonly int ID_CSTexSeedOut = Shader.PropertyToID("_SeedOut");
    static readonly int ID_CSTexSdfIn = Shader.PropertyToID("_SdfIn");
    static readonly int ID_CSTexSdfOut = Shader.PropertyToID("_SdfOut");
    static readonly int ID_CSJump = Shader.PropertyToID("_Jump");
    static readonly int ID_CSDim = Shader.PropertyToID("_Dim");
    static readonly int ID_CSRadius = Shader.PropertyToID("_Radius");
    static readonly int ID_CSDt = Shader.PropertyToID("_Dt");

    void Awake() {
        cam = GetComponent<Camera>();
        mpb ??= new MaterialPropertyBlock();

        if (sdfCompute != null) {
            kInitKernel = sdfCompute.FindKernel("KInitBoundarySeeds");
            kJfaKernel = sdfCompute.FindKernel("KJumpFlood");
            kFinalizeKernel = sdfCompute.FindKernel("KFinalizeSdf");
            kBlurHKernel = sdfCompute.FindKernel("KBlurH");
            kBlurVKernel = sdfCompute.FindKernel("KBlurV");
            kCurvatureKernel = sdfCompute.FindKernel("KCurvatureFlow");
        }
    }

    void OnDisable() {
        foreach (var kv in states) {
            kv.Value.materialIds?.Dispose();
            kv.Value.restVolumes?.Dispose();
            kv.Value.restNorm?.Dispose();
        }
        states.Clear();

        cr?.Release(cam);
        cr = null;

        if (accumMaterial != null) Destroy(accumMaterial);
        if (compositeMaterial != null) Destroy(compositeMaterial);
        if (wireMaterial != null) Destroy(wireMaterial);

        accumMaterial = null;
        compositeMaterial = null;
        wireMaterial = null;

        mpb = null;
        cam = null;
    }

    void OnPreCull() {
        if (cam == null) return;

        var lib = MaterialLibrary.Instance;
        bool canFill = drawLayer0Fill && accumShader != null && compositeShader != null && sdfCompute != null && lib != null && lib.AlbedoArray != null;
        bool canWire = showWireframe && wireShader != null;

        // Early out if neither draw path is active
        if (!canFill && !canWire) return;

        int w = Mathf.Max(1, cam.pixelWidth / Mathf.Max(1, sdfDownsample));
        int h = Mathf.Max(1, cam.pixelHeight / Mathf.Max(1, sdfDownsample));
        EnsureCameraResources(w, h);

        if (canFill) {
            accumMaterial ??= new Material(accumShader);

            cr.fillCmd.Clear();
            cr.fillCmd.SetRenderTarget(cr.accum);
            cr.fillCmd.ClearRenderTarget(true, true, Color.clear);

            var list = Meshless.Active;
            for (int mi = 0; mi < list.Count; mi++) {
                var m = list[mi];
                if (m == null || !m.isActiveAndEnabled) continue;
                if (m.nodes == null || m.nodes.Count < 3) continue;

                if (!m.TryGetLayerDt(0, out var dt0) || dt0 == null || dt0.TriCount <= 0) continue;

                EnsurePerNodeBuffers(m);

                int realCount = m.NodeCount(0);
                if (realCount <= 0) continue;

                SetupCommon(m, dt0, lib, realCount, 0);
                mpb.SetFloat(ID_UvScale, uvScale);

                cr.fillCmd.DrawProcedural(Matrix4x4.identity, accumMaterial, 0, MeshTopology.Triangles, dt0.TriCount * 15, 1, mpb);
            }
        } else {
            cr.fillCmd.Clear();
        }

        if (canWire) {
            wireMaterial ??= new Material(wireShader);

            cr.wireCmd.Clear();
            cr.wireCmd.SetRenderTarget(BuiltinRenderTextureType.CameraTarget);

            var list = Meshless.Active;
            for (int mi = 0; mi < list.Count; mi++) {
                var m = list[mi];
                if (m == null || !m.isActiveAndEnabled) continue;
                if (m.nodes == null || m.nodes.Count < 3) continue;

                EnsurePerNodeBuffers(m);

                if (m.NodeCount(0) > 1000) continue;

                int maxLayer = m.maxLayer;
                for (int layer = 0; layer <= maxLayer; layer++) {
                    if (layer != 0 && !drawCoarseLayers) continue;

                    if (!m.TryGetLayerDt(layer, out var dt) || dt == null) continue;
                    if (dt.TriCount <= 0) continue;

                    int realCount = m.NodeCount(layer);
                    if (realCount <= 0) continue;

                    float t = maxLayer <= 0 ? 0f : (float)layer / maxLayer;
                    Color wireColor = Color.Lerp(wireColorLayer0, wireColorMaxLayer, t);

                    SetupCommon(m, dt, lib, realCount, layer);
                    mpb.SetColor("_WireColor", wireColor);
                    mpb.SetFloat("_WireWidthPx", wireWidthPixels);

                    int vertexCount = dt.TriCount * 3 * 6;
                    cr.wireCmd.DrawProcedural(Matrix4x4.identity, wireMaterial, 0, MeshTopology.Triangles, vertexCount, 1, mpb);
                }
            }
        } else {
            cr.wireCmd.Clear();
        }
    }

    void OnRenderImage(RenderTexture src, RenderTexture dest) {
        if (!drawLayer0Fill || compositeShader == null || sdfCompute == null || cr == null || cr.accum == null) {
            Graphics.Blit(src, dest);
            return;
        }

        compositeMaterial ??= new Material(compositeShader);

        RenderTexture sdfFinal = RunSdfAndSmooth();

        compositeMaterial.SetTexture(ID_AccumTex, cr.accum);
        compositeMaterial.SetTexture(ID_SdfTex, sdfFinal);
        compositeMaterial.SetFloat(ID_RoundPixels, roundPixels);

        if (debugShowAccum) {
            Graphics.Blit(src, dest, compositeMaterial, 1);
            return;
        }

        if (debugShowSdf) {
            Graphics.Blit(src, dest, compositeMaterial, 2);
            return;
        }

        Graphics.Blit(src, dest, compositeMaterial, 0);
        // Wireframe is drawn AFTER this via cr.wireCmd at CameraEvent.AfterImageEffects.
    }

    void EnsureCameraResources(int w, int h) {
        if (cr != null && cr.w == w && cr.h == h && cr.accum != null && cr.fillCmd != null && cr.wireCmd != null)
            return;

        cr?.Release(cam);
        cr = new CameraResources { w = w, h = h };

        cr.accum = NewRT(w, h, false, FilterMode.Bilinear);
        cr.seedA = NewRT(w, h, true, FilterMode.Point);
        cr.seedB = NewRT(w, h, true, FilterMode.Point);
        cr.sdf = NewRT(w, h, true, FilterMode.Point);
        cr.tmp = NewRT(w, h, true, FilterMode.Point);

        cr.fillCmd = new CommandBuffer { name = "Triangulation Accum (Mask+Color)" };
        cr.wireCmd = new CommandBuffer { name = "Triangulation Wire Overlay" };

        // Built-in pipeline: schedule at specific points in the camera render loop. [web:134][web:136]
        cam.AddCommandBuffer(CameraEvent.BeforeImageEffects, cr.fillCmd);
        cam.AddCommandBuffer(CameraEvent.AfterImageEffects, cr.wireCmd); // overlay after postprocess/composite [web:135][web:136]
    }

    static RenderTexture NewRT(int w, int h, bool randomWrite, FilterMode filter) {
        var rt = new RenderTexture(w, h, 0, RenderTextureFormat.ARGBHalf, RenderTextureReadWrite.Linear) {
            filterMode = filter,
            wrapMode = TextureWrapMode.Clamp,
            useMipMap = false,
            autoGenerateMips = false,
            enableRandomWrite = randomWrite
        };
        rt.Create();
        return rt;
    }

    RenderTexture RunSdfAndSmooth() {
        int tgx = (cr.w + 7) >> 3;
        int tgy = (cr.h + 7) >> 3;

        sdfCompute.SetInts(ID_CSDim, cr.w, cr.h);

        sdfCompute.SetTexture(kInitKernel, ID_CSTexAccum, cr.accum);
        sdfCompute.SetTexture(kInitKernel, ID_CSTexSeedOut, cr.seedA);
        sdfCompute.Dispatch(kInitKernel, tgx, tgy, 1);

        int maxDim = Mathf.Max(cr.w, cr.h);
        int jump = 1;
        while (jump < maxDim) jump <<= 1;
        jump >>= 1;

        RenderTexture seedIn = cr.seedA;
        RenderTexture seedOut = cr.seedB;

        while (jump >= 1) {
            sdfCompute.SetInt(ID_CSJump, jump);
            sdfCompute.SetTexture(kJfaKernel, ID_CSTexSeedIn, seedIn);
            sdfCompute.SetTexture(kJfaKernel, ID_CSTexSeedOut, seedOut);
            sdfCompute.Dispatch(kJfaKernel, tgx, tgy, 1);

            var tmp = seedIn;
            seedIn = seedOut;
            seedOut = tmp;

            jump >>= 1;
        }

        sdfCompute.SetTexture(kFinalizeKernel, ID_CSTexAccum, cr.accum);
        sdfCompute.SetTexture(kFinalizeKernel, ID_CSTexSeedIn, seedIn);
        sdfCompute.SetTexture(kFinalizeKernel, ID_CSTexSdfOut, cr.sdf);
        sdfCompute.Dispatch(kFinalizeKernel, tgx, tgy, 1);

        if (smoothMode == SdfSmoothMode.None)
            return cr.sdf;

        if (smoothMode == SdfSmoothMode.Blur) {
            int r = Mathf.Clamp(blurRadiusPixels, 0, 32);
            int it = Mathf.Clamp(blurIterations, 0, 16);
            if (r <= 0 || it <= 0) return cr.sdf;

            RenderTexture a = cr.sdf;
            RenderTexture b = cr.tmp;

            sdfCompute.SetInt(ID_CSRadius, r);

            for (int i = 0; i < it; i++) {
                sdfCompute.SetTexture(kBlurHKernel, ID_CSTexSdfIn, a);
                sdfCompute.SetTexture(kBlurHKernel, ID_CSTexSdfOut, b);
                sdfCompute.Dispatch(kBlurHKernel, tgx, tgy, 1);

                sdfCompute.SetTexture(kBlurVKernel, ID_CSTexSdfIn, b);
                sdfCompute.SetTexture(kBlurVKernel, ID_CSTexSdfOut, a);
                sdfCompute.Dispatch(kBlurVKernel, tgx, tgy, 1);
            }

            return a;
        }

        {
            int it = Mathf.Clamp(curvatureIterations, 0, 200);
            float dt = Mathf.Clamp01(curvatureDt);
            if (it <= 0 || dt <= 0f) return cr.sdf;

            RenderTexture a = cr.sdf;
            RenderTexture b = cr.tmp;

            sdfCompute.SetFloat(ID_CSDt, dt);

            for (int i = 0; i < it; i++) {
                sdfCompute.SetTexture(kCurvatureKernel, ID_CSTexSdfIn, a);
                sdfCompute.SetTexture(kCurvatureKernel, ID_CSTexSdfOut, b);
                sdfCompute.Dispatch(kCurvatureKernel, tgx, tgy, 1);

                var tmp = a;
                a = b;
                b = tmp;
            }

            if (a != cr.sdf)
                Graphics.Blit(a, cr.sdf);

            return cr.sdf;
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
            st.restVolumes?.Dispose();
            st.restNorm?.Dispose();

            st.materialIds = new ComputeBuffer(n, sizeof(int), ComputeBufferType.Structured);
            st.materialIdsCpu = new int[n];

            st.restVolumes = new ComputeBuffer(n, sizeof(float), ComputeBufferType.Structured);
            st.restVolumesCpu = new float[n];

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

        bool restVolumesChanged = reallocated;
        for (int i = 0; i < n; i++) {
            float v = m.nodes[i].restVolume;
            if (!Mathf.Approximately(st.restVolumesCpu[i], v)) {
                st.restVolumesCpu[i] = v;
                restVolumesChanged = true;
            }
        }
        if (restVolumesChanged)
            st.restVolumes.SetData(st.restVolumesCpu);

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

    void SetupCommon(Meshless m, DT dt, MaterialLibrary lib, int realPointCount, int layer) {
        if (!states.TryGetValue(m, out var st) || st.materialIds == null || st.restNorm == null) return;

        float alpha = 0f;
        var sc = SimulationController.Instance;
        if (sc != null) alpha = sc.RenderAlpha;

        mpb.Clear();

        if (lib != null && lib.AlbedoArray != null) {
            mpb.SetTexture(ID_AlbedoArray, lib.AlbedoArray);
            mpb.SetInt(ID_MaterialCount, lib.MaterialCount);
        }

        mpb.SetBuffer(ID_PositionsPrev, dt.PositionsBuffer);
        mpb.SetBuffer(ID_PositionsCurr, dt.PositionsBuffer);
        mpb.SetFloat(ID_RenderAlpha, alpha);

        mpb.SetBuffer(ID_Positions, dt.PositionsBuffer);

        mpb.SetBuffer(ID_HalfEdges, dt.HalfEdgesBuffer);
        mpb.SetBuffer(ID_TriToHE, dt.TriToHEBuffer);

        mpb.SetBuffer(ID_MaterialIds, st.materialIds);
        mpb.SetBuffer(ID_RestVolumes, st.restVolumes);
        mpb.SetBuffer(ID_RestNormPositions, st.restNorm);

        float layerKernelH = m.GetLayerKernelH(layer);
        float layerKernelHNorm = layerKernelH * m.DtNormInvHalfExtent;
        mpb.SetFloat(ID_LayerKernelH, layerKernelHNorm);
        mpb.SetFloat(ID_WendlandSupportScale, Const.WendlandSupport);

        mpb.SetInt(ID_RealPointCount, realPointCount);

        mpb.SetVector(ID_NormCenter, new Vector4(m.DtNormCenter.x, m.DtNormCenter.y, 0f, 0f));
        mpb.SetFloat(ID_NormInvHalfExtent, m.DtNormInvHalfExtent);
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
