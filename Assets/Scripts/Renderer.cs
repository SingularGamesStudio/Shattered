using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Profiling;
using UnityEngine.Rendering;
using GPU.Delaunay;

[DefaultExecutionOrder(1000)]
public sealed class Renderer : MonoBehaviour {
    enum WireframeMode {
        Off = 0,
        SingleLevel = 1,
        MultiLevel = 2,
        SingleLevelCulled = 3,
        MultiLevelCulled = 4
    }

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
    public bool cullOverstretchedWireEdges = false;
    public KeyCode wireframeCycleKey = KeyCode.W;
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

    sealed class GlobalLayerState {
        public ComputeBuffer ownerByLocal;
        public int[] ownerByLocalCpu;

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

    readonly Dictionary<int, GlobalLayerState> layerStates = new Dictionary<int, GlobalLayerState>(8);
    CameraResources cr;

    const int WireframeNodeCap = 2000;

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
    static readonly int ID_OwnerByLocal = Shader.PropertyToID("_OwnerByLocal");

    static readonly int ID_MaterialIds = Shader.PropertyToID("_MaterialIds");
    static readonly int ID_MaterialCount = Shader.PropertyToID("_MaterialCount");
    static readonly int ID_RestVolumes = Shader.PropertyToID("_RestVolumes");

    static readonly int ID_RestNormPositions = Shader.PropertyToID("_RestNormPositions");
    static readonly int ID_RealPointCount = Shader.PropertyToID("_RealPointCount");
    static readonly int ID_LayerKernelH = Shader.PropertyToID("_LayerKernelH");
    static readonly int ID_WendlandSupportScale = Shader.PropertyToID("_WendlandSupportScale");
    static readonly int ID_WireCullOverstretched = Shader.PropertyToID("_CullOverstretched");

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
        foreach (var kv in layerStates) {
            kv.Value.ownerByLocal?.Dispose();
            kv.Value.materialIds?.Dispose();
            kv.Value.restVolumes?.Dispose();
            kv.Value.restNorm?.Dispose();
        }
        layerStates.Clear();

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

    void Update() {
        if (Input.GetKeyDown(wireframeCycleKey))
            CycleWireframeMode();
    }

    void CycleWireframeMode() {
        WireframeMode mode = GetWireframeMode();
        mode = (WireframeMode)(((int)mode + 1) % 5);
        SetWireframeMode(mode);
    }

    WireframeMode GetWireframeMode() {
        if (!showWireframe)
            return WireframeMode.Off;
        if (!drawCoarseLayers)
            return cullOverstretchedWireEdges ? WireframeMode.SingleLevelCulled : WireframeMode.SingleLevel;
        return cullOverstretchedWireEdges ? WireframeMode.MultiLevelCulled : WireframeMode.MultiLevel;
    }

    void SetWireframeMode(WireframeMode mode) {
        switch (mode) {
            case WireframeMode.Off:
                showWireframe = false;
                drawCoarseLayers = false;
                cullOverstretchedWireEdges = false;
                return;
            case WireframeMode.SingleLevel:
                showWireframe = true;
                drawCoarseLayers = false;
                cullOverstretchedWireEdges = false;
                return;
            case WireframeMode.MultiLevel:
                showWireframe = true;
                drawCoarseLayers = true;
                cullOverstretchedWireEdges = false;
                return;
            case WireframeMode.SingleLevelCulled:
                showWireframe = true;
                drawCoarseLayers = false;
                cullOverstretchedWireEdges = true;
                return;
            default:
                showWireframe = true;
                drawCoarseLayers = true;
                cullOverstretchedWireEdges = true;
                return;
        }
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
            var sc = SimulationController.Instance;
            if (sc != null &&
                sc.TryGetStableReadSlot(out int slot) &&
                sc.TryGetGlobalRenderBatch(out var globalHierarchy, out var globalMeshes, out var globalBaseOffsets) &&
                globalHierarchy.TryGetLayerDt(0, out var dt0) && dt0 != null && dt0.TriCount > 0 &&
                    globalHierarchy.TryGetLayerMappings(0, out int[] ownerBodyByLocal, out int[] globalNodeByLocal, out _, out int activeCount, out _) &&
                globalHierarchy.TryGetLayerExecutionContext(0, out _, out _, out float layerKernelH)) {
                cr.fillCmd.SetRenderTarget(cr.accum);
                cr.fillCmd.ClearRenderTarget(true, true, Color.clear);

                var layerState = EnsureGlobalLayerBuffers(0, ownerBodyByLocal, globalNodeByLocal, activeCount, globalMeshes, globalBaseOffsets, globalHierarchy.NormCenter, globalHierarchy.NormInvHalfExtent);
                SetupCommonGlobal(layerState, dt0, lib, activeCount, layerKernelH, globalHierarchy.NormCenter, globalHierarchy.NormInvHalfExtent, slot);
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

            var sc = SimulationController.Instance;
            if (sc != null &&
                sc.TryGetStableReadSlot(out int slot) &&
                sc.TryGetGlobalRenderBatch(out var globalHierarchy, out var globalMeshes, out var globalBaseOffsets)) {
                bool skipWire = globalHierarchy.TryGetLayerMappings(0, out _, out _, out _, out int layer0ActiveCount, out _) &&
                    layer0ActiveCount > WireframeNodeCap;

                if (!skipWire) {
                    int maxLayer = globalHierarchy.MaxLayer;
                    for (int layer = 0; layer <= maxLayer; layer++) {
                        if (layer != 0 && !drawCoarseLayers) continue;

                        if (!globalHierarchy.TryGetLayerDt(layer, out var dt) || dt == null || dt.TriCount <= 0)
                            continue;

                        if (!globalHierarchy.TryGetLayerMappings(layer, out int[] ownerBodyByLocal, out int[] globalNodeByLocal, out _, out int activeCount, out _))
                            continue;
                        if (activeCount <= 0)
                            continue;

                        if (!globalHierarchy.TryGetLayerExecutionContext(layer, out _, out _, out float layerKernelH))
                            continue;

                        float t = maxLayer <= 0 ? 0f : (float)layer / maxLayer;
                        Color wireColor = Color.Lerp(wireColorLayer0, wireColorMaxLayer, t);

                        var layerState = EnsureGlobalLayerBuffers(layer, ownerBodyByLocal, globalNodeByLocal, activeCount, globalMeshes, globalBaseOffsets, globalHierarchy.NormCenter, globalHierarchy.NormInvHalfExtent);
                        SetupCommonGlobal(layerState, dt, lib, activeCount, layerKernelH, globalHierarchy.NormCenter, globalHierarchy.NormInvHalfExtent, slot);
                        mpb.SetColor("_WireColor", wireColor);
                        mpb.SetFloat("_WireWidthPx", wireWidthPixels);
                        mpb.SetInt(ID_WireCullOverstretched, cullOverstretchedWireEdges ? 1 : 0);

                        int vertexCount = dt.TriCount * 3 * 6;
                        cr.wireCmd.DrawProcedural(Matrix4x4.identity, wireMaterial, 0, MeshTopology.Triangles, vertexCount, 1, mpb);
                    }
                }
            }
        } else {
            cr.wireCmd.Clear();
        }
    }

    void OnRenderImage(RenderTexture src, RenderTexture dest) {
        if (!drawLayer0Fill || compositeShader == null || sdfCompute == null || cr?.accum == null) {
            Graphics.Blit(src, dest);
            return;
        }

        RenderTexture sdfTex = RunSdfAndSmooth();

        compositeMaterial ??= new Material(compositeShader);
        compositeMaterial.SetTexture(ID_AccumTex, cr.accum);
        compositeMaterial.SetTexture(ID_SdfTex, sdfTex);
        compositeMaterial.SetFloat(ID_RoundPixels, roundPixels);
        Graphics.Blit(src, dest, compositeMaterial, 0);
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

    static void Dispatch(ComputeShader shader, int kernel, int x, int y, int z, string marker) {
        Profiler.BeginSample(marker);
        shader.Dispatch(kernel, x, y, z);
        Profiler.EndSample();
    }

    RenderTexture RunSdfAndSmooth() {
        int tgx = (cr.w + 7) >> 3;
        int tgy = (cr.h + 7) >> 3;

        sdfCompute.SetInts(ID_CSDim, cr.w, cr.h);

        sdfCompute.SetTexture(kInitKernel, ID_CSTexAccum, cr.accum);
        sdfCompute.SetTexture(kInitKernel, ID_CSTexSeedOut, cr.seedA);
        Dispatch(sdfCompute, kInitKernel, tgx, tgy, 1, "XPBI.Renderer.SdfInit");

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
            Dispatch(sdfCompute, kJfaKernel, tgx, tgy, 1, "XPBI.Renderer.SdfJfa");

            var tmp = seedIn;
            seedIn = seedOut;
            seedOut = tmp;

            jump >>= 1;
        }

        sdfCompute.SetTexture(kFinalizeKernel, ID_CSTexAccum, cr.accum);
        sdfCompute.SetTexture(kFinalizeKernel, ID_CSTexSeedIn, seedIn);
        sdfCompute.SetTexture(kFinalizeKernel, ID_CSTexSdfOut, cr.sdf);
        Dispatch(sdfCompute, kFinalizeKernel, tgx, tgy, 1, "XPBI.Renderer.SdfFinalize");

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
                Dispatch(sdfCompute, kBlurHKernel, tgx, tgy, 1, "XPBI.Renderer.SdfBlurH");

                sdfCompute.SetTexture(kBlurVKernel, ID_CSTexSdfIn, b);
                sdfCompute.SetTexture(kBlurVKernel, ID_CSTexSdfOut, a);
                Dispatch(sdfCompute, kBlurVKernel, tgx, tgy, 1, "XPBI.Renderer.SdfBlurV");
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
                Dispatch(sdfCompute, kCurvatureKernel, tgx, tgy, 1, "XPBI.Renderer.SdfCurvature");

                var tmp = a;
                a = b;
                b = tmp;
            }

            if (a != cr.sdf)
                Graphics.Blit(a, cr.sdf);

            return cr.sdf;
        }
    }

    GlobalLayerState EnsureGlobalLayerBuffers(
        int layer,
        int[] ownerBodyByLocal,
        int[] globalNodeByLocal,
        int activeCount,
        IReadOnlyList<Meshless> meshes,
        IReadOnlyList<int> baseOffsets,
        float2 normCenter,
        float normInvHalfExtent
    ) {
        if (!layerStates.TryGetValue(layer, out var st)) {
            st = new GlobalLayerState();
            layerStates[layer] = st;
        }

        bool reallocated = st.capacity != activeCount || st.ownerByLocal == null || st.materialIds == null || st.restNorm == null;
        if (reallocated) {
            st.ownerByLocal?.Dispose();
            st.materialIds?.Dispose();
            st.restVolumes?.Dispose();
            st.restNorm?.Dispose();

            st.ownerByLocal = new ComputeBuffer(activeCount, sizeof(int), ComputeBufferType.Structured);
            st.ownerByLocalCpu = new int[activeCount];

            st.materialIds = new ComputeBuffer(activeCount, sizeof(int), ComputeBufferType.Structured);
            st.materialIdsCpu = new int[activeCount];

            st.restVolumes = new ComputeBuffer(activeCount, sizeof(float), ComputeBufferType.Structured);
            st.restVolumesCpu = new float[activeCount];

            st.restNorm = new ComputeBuffer(activeCount, sizeof(float) * 2, ComputeBufferType.Structured);
            st.restNormCpu = new Vector2[activeCount];

            st.capacity = activeCount;
            st.lastCenter = new float2(float.NaN, float.NaN);
            st.lastInvHalfExtent = float.NaN;
        }

        bool ownerChanged = reallocated;
        bool materialIdsChanged = reallocated;
        bool restVolumesChanged = reallocated;
        bool restNormChanged = reallocated || !math.all(st.lastCenter == normCenter) || st.lastInvHalfExtent != normInvHalfExtent;
        int meshHint = 0;

        for (int li = 0; li < activeCount; li++) {
            int owner = (ownerBodyByLocal != null && li < ownerBodyByLocal.Length) ? ownerBodyByLocal[li] : -1;
            if (st.ownerByLocalCpu[li] != owner) {
                st.ownerByLocalCpu[li] = owner;
                ownerChanged = true;
            }

            int gi = (globalNodeByLocal != null && li < globalNodeByLocal.Length) ? globalNodeByLocal[li] : -1;
            if (!TryResolveGlobalNode(gi, meshes, baseOffsets, ref meshHint, out Node node))
                continue;

            int materialId = node.materialId;
            if (st.materialIdsCpu[li] != materialId) {
                st.materialIdsCpu[li] = materialId;
                materialIdsChanged = true;
            }

            float restVolume = node.restVolume;
            if (!Mathf.Approximately(st.restVolumesCpu[li], restVolume)) {
                st.restVolumesCpu[li] = restVolume;
                restVolumesChanged = true;
            }

            if (restNormChanged) {
                float2 norm = (node.originalPos - normCenter) * normInvHalfExtent;
                st.restNormCpu[li] = new Vector2(norm.x, norm.y);
            }
        }

        if (ownerChanged)
            st.ownerByLocal.SetData(st.ownerByLocalCpu, 0, 0, activeCount);
        if (materialIdsChanged)
            st.materialIds.SetData(st.materialIdsCpu, 0, 0, activeCount);
        if (restVolumesChanged)
            st.restVolumes.SetData(st.restVolumesCpu, 0, 0, activeCount);
        if (restNormChanged) {
            st.restNorm.SetData(st.restNormCpu, 0, 0, activeCount);
            st.lastCenter = normCenter;
            st.lastInvHalfExtent = normInvHalfExtent;
        }

        return st;
    }

    static bool TryResolveGlobalNode(
        int globalNode,
        IReadOnlyList<Meshless> meshes,
        IReadOnlyList<int> baseOffsets,
        ref int meshHint,
        out Node node
    ) {
        node = default;

        if (globalNode < 0 || meshes == null || baseOffsets == null || meshes.Count != baseOffsets.Count)
            return false;

        int meshCount = meshes.Count;
        if (meshCount <= 0)
            return false;

        if (meshHint < 0 || meshHint >= meshCount)
            meshHint = 0;

        if (TryResolveGlobalNodeAt(globalNode, meshes, baseOffsets, meshHint, out node))
            return true;

        int lo = 0;
        int hi = meshCount - 1;
        while (lo <= hi) {
            int mid = lo + ((hi - lo) >> 1);
            int baseIndex = baseOffsets[mid];
            int nextBase = (mid + 1 < meshCount) ? baseOffsets[mid + 1] : int.MaxValue;

            if (globalNode < baseIndex) {
                hi = mid - 1;
                continue;
            }

            if (globalNode >= nextBase) {
                lo = mid + 1;
                continue;
            }

            meshHint = mid;
            return TryResolveGlobalNodeAt(globalNode, meshes, baseOffsets, mid, out node);
        }

        return false;
    }

    static bool TryResolveGlobalNodeAt(
        int globalNode,
        IReadOnlyList<Meshless> meshes,
        IReadOnlyList<int> baseOffsets,
        int meshIndex,
        out Node node
    ) {
        node = default;

        if (meshIndex < 0 || meshIndex >= meshes.Count)
            return false;

        Meshless m = meshes[meshIndex];
        if (m == null || m.nodes == null || m.nodes.Count <= 0)
            return false;

        int baseIndex = baseOffsets[meshIndex];
        int local = globalNode - baseIndex;
        if (local < 0 || local >= m.nodes.Count)
            return false;

        node = m.nodes[local];
        return true;
    }

    void SetupCommonGlobal(
        GlobalLayerState st,
        DT dt,
        MaterialLibrary lib,
        int realPointCount,
        float layerKernelH,
        float2 normCenter,
        float normInvHalfExtent,
        int slot
    ) {
        if (st == null || st.ownerByLocal == null || st.materialIds == null || st.restNorm == null)
            return;

        float alpha = 0f;
        var sc = SimulationController.Instance;
        if (sc != null) alpha = sc.RenderAlpha;

        var positions = dt.GetPositionsBuffer(slot);
        var halfEdges = dt.GetHalfEdgesBuffer(slot);
        var triToHe = dt.GetTriToHEBuffer(slot);
        if (positions == null || halfEdges == null || triToHe == null)
            return;

        mpb.Clear();

        if (lib != null && lib.AlbedoArray != null) {
            mpb.SetTexture(ID_AlbedoArray, lib.AlbedoArray);
            mpb.SetInt(ID_MaterialCount, lib.MaterialCount);
        }

        mpb.SetBuffer(ID_PositionsPrev, positions);
        mpb.SetBuffer(ID_PositionsCurr, positions);
        mpb.SetFloat(ID_RenderAlpha, alpha);

        mpb.SetBuffer(ID_Positions, positions);
        mpb.SetBuffer(ID_HalfEdges, halfEdges);
        mpb.SetBuffer(ID_TriToHE, triToHe);
        mpb.SetBuffer(ID_OwnerByLocal, st.ownerByLocal);

        mpb.SetBuffer(ID_MaterialIds, st.materialIds);
        mpb.SetBuffer(ID_RestVolumes, st.restVolumes);
        mpb.SetBuffer(ID_RestNormPositions, st.restNorm);

        float layerKernelHNorm = layerKernelH * normInvHalfExtent;
        mpb.SetFloat(ID_LayerKernelH, layerKernelHNorm);
        mpb.SetFloat(ID_WendlandSupportScale, Const.WendlandSupport);

        mpb.SetInt(ID_RealPointCount, realPointCount);

        mpb.SetVector(ID_NormCenter, new Vector4(normCenter.x, normCenter.y, 0f, 0f));
        mpb.SetFloat(ID_NormInvHalfExtent, normInvHalfExtent);
    }
}
