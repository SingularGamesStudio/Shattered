using System.Collections.Generic;
using UnityEngine;

[DefaultExecutionOrder(-2000)]
public sealed class MaterialLibrary : MonoBehaviour {
    public static MaterialLibrary Instance { get; private set; }

    [Header("Material defs")]
    public MaterialDef[] materials;

    [Header("Fallback (used when material sprite is null)")]
    public Texture2D fallbackTexture;

    Texture2DArray albedoArray;
    ComputeBuffer physicalParams;

    Dictionary<MaterialDef, int> defToIndex;

    public Texture2DArray AlbedoArray => albedoArray;
    public ComputeBuffer PhysicalParamsBuffer => physicalParams;
    public int MaterialCount => materials != null ? materials.Length : 0;

    public float GetDensityByIndex(int materialIndex) {
        if (materials == null || materials.Length == 0)
            return 1f;

        if (materialIndex < 0 || materialIndex >= materials.Length)
            return 1f;

        var def = materials[materialIndex];
        if (def == null)
            return 1f;

        return Mathf.Max(0.0001f, def.physical.density);
    }

    public int AddRuntimeMaterial(MaterialDef def) {
        if (def == null)
            throw new System.ArgumentNullException(nameof(def));

        int oldCount = materials != null ? materials.Length : 0;
        var next = new MaterialDef[oldCount + 1];
        for (int i = 0; i < oldCount; i++)
            next[i] = materials[i];

        next[oldCount] = def;
        materials = next;
        Rebuild();
        return oldCount;
    }

    struct MaterialGpu {
        public Vector4 p0; // young, poisson, yieldHencky, volumetricHenckyLimit
    }

    void Awake() {
        if (Instance != null && Instance != this) { Destroy(gameObject); return; }
        Instance = this;
        Rebuild();
    }

    void OnDestroy() {
        if (Instance == this) Instance = null;
        physicalParams?.Dispose();
        physicalParams = null;
    }

    public int GetMaterialIndex(MaterialDef def) {
        if (def == null) return 0;
        defToIndex ??= new Dictionary<MaterialDef, int>(64);
        if (!defToIndex.TryGetValue(def, out int idx)) {
            Debug.LogWarning($"MeshlessMaterialLibrary: material def '{def.name}' is not in MeshlessMaterialLibrary.materials; returning 0.");
            return 0;
        }
        return idx;
    }

    static bool TryGetSliceSize(MaterialDef[] defs, out int w, out int h) {
        w = 0;
        h = 0;

        if (defs == null) return false;

        for (int i = 0; i < defs.Length; i++) {
            var s = defs[i] != null ? defs[i].sprite : null;
            if (s == null) continue;

            Rect r = s.textureRect;
            w = Mathf.RoundToInt(r.width);
            h = Mathf.RoundToInt(r.height);
            return w > 0 && h > 0;
        }

        return false;
    }

    static Color32[] MakeSolid(int w, int h, Color32 c) {
        var arr = new Color32[w * h];
        for (int i = 0; i < arr.Length; i++) arr[i] = c;
        return arr;
    }

    [ContextMenu("Rebuild")]
    public void Rebuild() {
        defToIndex ??= new Dictionary<MaterialDef, int>(64);
        defToIndex.Clear();

        physicalParams?.Dispose();
        physicalParams = null;

        if (materials == null || materials.Length == 0) {
            albedoArray = null;
            return;
        }

        int sliceW, sliceH;
        if (!TryGetSliceSize(materials, out sliceW, out sliceH)) {
            if (fallbackTexture == null) { albedoArray = null; return; }
            sliceW = fallbackTexture.width;
            sliceH = fallbackTexture.height;
        }

        // RGBA32 so we can always write the readback pixels.
        // (This avoids platform graphicsFormat support problems you hit earlier.)
        bool linear = false; // store as sRGB-like for albedo; shader sampling will be fine for visualization.
        albedoArray = new Texture2DArray(sliceW, sliceH, materials.Length, TextureFormat.RGBA32, false, linear) {
            wrapMode = TextureWrapMode.Repeat,
            filterMode = FilterMode.Bilinear
        };

        var gpu = new MaterialGpu[materials.Length];

        var rt = RenderTexture.GetTemporary(sliceW, sliceH, 0, RenderTextureFormat.ARGB32,
            QualitySettings.activeColorSpace == ColorSpace.Linear ? RenderTextureReadWrite.sRGB : RenderTextureReadWrite.Default);
        rt.filterMode = FilterMode.Bilinear;
        rt.wrapMode = TextureWrapMode.Clamp;

        var readback = new Texture2D(sliceW, sliceH, TextureFormat.RGBA32, false, false);

        Color32[] magenta = null;

        bool oldSrgbWrite = GL.sRGBWrite;
        GL.sRGBWrite = QualitySettings.activeColorSpace == ColorSpace.Linear;

        for (int i = 0; i < materials.Length; i++) {
            var def = materials[i];
            defToIndex[def] = i;

            if (def != null) {
                gpu[i] = new MaterialGpu {
                    p0 = new Vector4(def.physical.youngModulus, def.physical.poissonRatio, def.physical.yieldHencky, def.physical.volumetricHenckyLimit),
                };
            }

            Sprite s = def != null ? def.sprite : null;

            if (s != null && s.packed && s.packingRotation != SpritePackingRotation.None) {
                Debug.LogWarning($"MeshlessMaterialLibrary: Sprite '{s.name}' is packed with rotation ({s.packingRotation}). Disable rotation in packing or it will be wrong.");
            }

            Texture src = s != null ? (Texture)s.texture : (Texture)fallbackTexture;
            if (src == null) {
                magenta ??= MakeSolid(sliceW, sliceH, new Color32(255, 0, 255, 255));
                albedoArray.SetPixels32(magenta, i, 0);
                continue;
            }

            if (s == null) {
                Graphics.Blit(src, rt);
            } else {
                Rect r = s.textureRect;
                int w = Mathf.RoundToInt(r.width);
                int h = Mathf.RoundToInt(r.height);
                if (w != sliceW || h != sliceH) {
                    Debug.LogWarning($"MeshlessMaterialLibrary: Sprite '{s.name}' is {w}x{h}, expected {sliceW}x{sliceH}.");
                    magenta ??= MakeSolid(sliceW, sliceH, new Color32(255, 0, 255, 255));
                    albedoArray.SetPixels32(magenta, i, 0);
                    continue;
                }

                float invW = 1f / s.texture.width;
                float invH = 1f / s.texture.height;

                Vector2 scale = new Vector2(r.width * invW, r.height * invH);
                Vector2 offset = new Vector2(r.x * invW, r.y * invH);

                Graphics.Blit(src, rt, scale, offset);
            }

            var old = RenderTexture.active;
            RenderTexture.active = rt;
            readback.ReadPixels(new Rect(0, 0, sliceW, sliceH), 0, 0, false);
            readback.Apply(false, false);
            RenderTexture.active = old;

            // Avoid per-slice allocations: use raw texture data.
            albedoArray.SetPixelData(readback.GetRawTextureData<Color32>(), 0, i);
        }

        GL.sRGBWrite = oldSrgbWrite;

        Object.Destroy(readback);
        RenderTexture.ReleaseTemporary(rt);

        albedoArray.Apply(false, true);

        physicalParams = new ComputeBuffer(materials.Length, sizeof(float) * 4, ComputeBufferType.Structured);
        physicalParams.SetData(gpu);
    }
}
