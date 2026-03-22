using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

[RequireComponent(typeof(SpriteRenderer))]
public sealed class SpriteMesh : Box {
    static readonly Dictionary<Sprite, MaterialDef> RuntimeDefsBySprite = new Dictionary<Sprite, MaterialDef>(64);

    [Header("Sprite")]
    public SpriteRenderer spriteRenderer;
    public MaterialDef physicsTemplate;
    public bool generateOnAwakeIfMissing = true;

    public override bool UsesSpriteUv => true;

    void Reset() {
        spriteRenderer = GetComponent<SpriteRenderer>();
        defaultAnchorMode = AnchorMode.None;
    }

    void Awake() {
        if (spriteRenderer == null)
            spriteRenderer = GetComponent<SpriteRenderer>();

        if (generateOnAwakeIfMissing && (nodes == null || nodes.Count == 0))
            Generate(0, 0);
    }

    public override void Generate(int count, short material) {
        if (spriteRenderer == null)
            spriteRenderer = GetComponent<SpriteRenderer>();
        if (spriteRenderer == null || spriteRenderer.sprite == null)
            return;

        size = new float2(
            Mathf.Max(1e-4f, spriteRenderer.bounds.size.x),
            Mathf.Max(1e-4f, spriteRenderer.bounds.size.y)
        );

        EnsureRuntimeMaterial();

        if (!GenerateFromSpriteShape()) {
            base.Generate(count, material);
            BakeSpriteUvs();
            return;
        }

        if (defaultAnchorMode == AnchorMode.TopRightCorner)
            FixTopRightCorner();

        Build();

        BakeSpriteUvs();
    }

    bool GenerateFromSpriteShape() {
        Sprite sprite = spriteRenderer != null ? spriteRenderer.sprite : null;
        if (sprite == null)
            return false;

        Vector2[] localVerts2 = sprite.vertices;
        ushort[] tris = sprite.triangles;
        if (localVerts2 == null || localVerts2.Length < 3 || tris == null || tris.Length < 3)
            return false;

        bool flipX = spriteRenderer.flipX;
        bool flipY = spriteRenderer.flipY;
        Transform tr = spriteRenderer.transform;

        var worldVerts = new float2[localVerts2.Length];
        float2 min = new float2(float.PositiveInfinity, float.PositiveInfinity);
        float2 max = new float2(float.NegativeInfinity, float.NegativeInfinity);

        for (int i = 0; i < localVerts2.Length; i++) {
            Vector2 v = localVerts2[i];
            if (flipX) v.x = -v.x;
            if (flipY) v.y = -v.y;

            Vector3 w = tr.TransformPoint(new Vector3(v.x, v.y, 0f));
            float2 p = new float2(w.x, w.y);
            worldVerts[i] = p;
            min = math.min(min, p);
            max = math.max(max, p);
        }

        int triCount = tris.Length / 3;
        var triA = new float2[triCount];
        var triB = new float2[triCount];
        var triC = new float2[triCount];
        var triCdf = new float[triCount];

        float totalArea = 0f;
        int usedTriCount = 0;
        for (int t = 0; t < triCount; t++) {
            int i0 = tris[t * 3 + 0];
            int i1 = tris[t * 3 + 1];
            int i2 = tris[t * 3 + 2];

            if ((uint)i0 >= (uint)worldVerts.Length || (uint)i1 >= (uint)worldVerts.Length || (uint)i2 >= (uint)worldVerts.Length)
                continue;

            float2 a = worldVerts[i0];
            float2 b = worldVerts[i1];
            float2 c = worldVerts[i2];
            float area = 0.5f * math.abs((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
            if (area <= 1e-10f)
                continue;

            totalArea += area;
            triA[usedTriCount] = a;
            triB[usedTriCount] = b;
            triC[usedTriCount] = c;
            triCdf[usedTriCount] = totalArea;
            usedTriCount++;
        }

        if (usedTriCount <= 0 || totalArea <= 1e-10f)
            return false;

        if (usedTriCount < triCount) {
            System.Array.Resize(ref triA, usedTriCount);
            System.Array.Resize(ref triB, usedTriCount);
            System.Array.Resize(ref triC, usedTriCount);
            System.Array.Resize(ref triCdf, usedTriCount);
        }

        int layer0Count = ComputeLayer0PointCountFromArea(totalArea);
        int[] perLayer = ComputeAutoLayerCountsLocal(
            layer0Count,
            Const.LayerDownsampleRatio,
            Const.MinVerticesPerLayer,
            Const.MaxAutoLayers
        );

        int layerCount = perLayer.Length;
        maxLayer = layerCount - 1;
        generatedLayer0PointCount = layer0Count;
        generatedLayerCount = layerCount;

        layerNodeCounts = new int[layerCount];
        for (int l = 0; l < layerCount; l++)
            layerNodeCounts[l] = perLayer[l];

        layerRadii = new float[layerCount];
        layerKernelH = new float[layerCount];

        layerRadii[0] = Const.PoissonRadiusScale * math.sqrt(totalArea / math.max(1, layerNodeCounts[0]));
        for (int l = 1; l < layerCount; l++)
            layerRadii[l] = Const.PoissonRadiusScale * math.sqrt(totalArea / math.max(1, layerNodeCounts[l]));

        for (int l = 1; l < layerCount; l++)
            layerRadii[l] = math.max(layerRadii[l], layerRadii[l - 1] * 1.01f);

        uint rndSeed = seed == 0 ? 1u : seed;
        var rnd = new Unity.Mathematics.Random(rndSeed);

        var points = new List<float2>(perLayer[0]);
        var pointsMaxLayer = new List<short>(perLayer[0]);

        for (int l = maxLayer; l >= 0; l--) {
            int target = perLayer[l];
            if (target <= 0)
                continue;

            float radius = layerRadii[l];
            radius = FillPoissonToTargetOnSprite(
                ref rnd,
                min,
                max,
                triA,
                triB,
                triC,
                triCdf,
                totalArea,
                radius,
                target,
                (short)l,
                Const.PoissonK,
                points,
                pointsMaxLayer
            );

            layerRadii[l] = radius;
        }

        for (int l = 0; l < layerCount; l++)
            layerKernelH[l] = math.max(layerRadii[l] * Const.LayerKernelHFromPoissonRadius, 1e-4f);

        nodes.Clear();
        for (int i = 0; i < points.Count; i++) {
            Add(points[i]);
            int idx = nodes.Count - 1;
            var node = nodes[idx];
            node.maxLayer = pointsMaxLayer[i];
            nodes[idx] = node;
        }

        generatedLayer0PointCount = points.Count;
        return points.Count > 0;
    }

    static int ComputeLayer0PointCountFromArea(float area) {
        int densityTarget = (int)math.round(area * Const.Layer0PointDensity);
        return math.max(Const.MinVerticesPerLayer, densityTarget);
    }

    static int[] ComputeAutoLayerCountsLocal(int layer0Count, float ratio, int minPerLayer, int maxLayers) {
        int clampedMinPerLayer = math.max(1, minPerLayer);
        int clampedLayer0Count = math.max(clampedMinPerLayer, layer0Count);
        int clampedMaxLayers = math.max(1, maxLayers);
        float clampedRatio = math.clamp(ratio, 0.01f, 0.99f);

        var counts = new List<int>(clampedMaxLayers) { clampedLayer0Count };
        for (int layer = 1; layer < clampedMaxLayers; layer++) {
            int prev = counts[layer - 1];
            int next = (int)math.floor(prev * clampedRatio);
            if (next >= prev)
                next = prev - 1;
            if (next < clampedMinPerLayer)
                break;
            counts.Add(next);
        }

        return counts.ToArray();
    }

    static float FillPoissonToTargetOnSprite(
        ref Unity.Mathematics.Random rnd,
        float2 min,
        float2 max,
        float2[] triA,
        float2[] triB,
        float2[] triC,
        float[] triCdf,
        float triAreaTotal,
        float initialRadius,
        int targetCount,
        short newPointsMaxLayer,
        int k,
        List<float2> points,
        List<short> pointsMaxLayer
    ) {
        float radius = math.max(1e-7f, initialRadius);
        if (points.Count >= targetCount)
            return radius;

        for (int radiusAdjust = 0; radiusAdjust < 32 && points.Count < targetCount; radiusAdjust++) {
            float cellSize = radius * 0.70710678f * 0.999f;
            float invCellSize = 1f / math.max(1e-12f, cellSize);

            int gridW = math.max(1, (int)math.ceil((max.x - min.x) * invCellSize));
            int gridH = math.max(1, (int)math.ceil((max.y - min.y) * invCellSize));
            var grid = new int[gridW * gridH];
            System.Array.Fill(grid, -1);

            var active = new List<int>(math.max(16, points.Count));

            for (int i = 0; i < points.Count; i++) {
                int2 c = ToCell(points[i], min, invCellSize);
                int gi = c.x + c.y * gridW;
                if ((uint)c.x < (uint)gridW && (uint)c.y < (uint)gridH && grid[gi] == -1)
                    grid[gi] = i;
                active.Add(i);
            }

            if (points.Count == 0) {
                if (TryAddRandomSeedOnSprite(ref rnd, triA, triB, triC, triCdf, triAreaTotal, radius, points, pointsMaxLayer, newPointsMaxLayer, grid, gridW, gridH, min, invCellSize))
                    active.Add(points.Count - 1);
            }

            float radiusSq = radius * radius;

            while (points.Count < targetCount) {
                if (active.Count == 0) {
                    if (TryAddRandomSeedOnSprite(ref rnd, triA, triB, triC, triCdf, triAreaTotal, radius, points, pointsMaxLayer, newPointsMaxLayer, grid, gridW, gridH, min, invCellSize))
                        active.Add(points.Count - 1);
                    else
                        break;
                }

                int a = rnd.NextInt(0, active.Count);
                int idx = active[a];
                float2 p = points[idx];

                bool found = false;
                for (int t = 0; t < k; t++) {
                    float angle = rnd.NextFloat(0f, 6.28318530718f);
                    float u = rnd.NextFloat();
                    float rr = math.sqrt(radiusSq * (1f + 3f * u));
                    float2 cand = p + new float2(math.cos(angle), math.sin(angle)) * rr;

                    if (!InsideTriangleSoup(cand, triA, triB, triC))
                        continue;
                    if (!FarFromExisting(cand, points, grid, gridW, gridH, min, invCellSize, radiusSq))
                        continue;

                    int newIndex = points.Count;
                    points.Add(cand);
                    pointsMaxLayer.Add(newPointsMaxLayer);

                    int2 cell = ToCell(cand, min, invCellSize);
                    grid[cell.x + cell.y * gridW] = newIndex;
                    active.Add(newIndex);

                    found = true;
                    break;
                }

                if (!found) {
                    int last = active.Count - 1;
                    active[a] = active[last];
                    active.RemoveAt(last);
                }
            }

            if (points.Count >= targetCount)
                return radius;

            radius *= 0.90f;
        }

        return radius;
    }

    static bool TryAddRandomSeedOnSprite(
        ref Unity.Mathematics.Random rnd,
        float2[] triA,
        float2[] triB,
        float2[] triC,
        float[] triCdf,
        float triAreaTotal,
        float radius,
        List<float2> points,
        List<short> pointsMaxLayer,
        short newPointsMaxLayer,
        int[] grid,
        int gridW,
        int gridH,
        float2 min,
        float invCellSize
    ) {
        float radiusSq = radius * radius;

        for (int attempt = 0; attempt < 2048; attempt++) {
            float2 p = SamplePointInTriangleSoup(ref rnd, triA, triB, triC, triCdf, triAreaTotal);
            if (!FarFromExisting(p, points, grid, gridW, gridH, min, invCellSize, radiusSq))
                continue;

            int newIndex = points.Count;
            points.Add(p);
            pointsMaxLayer.Add(newPointsMaxLayer);

            int2 cell = ToCell(p, min, invCellSize);
            if ((uint)cell.x < (uint)gridW && (uint)cell.y < (uint)gridH)
                grid[cell.x + cell.y * gridW] = newIndex;

            return true;
        }

        return false;
    }

    static float2 SamplePointInTriangleSoup(
        ref Unity.Mathematics.Random rnd,
        float2[] triA,
        float2[] triB,
        float2[] triC,
        float[] triCdf,
        float triAreaTotal
    ) {
        float pick = rnd.NextFloat(0f, triAreaTotal);

        int tri = System.Array.BinarySearch(triCdf, pick);
        if (tri < 0)
            tri = ~tri;
        tri = math.clamp(tri, 0, triCdf.Length - 1);

        float2 a = triA[tri];
        float2 b = triB[tri];
        float2 c = triC[tri];

        float u = rnd.NextFloat();
        float v = rnd.NextFloat();
        float su = math.sqrt(u);

        float w0 = 1f - su;
        float w1 = su * (1f - v);
        float w2 = su * v;

        return a * w0 + b * w1 + c * w2;
    }

    static bool InsideTriangleSoup(float2 p, float2[] triA, float2[] triB, float2[] triC) {
        for (int i = 0; i < triA.Length; i++) {
            if (PointInTriangle(p, triA[i], triB[i], triC[i]))
                return true;
        }
        return false;
    }

    static bool PointInTriangle(float2 p, float2 a, float2 b, float2 c) {
        float2 v0 = c - a;
        float2 v1 = b - a;
        float2 v2 = p - a;

        float dot00 = math.dot(v0, v0);
        float dot01 = math.dot(v0, v1);
        float dot02 = math.dot(v0, v2);
        float dot11 = math.dot(v1, v1);
        float dot12 = math.dot(v1, v2);

        float denom = dot00 * dot11 - dot01 * dot01;
        if (math.abs(denom) <= 1e-12f)
            return false;

        float inv = 1f / denom;
        float u = (dot11 * dot02 - dot01 * dot12) * inv;
        float v = (dot00 * dot12 - dot01 * dot02) * inv;

        return u >= -1e-5f && v >= -1e-5f && (u + v) <= 1.00001f;
    }

    static bool FarFromExisting(
        float2 p,
        List<float2> points,
        int[] grid,
        int gridW,
        int gridH,
        float2 min,
        float invCellSize,
        float radiusSq
    ) {
        int2 c = ToCell(p, min, invCellSize);

        int x0 = math.max(0, c.x - 2);
        int y0 = math.max(0, c.y - 2);
        int x1 = math.min(gridW - 1, c.x + 2);
        int y1 = math.min(gridH - 1, c.y + 2);

        for (int y = y0; y <= y1; y++) {
            int row = y * gridW;
            for (int x = x0; x <= x1; x++) {
                int pi = grid[row + x];
                if (pi < 0)
                    continue;
                if (math.lengthsq(p - points[pi]) < radiusSq)
                    return false;
            }
        }

        return true;
    }

    static int2 ToCell(float2 p, float2 min, float invCellSize) {
        return (int2)math.floor((p - min) * invCellSize);
    }

    void EnsureRuntimeMaterial() {
        var lib = MaterialLibrary.Instance;
        var sprite = spriteRenderer != null ? spriteRenderer.sprite : null;
        if (lib == null || sprite == null)
            return;

        if (!RuntimeDefsBySprite.TryGetValue(sprite, out MaterialDef def) || def == null) {
            def = ScriptableObject.CreateInstance<MaterialDef>();
            def.name = "Runtime_Sprite_" + sprite.name;
            def.sprite = sprite;

            MaterialDef source = physicsTemplate != null ? physicsTemplate : baseMaterialDef;
            if (source == null && lib.materials != null && lib.materials.Length > 0)
                source = lib.materials[0];

            if (source != null)
                def.physical = source.physical;

            lib.AddRuntimeMaterial(def);
            RuntimeDefsBySprite[sprite] = def;
        }

        baseMaterialDef = def;
    }

    void BakeSpriteUvs() {
        if (spriteRenderer == null || spriteRenderer.sprite == null || nodes == null || nodes.Count == 0)
            return;

        Bounds localBounds = spriteRenderer.sprite.bounds;
        Vector3 min = localBounds.min;
        Vector3 size3 = localBounds.size;

        float invW = size3.x > 1e-6f ? 1f / size3.x : 0f;
        float invH = size3.y > 1e-6f ? 1f / size3.y : 0f;

        bool flipX = spriteRenderer.flipX;
        bool flipY = spriteRenderer.flipY;

        Transform tr = spriteRenderer.transform;
        for (int i = 0; i < nodes.Count; i++) {
            Node node = nodes[i];
            Vector3 local = tr.InverseTransformPoint(new Vector3(node.originalPos.x, node.originalPos.y, 0f));

            float u = (local.x - min.x) * invW;
            float v = (local.y - min.y) * invH;

            if (flipX) u = 1f - u;
            if (flipY) v = 1f - v;

            node.materialUv = new float2(math.saturate(u), math.saturate(v));
            nodes[i] = node;
        }
    }
}
