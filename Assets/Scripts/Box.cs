using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

public class Box : Meshless {
    public float2 size;

    [Header("Generator")]
    public int pointCount;
    public bool generateOnStart = false;

    [Header("Hierarchy")]
    [Range(0.01f, 0.99f)] public float layerRatio = 0.10f;
    [Min(0.01f)] public float radiusScale = 0.85f;
    [Min(1)] public int poissonK = 30;

    [HideInInspector]
    public int[] levelNodeCounts;
    void Start() {
        if (generateOnStart) Generate(pointCount, 0);
    }

    public void Generate(int count, short material) {
        GenerateHierarchy(count);
        if (nodes.Count > 0) {
            int cornerIdx = 0;
            float maxX = nodes[0].pos.x;
            float maxY = nodes[0].pos.y;
            for (int i = 1; i < nodes.Count; i++) {
                var pos = nodes[i].pos;
                if (pos.x > maxX || (pos.x == maxX && pos.y > maxY)) {
                    maxX = pos.x;
                    maxY = pos.y;
                    cornerIdx = i;
                }
            }
            FixNode(cornerIdx);
        }

        Build();
    }

    void GenerateHierarchy(int randomPointCount) {
        if (maxLayer < 0) maxLayer = 0;

        var half = size * 0.5f;
        var min = new float2(transform.position.x, transform.position.y) - half;
        var max = new float2(transform.position.x, transform.position.y) + half;
        var area = math.max(1e-12f, size.x * size.y);

        int layerCount = maxLayer + 1;

        // Counts: level 0 total is fixed: random points + 4 corners.
        // Higher levels are computed only from the random points (corners do not propagate up).
        var perLevelRandom = ComputeStrictlyDecreasingCounts(
            math.max(0, randomPointCount),
            layerCount,
            layerRatio
        );

        levelNodeCounts = new int[layerCount];
        for (int l = 0; l < layerCount; l++)
            levelNodeCounts[l] = perLevelRandom[l];

        layerRadii = new float[layerCount];

        // Radii (initial guess). Level 0 uses total (random + fixed) because corners constrain it.
        layerRadii[0] = radiusScale * math.sqrt(area / math.max(1, levelNodeCounts[0]));
        for (int l = 1; l < layerCount; l++)
            layerRadii[l] = radiusScale * math.sqrt(area / math.max(1, levelNodeCounts[l]));

        // Ensure non-decreasing radii with layer index (coarser layers should have >= radius).
        for (int l = 1; l < layerCount; l++)
            layerRadii[l] = math.max(layerRadii[l], layerRadii[l - 1] * 1.01f);

        // Generate shared nodes (excluding fixed points) from coarse -> fine.
        uint seed = (uint)Guid.NewGuid().GetHashCode();
        if (seed == 0) seed = 1;
        var rnd = new Unity.Mathematics.Random(seed);

        var pts = new List<float2>(perLevelRandom[0]);
        var ptsMaxLayer = new List<short>(perLevelRandom[0]);

        // Start at top level (coarsest), then progressively densify down to level 0.
        for (int l = maxLayer; l >= 0; l--) {
            int target = perLevelRandom[l];
            if (target == 0) continue;

            float radius = layerRadii[l];
            radius = FillPoissonToTarget(
                ref rnd,
                min,
                max,
                radius,
                target,
                (short)l,
                poissonK,
                pts,
                ptsMaxLayer
            );

            layerRadii[l] = radius;
        }

        // Commit into Meshless.nodes.
        nodes.Clear();

        for (int i = 0; i < pts.Count; i++)
            AddAndSetMaxLayer(pts[i], ptsMaxLayer[i]);
    }

    int AddAndSetMaxLayer(float2 p, short maxLayer) {
        Add(p);
        int idx = nodes.Count - 1;

        // Works whether Node is a struct or class (struct needs copy-back).
        var n = nodes[idx];
        n.maxLayer = maxLayer;
        nodes[idx] = n;

        return idx;
    }

    static int[] ComputeStrictlyDecreasingCounts(int baseCount, int levels, float ratio) {
        var counts = new int[levels];
        counts[0] = baseCount;

        for (int l = 1; l < levels; l++) {
            int next = (int)math.round(counts[l - 1] * ratio);
            next = math.max(1, next);

            if (counts[l - 1] > 1)
                next = math.min(next, counts[l - 1] - 1);

            counts[l] = next;
        }

        // If baseCount is 0, keep all levels at 0 (except we don't want negative).
        if (baseCount == 0) {
            counts[0] = 0;
            for (int l = 1; l < levels; l++) counts[l] = 0;
        }

        return counts;
    }

    static float FillPoissonToTarget(
        ref Unity.Mathematics.Random rnd,
        float2 min,
        float2 max,
        float initialRadius,
        int targetCount,
        short newPointsMaxLayer,
        int k,
        List<float2> points,
        List<short> pointsMaxLayer
    ) {
        float radius = math.max(1e-7f, initialRadius);

        // If we already have enough points for this layer, do nothing.
        if (points.Count >= targetCount) return radius;

        // Shrink radius as needed to hit exact target count.
        for (int radiusAdjust = 0; radiusAdjust < 32 && points.Count < targetCount; radiusAdjust++) {
            float cellSize = radius * 0.70710678f * 0.999f; // ~ r/sqrt(2), slightly smaller to enforce 1 point/cell.
            float invCellSize = 1f / math.max(1e-12f, cellSize);

            int gridW = math.max(1, (int)math.ceil((max.x - min.x) * invCellSize));
            int gridH = math.max(1, (int)math.ceil((max.y - min.y) * invCellSize));
            var grid = new int[gridW * gridH];
            Array.Fill(grid, -1);

            var active = new List<int>(math.max(16, points.Count));

            // Insert existing points into grid + active list.
            for (int i = 0; i < points.Count; i++) {
                var c = ToCell(points[i], min, invCellSize);
                int gi = c.x + c.y * gridW;
                if ((uint)c.x < (uint)gridW && (uint)c.y < (uint)gridH && grid[gi] == -1) grid[gi] = i;
                active.Add(i);
            }

            // Seed if empty.
            if (points.Count == 0) {
                if (TryAddRandomSeed(ref rnd, min, max, radius, points, pointsMaxLayer, newPointsMaxLayer, grid, gridW, gridH, invCellSize))
                    active.Add(points.Count - 1);
            }

            float radiusSq = radius * radius;

            while (points.Count < targetCount) {
                if (active.Count == 0) {
                    if (TryAddRandomSeed(ref rnd, min, max, radius, points, pointsMaxLayer, newPointsMaxLayer, grid, gridW, gridH, invCellSize))
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
                    float rr = math.sqrt((radiusSq * (1f + 3f * u))); // sqrt(lerp(r^2, (2r)^2, u))
                    float2 cand = p + new float2(math.cos(angle), math.sin(angle)) * rr;

                    if (!Inside(cand, min, max)) continue;
                    if (!FarFromExisting(cand, points, grid, gridW, gridH, min, invCellSize, radiusSq)) continue;

                    int newIndex = points.Count;
                    points.Add(cand);
                    pointsMaxLayer.Add(newPointsMaxLayer);

                    var cell = ToCell(cand, min, invCellSize);
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

            if (points.Count >= targetCount) return radius;
            radius *= 0.90f;
        }

        return radius;
    }

    static bool TryAddRandomSeed(
        ref Unity.Mathematics.Random rnd,
        float2 min,
        float2 max,
        float radius,
        List<float2> points,
        List<short> pointsMaxLayer,
        short newPointsMaxLayer,
        int[] grid,
        int gridW,
        int gridH,
        float invCellSize
    ) {
        float radiusSq = radius * radius;

        for (int attempt = 0; attempt < 2048; attempt++) {
            float2 p = rnd.NextFloat2(min, max);
            if (!FarFromExisting(p, points, grid, gridW, gridH, min, invCellSize, radiusSq)) continue;

            int newIndex = points.Count;
            points.Add(p);
            pointsMaxLayer.Add(newPointsMaxLayer);

            var cell = ToCell(p, min, invCellSize);
            grid[cell.x + cell.y * gridW] = newIndex;

            return true;
        }

        return false;
    }

    static bool Inside(float2 p, float2 min, float2 max) =>
        p.x >= min.x && p.y >= min.y && p.x <= max.x && p.y <= max.y;

    static bool FarFromFixed(float2 p, float2[] fixedPts, float fixedRadius) {
        float r2 = fixedRadius * fixedRadius;
        for (int i = 0; i < fixedPts.Length; i++) {
            if (math.lengthsq(p - fixedPts[i]) < r2) return false;
        }
        return true;
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
        var c = ToCell(p, min, invCellSize);

        int x0 = math.max(0, c.x - 2);
        int y0 = math.max(0, c.y - 2);
        int x1 = math.min(gridW - 1, c.x + 2);
        int y1 = math.min(gridH - 1, c.y + 2);

        for (int y = y0; y <= y1; y++) {
            int row = y * gridW;
            for (int x = x0; x <= x1; x++) {
                int pi = grid[row + x];
                if (pi < 0) continue;
                if (math.lengthsq(p - points[pi]) < radiusSq) return false;
            }
        }

        return true;
    }

    static int2 ToCell(float2 p, float2 min, float invCellSize) =>
        (int2)math.floor((p - min) * invCellSize);
}
