using Unity.Mathematics;
using UnityEngine;

[ExecuteInEditMode]
public class MeshlessVisualiser : MonoBehaviour {
    [Header("Display Options")]
    public bool show = true;
    public bool showNodes = true;
    public bool showEdges = true;
    public bool showPrincipalStretch = true;
    public bool showDeformationGradient = false;
    public bool showVelocity = false;
    public bool showFixedNodes = true;

    [Header("Edge Display Mode")]
    public EdgeDisplayMode edgeMode = EdgeDisplayMode.HNSW;
    public enum EdgeDisplayMode {
        HNSW,           // Show HNSW level 0 connections (always available)
        CachedNeighbors // Show actual constraint neighbors (only after first physics step)
    }

    [Header("Yield Threshold")]
    public float yieldStretch = 1.05f;
    public float yieldAreaFrac = 1.05f;

    [Header("Scale Factors")]
    public float velocityScale = 0.5f;
    public float deformationScale = 0.4f;

    const float nodeSphereRadius = 0.09f;
    const float fixedNodeRadius = 0.12f;

    // Colors
    static readonly Color contractColor = new Color(0.2f, 0.45f, 1f, 1f);
    static readonly Color neutralColor = new Color(0.85f, 0.85f, 0.85f, 1f);
    static readonly Color expandColor = new Color(1f, 0.35f, 0.25f, 1f);
    static readonly Color fixedColor = new Color(0.3f, 0.3f, 0.3f, 1f);
    static readonly Color velocityColor = new Color(0f, 1f, 0.5f, 1f);
    static readonly Color fpColor = new Color(1f, 0.8f, 0f, 1f);

    private Camera mainCamera;
    public Meshless meshless;

    void Awake() {
        mainCamera = Camera.main;
        meshless = gameObject.GetComponent<Meshless>();
    }

    void Update() {
        if (Input.GetKeyDown(KeyCode.V)) showVelocity = !showVelocity;
        if (Input.GetKeyDown(KeyCode.D)) showDeformationGradient = !showDeformationGradient;
        if (Input.GetKeyDown(KeyCode.S)) showPrincipalStretch = !showPrincipalStretch;
        if (Input.GetKeyDown(KeyCode.E)) {
            edgeMode = edgeMode == EdgeDisplayMode.HNSW ?
                       EdgeDisplayMode.CachedNeighbors :
                       EdgeDisplayMode.HNSW;
        }
    }

    void OnDrawGizmos() {
        if (meshless == null || meshless.nodes == null || meshless.nodes.Count == 0 || !show)
            return;

        DrawEdgesAndStretch();
        DrawNodes();
        DrawDeformationGradients();
        DrawVelocities();
    }

    void DrawEdgesAndStretch() {
        if (!showEdges) return;

        if (edgeMode == EdgeDisplayMode.CachedNeighbors) {
            DrawCachedNeighborEdges();
        } else {
            DrawHNSWEdges();
        }
    }

    void DrawCachedNeighborEdges() {
        if (meshless.lastBatchDebug == null) {
            // No simulation data yet, fall back to HNSW
            DrawHNSWEdges();
            return;
        }

        for (int i = 0; i < meshless.nodes.Count; i++) {
            var node = meshless.nodes[i];
            var cache = meshless.lastBatchDebug.caches[i];
            if (cache?.neighbors == null) continue;

            foreach (var neighborIdx in cache.neighbors) {
                if (neighborIdx <= i) continue; // Draw each edge once
                if (neighborIdx < 0 || neighborIdx >= meshless.nodes.Count) continue;

                var neighbor = meshless.nodes[neighborIdx];

                // Calculate stretch
                float2 restEdge = math.mul(node.Fp, neighbor.originalPos - node.originalPos);
                float restLen = math.length(restEdge);
                float curLen = math.length(neighbor.pos - node.pos);
                float stretch = curLen / (restLen + 1e-8f);

                Color edgeColor = GetStretchColor(stretch);

                Gizmos.color = edgeColor;
                Gizmos.DrawLine(ToVector3(node.pos), ToVector3(neighbor.pos));
            }
        }
    }

    Color GetStretchColor(float stretch) {
        float compressYield = 1f / yieldStretch;

        if (stretch < compressYield) {
            return contractColor;
        } else if (stretch < 1f) {
            float t = (1f - stretch) / (1f - compressYield);
            return Color.Lerp(neutralColor, contractColor, t);
        } else if (stretch <= yieldStretch) {
            float t = (stretch - 1f) / (yieldStretch - 1f);
            return Color.Lerp(neutralColor, expandColor, t);
        } else {
            return expandColor;
        }
    }

    void DrawNodes() {
        if (!showNodes && !showPrincipalStretch) return;

        for (int i = 0; i < meshless.nodes.Count; i++) {
            var node = meshless.nodes[i];
            Color sphereColor = neutralColor;

            // Fixed nodes get special color and size
            if (node.isFixed) {
                if (!showFixedNodes) continue;
                Gizmos.color = fixedColor;
                Gizmos.DrawSphere(ToVector3(node.pos), fixedNodeRadius);
                continue;
            }

            // Color based on principal stretch
            if (showPrincipalStretch) {
                sphereColor = GetNodeStretchColor(i);
            }

            Gizmos.color = sphereColor;
            Gizmos.DrawSphere(ToVector3(node.pos), nodeSphereRadius);
        }
    }

    void DrawHNSWEdges() {
        for (int i = 0; i < meshless.nodes.Count; i++) {
            var node = meshless.nodes[i];
            if (node.HNSWNeighbors == null || node.HNSWNeighbors.Count == 0) continue;

            // HNSW level 0 neighbors
            foreach (var neighborIdx in node.HNSWNeighbors[0]) {
                if (neighborIdx <= i) continue; // Draw each edge once
                if (neighborIdx < 0 || neighborIdx >= meshless.nodes.Count) continue;

                var neighbor = meshless.nodes[neighborIdx];

                // Calculate stretch
                float2 restEdge = math.mul(node.Fp, neighbor.originalPos - node.originalPos);
                float restLen = math.length(restEdge);
                float curLen = math.length(neighbor.pos - node.pos);
                float stretch = curLen / (restLen + 1e-8f);

                Color edgeColor = GetStretchColor(stretch);

                Gizmos.color = edgeColor;
                Gizmos.DrawLine(ToVector3(node.pos), ToVector3(neighbor.pos));
            }
        }
    }

    Color GetNodeStretchColor(int nodeIdx) {
        var node = meshless.nodes[nodeIdx];

        // Try cached neighbors first (more accurate) - this is a List<int>
        var cache = meshless.lastBatchDebug?.caches[nodeIdx];
        System.Collections.Generic.IEnumerable<int> neighbors = cache?.neighbors;

        // Fall back to HNSW if no cache available - this is a HashSet<int>
        if (neighbors == null && node.HNSWNeighbors != null && node.HNSWNeighbors.Count > 0) {
            neighbors = node.HNSWNeighbors[0];
        }

        if (neighbors == null) return neutralColor;

        float maxStretch = 1f;
        int validNeighbors = 0;

        foreach (var neighborIdx in neighbors) {
            if (neighborIdx == nodeIdx || neighborIdx < 0 || neighborIdx >= meshless.nodes.Count) continue;

            var neighbor = meshless.nodes[neighborIdx];
            float2 restEdge = math.mul(node.Fp, neighbor.originalPos - node.originalPos);
            float restLen = math.length(restEdge);
            if (restLen < 1e-6f) continue;

            float curLen = math.length(neighbor.pos - node.pos);
            float stretch = curLen / restLen;
            maxStretch = math.max(maxStretch, stretch);
            validNeighbors++;
        }

        if (validNeighbors == 0) return neutralColor;

        if (maxStretch < 1f) {
            float t = math.saturate((1f - maxStretch) / 0.15f);
            return Color.Lerp(neutralColor, contractColor, t);
        } else {
            float t = math.saturate((maxStretch - 1f) / 0.15f);
            return Color.Lerp(neutralColor, expandColor, t);
        }
    }

    void DrawDeformationGradients() {
        if (!showDeformationGradient) return;

        for (int i = 0; i < meshless.nodes.Count; i++) {
            var node = meshless.nodes[i];
            if (node.isFixed) continue;

            Vector3 pos = ToVector3(node.pos);
            float2x2 Fp = node.Fp;

            // Draw Fp basis vectors
            Gizmos.color = fpColor;
            float2 e1 = Fp.c0 * deformationScale;
            float2 e2 = Fp.c1 * deformationScale;

            DrawArrow(pos, pos + ToVector3(e1), fpColor);
            DrawArrow(pos, pos + ToVector3(e2), fpColor);

            // Highlight non-identity deformation with wire sphere
            float det = Fp.c0.x * Fp.c1.y - Fp.c0.y * Fp.c1.x;
            if (math.abs(det - 1.0f) > 0.1f) {
                Gizmos.color = Color.red;
                Gizmos.DrawWireSphere(pos, nodeSphereRadius * 1.5f);
            }
        }
    }

    void DrawVelocities() {
        if (!showVelocity) return;

        for (int i = 0; i < meshless.nodes.Count; i++) {
            var node = meshless.nodes[i];
            if (node.isFixed) continue;

            float velMag = math.length(node.vel);
            if (velMag < 1e-6f) continue;

            Vector3 pos = ToVector3(node.pos);
            Vector3 velEnd = pos + ToVector3(node.vel * velocityScale);

            DrawArrow(pos, velEnd, velocityColor);
        }
    }

    void DrawArrow(Vector3 start, Vector3 end, Color color) {
        Gizmos.color = color;
        Gizmos.DrawLine(start, end);

        // Arrow head
        Vector3 dir = (end - start).normalized;
        Vector3 perpendicular = new Vector3(-dir.y, dir.x, 0) * 0.1f;
        float headSize = 0.15f;

        Gizmos.DrawLine(end, end - dir * headSize + perpendicular);
        Gizmos.DrawLine(end, end - dir * headSize - perpendicular);
    }

    static Vector3 ToVector3(float2 pt) => new Vector3(pt.x, pt.y, 0f);
}
