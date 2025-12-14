using System.Collections.Generic;
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

    [Header("Rest State (estimated)")]
    public bool showRestArrow = true;
    public float restArrowScale = 1.0f;

    [Header("Edge Display Mode")]
    public EdgeDisplayMode edgeMode = EdgeDisplayMode.HNSW;
    public enum EdgeDisplayMode {
        HNSW,
        CachedNeighbors
    }

    [Header("Yield Threshold")]
    public float yieldStretch = 1.05f;

    [Header("Scale Factors")]
    public float velocityScale = 0.5f;
    public float deformationScale = 0.4f;

    const float nodeSphereRadius = 0.09f;
    const float fixedNodeRadius = 0.12f;
    const float eps = 1e-6f;

    static readonly Color contractColor = new Color(0.2f, 0.45f, 1f, 1f);
    static readonly Color neutralColor = new Color(0.85f, 0.85f, 0.85f, 1f);
    static readonly Color expandColor = new Color(1f, 0.35f, 0.25f, 1f);
    static readonly Color fixedColor = new Color(0.3f, 0.3f, 0.3f, 1f);
    static readonly Color velocityColor = new Color(0f, 1f, 0.5f, 1f);
    static readonly Color fpColor = new Color(1f, 0.8f, 0f, 1f);
    static readonly Color restColor = new Color(0.75f, 0.2f, 1f, 1f);

    Meshless meshless;

    void Awake() {
        meshless = GetComponent<Meshless>();
    }

    void OnDrawGizmos() {
        if (!show) return;

        meshless ??= GetComponent<Meshless>();
        if (meshless == null || meshless.nodes == null || meshless.nodes.Count == 0) return;

        if (showEdges) DrawEdges();
        if (showNodes || showPrincipalStretch) DrawNodes();
        if (showDeformationGradient) DrawDeformationGradients();
        if (showVelocity) DrawVelocities();
        if (showRestArrow) DrawRestArrows();
    }

    void DrawEdges() {
        if (edgeMode == EdgeDisplayMode.CachedNeighbors && meshless.lastBatchDebug != null) DrawCachedNeighborEdges();
        else DrawHNSWEdges();
    }

    void DrawCachedNeighborEdges() {
        var batch = meshless.lastBatchDebug;

        for (int i = 0; i < meshless.nodes.Count; i++) {
            var cache = batch.caches[i];
            var N = cache?.neighbors;
            if (N == null) continue;

            for (int k = 0; k < N.Count; k++) {
                int j = N[k];
                if (j <= i || (uint)j >= (uint)meshless.nodes.Count) continue;

                float stretch = EdgeStretch(i, j);
                Gizmos.color = GetStretchColor(stretch);
                Gizmos.DrawLine(ToVector3(meshless.nodes[i].pos), ToVector3(meshless.nodes[j].pos));
            }
        }
    }

    void DrawHNSWEdges() {
        for (int i = 0; i < meshless.nodes.Count; i++) {
            var node = meshless.nodes[i];
            if (node.HNSWNeighbors == null || node.HNSWNeighbors.Count == 0) continue;

            foreach (int j in node.HNSWNeighbors[0]) {
                if (j <= i || (uint)j >= (uint)meshless.nodes.Count) continue;

                float stretch = EdgeStretch(i, j);
                Gizmos.color = GetStretchColor(stretch);
                Gizmos.DrawLine(ToVector3(node.pos), ToVector3(meshless.nodes[j].pos));
            }
        }
    }

    float EdgeStretch(int i, int j) {
        var ni = meshless.nodes[i];
        var nj = meshless.nodes[j];

        float2 restEdge = math.mul(ni.Fp, nj.originalPos - ni.originalPos);
        float restLen = math.length(restEdge);
        float curLen = math.length(nj.pos - ni.pos);
        return curLen / math.max(restLen, eps);
    }

    Color GetStretchColor(float stretch) {
        float compressYield = 1f / math.max(yieldStretch, eps);

        if (stretch < compressYield) return contractColor;
        if (stretch < 1f) return Color.Lerp(neutralColor, contractColor, (1f - stretch) / math.max(1f - compressYield, eps));
        if (stretch <= yieldStretch) return Color.Lerp(neutralColor, expandColor, (stretch - 1f) / math.max(yieldStretch - 1f, eps));
        return expandColor;
    }

    void DrawNodes() {
        for (int i = 0; i < meshless.nodes.Count; i++) {
            var node = meshless.nodes[i];

            if (node.isFixed) {
                if (!showFixedNodes) continue;
                Gizmos.color = fixedColor;
                Gizmos.DrawSphere(ToVector3(node.pos), fixedNodeRadius);
                continue;
            }

            Gizmos.color = showPrincipalStretch ? GetNodeStretchColor(i) : neutralColor;
            if (showNodes || showPrincipalStretch) Gizmos.DrawSphere(ToVector3(node.pos), nodeSphereRadius);
        }
    }

    Color GetNodeStretchColor(int i) {
        IEnumerable<int> N = meshless.lastBatchDebug?.caches[i]?.neighbors;

        // Fallback to HNSW level-0 graph if the simulation hasn't produced cached neighbors yet.
        if (N == null) {
            var node = meshless.nodes[i];
            if (node.HNSWNeighbors != null && node.HNSWNeighbors.Count > 0) N = node.HNSWNeighbors[0];
        }
        if (N == null) return neutralColor;

        float maxStretch = 1f;
        int valid = 0;

        foreach (int j in N) {
            if (j == i || (uint)j >= (uint)meshless.nodes.Count) continue;

            float2 restEdge = math.mul(meshless.nodes[i].Fp, meshless.nodes[j].originalPos - meshless.nodes[i].originalPos);
            float restLen = math.length(restEdge);
            if (restLen < eps) continue;

            float stretch = math.length(meshless.nodes[j].pos - meshless.nodes[i].pos) / restLen;
            maxStretch = math.max(maxStretch, stretch);
            valid++;
        }

        if (valid == 0) return neutralColor;

        float t = math.saturate(math.abs(maxStretch - 1f) / 0.15f);
        return maxStretch < 1f ? Color.Lerp(neutralColor, contractColor, t)
                               : Color.Lerp(neutralColor, expandColor, t);
    }

    void DrawDeformationGradients() {
        for (int i = 0; i < meshless.nodes.Count; i++) {
            var node = meshless.nodes[i];
            if (node.isFixed) continue;

            Vector3 p = ToVector3(node.pos);
            float2 e1 = node.Fp.c0 * deformationScale;
            float2 e2 = node.Fp.c1 * deformationScale;

            DrawArrow(p, p + ToVector3(e1), fpColor);
            DrawArrow(p, p + ToVector3(e2), fpColor);

            float det = node.Fp.c0.x * node.Fp.c1.y - node.Fp.c0.y * node.Fp.c1.x;
            if (math.abs(det - 1f) > 0.1f) {
                Gizmos.color = Color.red;
                Gizmos.DrawWireSphere(p, nodeSphereRadius * 1.5f);
            }
        }
    }

    void DrawVelocities() {
        for (int i = 0; i < meshless.nodes.Count; i++) {
            var node = meshless.nodes[i];
            if (node.isFixed) continue;

            if (math.lengthsq(node.vel) < 1e-10f) continue;
            Vector3 p = ToVector3(node.pos);
            DrawArrow(p, p + ToVector3(node.vel * velocityScale), velocityColor);
        }
    }

    void DrawRestArrows() {
        for (int i = 0; i < meshless.nodes.Count; i++) {
            var node = meshless.nodes[i];
            if (node.isFixed) continue;

            IEnumerable<int> N = meshless.lastBatchDebug?.caches[i]?.neighbors;
            if (N == null) {
                if (node.HNSWNeighbors == null || node.HNSWNeighbors.Count == 0) continue;
                N = node.HNSWNeighbors[0];
            }

            float2x2 Ftot = math.mul(node.F, node.Fp);

            float2 sum = float2.zero;
            int count = 0;

            foreach (int j in N) {
                if (j == i || (uint)j >= (uint)meshless.nodes.Count) continue;

                float2 rij = math.mul(Ftot, meshless.nodes[j].originalPos - node.originalPos);
                sum += (meshless.nodes[j].pos - rij);
                count++;
            }

            if (count == 0) continue;

            float2 xRest = sum / count;
            Vector3 p = ToVector3(node.pos);
            Vector3 q = ToVector3(node.pos + (xRest - node.pos) * restArrowScale);

            DrawArrow(p, q, restColor);
        }
    }


    static void DrawArrow(Vector3 start, Vector3 end, Color color) {
        Gizmos.color = color;
        Gizmos.DrawLine(start, end);

        Vector3 d = end - start;
        float len = d.magnitude;
        if (len < 1e-6f) return;

        Vector3 dir = d / len;
        Vector3 perp = new Vector3(-dir.y, dir.x, 0f);

        float head = math.min(0.15f, 0.35f * len);
        Gizmos.DrawLine(end, end - dir * head + perp * (0.1f * head));
        Gizmos.DrawLine(end, end - dir * head - perp * (0.1f * head));
    }

    static Vector3 ToVector3(float2 p) => new Vector3(p.x, p.y, 0f);
}
