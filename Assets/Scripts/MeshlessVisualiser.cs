using Unity.Mathematics;
using UnityEngine;

[ExecuteInEditMode]
public class MeshlessVisualiser : MonoBehaviour {
    // Visibility toggles
    public bool show = true;                // Master toggle
    public bool showNodes = true;           // Draw node spheres
    public bool showEdges = true;           // Draw edges
    public bool colorEdgesByContraction = true; // Color edges using node-wise contraction
    public bool showVelocities = false;     // Draw velocity vectors

    // Internal constants (fixed as requested)
    const float levelAlphaFalloff = 0.15f;  // Per-level alpha fade
    const float nodeSphereRadius = 0.1f;    // Node gizmo size
    const float maxAbsContraction = 0.3f;   // Saturation point for color mapping
    const float velocityScale = 0.25f;      // World units per velocity unit
    static readonly Color contractColor = new Color(0.2f, 0.45f, 1f, 1f); // contracted (negative)
    static readonly Color neutralColor = new Color(0.85f, 0.85f, 0.85f, 1f); // near zero
    static readonly Color expandColor = new Color(1f, 0.35f, 0.25f, 1f); // expanded (positive)
    static readonly Color velocityColor = Color.yellow;

    private int draggedNodeIndex = -1;
    private Camera mainCamera;
    public Meshless meshless;

    void Awake() {
        mainCamera = Camera.main;
        meshless = gameObject.GetComponent<Meshless>();
    }

    void OnDrawGizmos() {
        if (meshless == null || meshless.hnsw == null || meshless.nodes == null || meshless.nodes.Count == 0 || !show)
            return;

        int maxLevel = meshless.maxLayer;

        if (showEdges) {
            for (int level = 0; level <= maxLevel; level++) {
                DrawEdges(level, maxLevel);
            }
        }

        if (showNodes) {
            foreach (var node in meshless.nodes) {
                Vector3 pos = ToVector3(node.pos);
                float str = (meshless.maxLayer == 0) ? 0f : ((float)node.maxLayer) / meshless.maxLayer;
                Gizmos.color = Color.blue * str + Color.red * (1f - str);
                Gizmos.DrawSphere(pos, nodeSphereRadius);
            }
        }

        if (showVelocities) {
            DrawVelocities();
        }
    }

    void DrawEdges(int level, int maxLevel) {
        float alpha = 1f - levelAlphaFalloff * level;

        for (int i = 0; i < meshless.nodes.Count; i++) {
            var node = meshless.nodes[i];
            if (level > node.maxLayer) continue;
            if (node.HNSWNeighbors == null || level >= node.HNSWNeighbors.Count || node.HNSWNeighbors[level] == null)
                continue;

            Vector3 posA = ToVector3(node.pos);

            foreach (var neighbor in node.HNSWNeighbors[level]) {
                if (neighbor <= i) continue; // avoid duplicates
                var other = meshless.nodes[neighbor];
                Vector3 posB = ToVector3(other.pos);

                Color edgeColor;
                if (colorEdgesByContraction) {
                    float cEdge = math.pow(node.contraction * other.contraction, 2f); // signed
                    edgeColor = ColorFromContraction(cEdge, alpha);
                } else {
                    edgeColor = new Color(0f, 0.5f, 1f, alpha);
                }

                Gizmos.color = edgeColor;
                Gizmos.DrawLine(posA, posB);
            }
        }
    }

    void DrawVelocities() {
        Gizmos.color = velocityColor;
        for (int i = 0; i < meshless.nodes.Count; i++) {
            var n = meshless.nodes[i];
            Vector3 p = ToVector3(n.pos);
            Vector3 v = new Vector3(n.vel.x, n.vel.y, 0f) * velocityScale;

            Gizmos.DrawLine(p, p + v);

            if (v.sqrMagnitude > 1e-8f) {
                Vector3 dir = v.normalized;
                Vector3 left = Quaternion.AngleAxis(25f, Vector3.forward) * (-dir);
                Vector3 right = Quaternion.AngleAxis(-25f, Vector3.forward) * (-dir);
                float ah = 0.12f; // fixed arrowhead size
                Vector3 tip = p + v;
                Gizmos.DrawLine(tip, tip + left * ah);
                Gizmos.DrawLine(tip, tip + right * ah);
            }
        }
    }

    static Color ColorFromContraction(float c, float alpha) {
        float t = Mathf.Max(c, 1f / c) - 1f;
        Color to = c >= 1f ? expandColor : contractColor;
        Color cOut = Color.Lerp(neutralColor, to, t);
        cOut.a = alpha;
        return cOut;
    }

    static Vector3 ToVector3(float2 point) {
        return new Vector3(point.x, point.y, 0f);
    }

    void Update() {
        HandleDragging();
    }

    void HandleDragging() {
        if (meshless == null || meshless.hnsw == null) return;

        if (Input.GetMouseButtonDown(0)) {
            Vector3 mousePos = Input.mousePosition;
            float minDist = 0.2f; // world units
            draggedNodeIndex = -1;

            for (int nodeIdx = 0; nodeIdx < meshless.nodes.Count; nodeIdx++) {
                Vector3 nodePos = ToVector3(meshless.nodes[nodeIdx].pos);
                Vector3 screenPoint = mainCamera.WorldToScreenPoint(nodePos);

                float dist = Vector2.Distance(new Vector2(screenPoint.x, screenPoint.y), new Vector2(mousePos.x, mousePos.y));
                if (dist < minDist * 100f) {
                    draggedNodeIndex = nodeIdx;
                    break;
                }
            }
        } else if (Input.GetMouseButton(0) && draggedNodeIndex != -1) {
            Ray ray = mainCamera.ScreenPointToRay(Input.mousePosition);
            if (PlaneRayIntersection(ray, Vector3.zero, Vector3.forward, out float t)) {
                Vector3 hitPos = ray.origin + ray.direction * t;
                var newPoint = new float2(hitPos.x, hitPos.y);
                meshless.hnsw.Shift(draggedNodeIndex, newPoint);
            }
        } else if (Input.GetMouseButtonUp(0)) {
            draggedNodeIndex = -1;
        }
    }

    static bool PlaneRayIntersection(Ray ray, Vector3 planePoint, Vector3 planeNormal, out float t) {
        float denom = Vector3.Dot(planeNormal, ray.direction);
        if (Mathf.Abs(denom) > 1e-6f) {
            Vector3 diff = planePoint - ray.origin;
            t = Vector3.Dot(diff, planeNormal) / denom;
            return t >= 0f;
        }
        t = 0;
        return false;
    }
}