using Unity.Mathematics;
using UnityEngine;

/// <summary>
/// XPBI visualizer: can show node positions, plastic reference positions, plastic deformation arrows,
/// and optionally color edges by magnitude of plastic shift.
/// </summary>
[ExecuteInEditMode]
public class MeshlessVisualiser : MonoBehaviour {
    // Visibility toggles
    public bool show = true;                  // Master toggle
    public bool showNodes = true;             // Draw node spheres
    public bool showEdges = true;             // Draw edges
    public bool colorEdgesByPlasticity = true;// If true, edges colored by plastic shift magnitude
    public bool showPlasticRefs = true;       // Show plastic reference positions
    public bool showPlasticArrows = true;     // Show plastic deformation arrows
    public bool showVelocities = false;       // Draw velocity vectors
    public bool showRestLengthSegment = true;

    const float levelAlphaFalloff = 0.15f;
    const float nodeSphereRadius = 0.1f;
    const float plasticRefRadius = 0.09f;
    const float plasticArrowScale = 0.8f;
    const float velocityScale = 0.25f;
    static readonly Color contractColor = new Color(0.2f, 0.45f, 1f, 1f);
    static readonly Color neutralColor = new Color(0.85f, 0.85f, 0.85f, 1f);
    static readonly Color expandColor = new Color(1f, 0.35f, 0.25f, 1f);
    static readonly Color velocityColor = Color.yellow;
    static readonly Color plasticRefColor = Color.magenta;

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

        if (showPlasticRefs) {
            foreach (var node in meshless.nodes) {
                Vector3 refPos = ToVector3(node.plasticReferencePos);
                Gizmos.color = plasticRefColor;
                Gizmos.DrawSphere(refPos, plasticRefRadius);
            }
        }

        if (showPlasticArrows) {
            DrawPlasticDeformationArrows();
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

            float2 posA2 = node.pos;
            Vector3 posA = ToVector3(posA2);

            foreach (var neighbor in node.HNSWNeighbors[level]) {
                if (neighbor <= i) continue; // avoid duplicates
                var other = meshless.nodes[neighbor];
                float2 posB2 = other.pos;
                Vector3 posB = ToVector3(posB2);

                // Compute signed expansion (positive), contraction (negative)
                float2 edgeDir = math.normalize(posB2 - posA2);

                float2 refA = node.plasticReferencePos;
                float2 refB = other.plasticReferencePos;
                float restLen = math.distance(refA, refB); // computed from plastic ref
                float currLen = math.distance(posA2, posB2);
                float contractionVal = (currLen - restLen) / restLen; // negative = contracted, positive = expanded

                // Compute color gradient between contractColor (blue) and expandColor (red)
                float blend = math.clamp(0.5f + 0.5f * contractionVal / 0.3f, 0.0f, 1.0f);
                // When contractionVal ==  0 -> blend = 0.5 (neutral)
                // When contractionVal == -0.3 -> blend ≈ 0.0 (full blue)
                // When contractionVal == +0.3 -> blend ≈ 1.0 (full red)
                Color edgeColor = Color.Lerp(contractColor, expandColor, blend);
                edgeColor = Color.Lerp(neutralColor, edgeColor, 0.85f);
                edgeColor.a = alpha;

                Gizmos.color = edgeColor;
                Gizmos.DrawLine(posA, posB);

                // Optionally draw rest length as a highlighted segment along the edge from posA
                if (showRestLengthSegment && restLen > 1e-6f) {
                    float t = math.saturate(restLen / math.distance(posA2, posB2));
                    Vector3 restEdgeEnd = Vector3.Lerp(posA, posB, math.clamp(restLen / (currLen + 1e-7f), 0f, 1f));
                    Gizmos.color = Color.yellow;
                    Gizmos.DrawLine(posA, restEdgeEnd);
                }
            }
        }
    }

    void DrawPlasticDeformationArrows() {
        Gizmos.color = plasticRefColor;
        foreach (var node in meshless.nodes) {
            Vector3 p = ToVector3(node.pos);
            Vector3 pRef = ToVector3(node.plasticReferencePos);
            Vector3 arrow = (pRef - p) * plasticArrowScale;

            if (arrow.sqrMagnitude > 1e-8f) {
                Gizmos.DrawLine(p, p + arrow);
                Vector3 dir = arrow.normalized;
                Vector3 left = Quaternion.AngleAxis(25f, Vector3.forward) * (-dir);
                Vector3 right = Quaternion.AngleAxis(-25f, Vector3.forward) * (-dir);
                float ah = 0.08f;
                Vector3 tip = p + arrow;
                Gizmos.DrawLine(tip, tip + left * ah);
                Gizmos.DrawLine(tip, tip + right * ah);
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
                float ah = 0.12f;
                Vector3 tip = p + v;
                Gizmos.DrawLine(tip, tip + left * ah);
                Gizmos.DrawLine(tip, tip + right * ah);
            }
        }
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
            float minDist = 0.2f;
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