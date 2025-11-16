using Unity.Mathematics;
using UnityEngine;

/// <summary>
/// XPBI Visualizer: shows node stretch, rest/vectors, and local rotation.
/// Step 5: Visual debugging for rotation-invariant plasticity.
/// </summary>
[ExecuteInEditMode]
public class MeshlessVisualiser : MonoBehaviour {
    public bool show = true;
    public bool showNodes = true;
    public bool showEdges = true;
    public bool showPrincipalStretch = true;
    public bool showLocalRotation = true;
    public bool showRestVectors = true;
    public float yieldStretch = 1.05f; // Set to match your yield checks

    const float nodeSphereRadius = 0.09f;
    const float restArrowScale = 0.5f;
    const float rotArrowScale = 0.3f;
    static readonly Color contractColor = new Color(0.2f, 0.45f, 1f, 1f);
    static readonly Color neutralColor = new Color(0.85f, 0.85f, 0.85f, 1f);
    static readonly Color expandColor = new Color(1f, 0.35f, 0.25f, 1f);
    static readonly Color plasticRefColor = Color.magenta;

    private Camera mainCamera;
    public Meshless meshless;

    void Awake() {
        mainCamera = Camera.main;
        meshless = gameObject.GetComponent<Meshless>();
    }

    void Update() {
        // Toggle visualization features
        if (Input.GetKeyDown(KeyCode.R)) showRestVectors = !showRestVectors;
        if (Input.GetKeyDown(KeyCode.L)) showLocalRotation = !showLocalRotation;
        if (Input.GetKeyDown(KeyCode.S)) showPrincipalStretch = !showPrincipalStretch;
    }

    void OnDrawGizmos() {
        if (meshless == null || meshless.nodes == null || meshless.nodes.Count == 0 || !show)
            return;

        DrawEdgesAndStretch();
        DrawNodes();
    }

    void DrawEdgesAndStretch() {
        if (!showEdges) return;
        for (int i = 0; i < meshless.nodes.Count; i++) {
            var node = meshless.nodes[i];
            if (node.HNSWNeighbors == null) continue;
            foreach (var neighborIdx in node.HNSWNeighbors[0]) {
                if (neighborIdx <= i) continue;
                var neighbor = meshless.nodes[neighborIdx];
                float2 xi0 = node.originalPos;
                float2 xj0 = neighbor.originalPos;
                float2 restEdge = math.mul(node.Fp, xj0 - xi0);
                float2 curEdge = neighbor.pos - node.pos;

                float restLen = math.length(restEdge);
                float curLen = math.length(curEdge);
                float stretch = curLen / (restLen + 1e-8f);

                // Color: blue<1, white≈1, red>1, magenta if yielding
                Color edgeColor = Color.Lerp(contractColor, expandColor, math.saturate((stretch - 1f) / 0.15f));
                edgeColor = Color.Lerp(neutralColor, edgeColor, math.saturate(math.abs(stretch - 1f) / 0.15f));
                if (stretch > yieldStretch || stretch < 1f / yieldStretch)
                    edgeColor = plasticRefColor;
                Gizmos.color = edgeColor;
                Gizmos.DrawLine(ToVector3(node.pos), ToVector3(neighbor.pos));

                // --- Rest direction as arrow ---
                if (showRestVectors) {
                    Gizmos.color = Color.yellow;
                    Vector3 restDirEnd = ToVector3(node.pos) + ToVector3(math.normalize(restEdge) * restArrowScale);
                    Gizmos.DrawLine(ToVector3(node.pos), restDirEnd);
                }
            }
        }
    }

    void DrawNodes() {
        if (!showNodes && !showPrincipalStretch) return;

        for (int i = 0; i < meshless.nodes.Count; i++) {
            var node = meshless.nodes[i];
            Color sphereColor = Color.white;

            if (showPrincipalStretch && node.HNSWNeighbors != null && node.HNSWNeighbors.Count > 0) {
                float maxStretch = 1f;
                int neighborCount = 0;
                foreach (var neighborIdx in node.HNSWNeighbors[0]) {
                    if (neighborIdx == i) continue;
                    var neighbor = meshless.nodes[neighborIdx];
                    float2 xi0 = node.originalPos;
                    float2 xj0 = neighbor.originalPos;
                    float2 restEdge = math.mul(node.Fp, xj0 - xi0);
                    float2 curEdge = neighbor.pos - node.pos;
                    float restLen = math.length(restEdge);
                    float curLen = math.length(curEdge);
                    if (restLen < 1e-6f) continue; // skip zero-length
                    float stretch = curLen / restLen;
                    maxStretch = math.max(maxStretch, stretch);
                    neighborCount++;
                }
                // Only color if there’s at least one neighbor!
                if (neighborCount > 0) {
                    sphereColor = Color.Lerp(contractColor, expandColor, math.saturate((maxStretch - 1f) / 0.15f));
                }
            }

            Gizmos.color = sphereColor;
            Gizmos.DrawSphere(ToVector3(node.pos), nodeSphereRadius);
        }
    }


    static Vector3 ToVector3(float2 pt) => new Vector3(pt.x, pt.y, 0f);
}