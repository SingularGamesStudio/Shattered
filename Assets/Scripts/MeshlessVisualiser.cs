using Unity.Mathematics;
using UnityEngine;

[ExecuteInEditMode]
public class MeshlessVisualiser : MonoBehaviour {
    public bool showNodes = true;
    public bool show = true;
    private int draggedNodeIndex = -1;
    private Camera mainCamera;
    public Meshless meshless;

    void Awake() {
        mainCamera = Camera.main;
        meshless = gameObject.GetComponent<Meshless>();
    }

    void OnDrawGizmos() {
        if (meshless.hnsw == null || meshless.nodes == null || meshless.nodes.Count == 0 || !show)
            return;

        int maxLevel = meshless.maxLayer;

        // Draw edges layer by layer with fading alpha for higher layers
        for (int level = 0; level <= maxLevel; level++) {
            float alpha = 1f - 0.15f * level;
            Gizmos.color = new Color(0f, 0.5f, 1f, alpha);
            DrawEdges(level);
        }
        if (!showNodes) {
            return;
        }
        // Draw nodes on top
        foreach (var node in meshless.nodes) {
            Vector3 pos = ToVector3(node.pos);
            float str = ((float)node.maxLayer) / meshless.maxLayer;
            if (maxLevel == 0) {
                str = 0f;
            }
            Gizmos.color = Color.blue * str + Color.red * (1f - str);
            Gizmos.DrawSphere(pos, 0.1f);
        }
    }

    void DrawEdges(int level) {
        for (int i = 0; i < meshless.nodes.Count; i++) {
            var node = meshless.nodes[i];
            if (level > node.maxLayer)
                continue; // skip layers above node max layer

            Vector3 posA = ToVector3(node.pos);

            foreach (var neighbor in node.HNSWNeighbors[level]) {
                // To avoid duplicate lines, draw only if higher index
                if (neighbor > i) {
                    Vector3 posB = ToVector3(meshless.nodes[neighbor].pos);
                    Gizmos.DrawLine(posA, posB);
                }
            }
        }
    }

    Vector3 ToVector3(float2 point) {
        // Map float2 XY plane to Unity Vector3 space
        return new Vector3(point.x, point.y, 0f);
    }

    void Update() {
        HandleDragging();
    }

    void HandleDragging() {
        if (meshless.hnsw == null) return;

        if (Input.GetMouseButtonDown(0)) {
            // On mouse down try to select nearest node within pick radius

            Vector3 mousePos = Input.mousePosition;
            Ray ray = mainCamera.ScreenPointToRay(mousePos);

            float minDist = 0.2f; // pick radius in world units
            draggedNodeIndex = -1;

            for (int nodeIdx = 0; nodeIdx < meshless.nodes.Count; nodeIdx++) {
                Vector3 nodePos = ToVector3(meshless.nodes[nodeIdx].pos);
                Vector3 screenPoint = mainCamera.WorldToScreenPoint(nodePos);

                float dist = Vector2.Distance(new Vector2(screenPoint.x, screenPoint.y), new Vector2(mousePos.x, mousePos.y));
                if (dist < minDist * 100f) // adjust scale for screen pixels vs world units
                {
                    draggedNodeIndex = nodeIdx;
                    break;
                }
            }
        } else if (Input.GetMouseButton(0) && draggedNodeIndex != -1) {
            // Drag selected node with mouse raycast onto XY plane z=0
            Ray ray = mainCamera.ScreenPointToRay(Input.mousePosition);
            float t;
            if (PlaneRayIntersection(ray, Vector3.zero, Vector3.forward, out t)) {
                Vector3 hitPos = ray.origin + ray.direction * t;
                var newPoint = new float2(hitPos.x, hitPos.y);
                meshless.hnsw.Shift(draggedNodeIndex, newPoint);
            }
        } else if (Input.GetMouseButtonUp(0)) {
            draggedNodeIndex = -1;
        }
    }

    bool PlaneRayIntersection(Ray ray, Vector3 planePoint, Vector3 planeNormal, out float t) {
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
