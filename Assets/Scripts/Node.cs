using System.Collections.Generic;
using Unity.Mathematics;

public class Node {
    public float2 pos;
    public float2 vel = float2.zero;
    public float2 originalPos;

    public float invMass = 1.0f;
    public bool isFixed = false;

    public float2x2 F = float2x2.identity;   // Total deformation gradient
    public float2x2 Fp = float2x2.identity;  // Plastic deformation gradient

    public int maxLayer;
    public List<HashSet<int>> HNSWNeighbors;
    public Meshless parent;

    public Node(float2 point, Meshless parent) {
        pos = point;
        originalPos = point;
        this.parent = parent;
        maxLayer = GetRandomLayer();
    }

    private static int GetRandomLayer(float ml = 0.6f) {
        return (int)(math.floor(math.log(1.0 / UnityEngine.Random.value) * ml) + 0.01);
    }
}