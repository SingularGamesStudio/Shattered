using System.Collections.Generic;
using Unity.Mathematics;

public class Node {
    public float2 pos;
    public float invMass = 1.0f;
    public bool isFixed = false;
    public int maxLayer;

    public float2 vel = float2.zero;
    public float2 predPos;
    public float contraction = 1.0f;

    public List<HashSet<int>> HNSWNeighbors;
    public Meshless parent;

    public Node(float2 point, Meshless parent) {
        pos = point;
        maxLayer = GetRandomLayer();
        this.parent = parent;
    }

    private static int GetRandomLayer(float ml = 0.6f) {
        return (int)(math.floor(math.log(1.0 / UnityEngine.Random.value) * ml) + 0.01);
    }
}
