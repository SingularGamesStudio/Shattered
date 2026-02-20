using System.Collections.Generic;
using Unity.Mathematics;

public class Node {
    public float2 pos;
    public float2 vel = float2.zero;
    public float2 originalPos;

    public float invMass = 1.0f;
    public bool isFixed = false;

    public float2x2 F = float2x2.identity;
    public float2x2 Fp = float2x2.identity;

    public float restVolume = 0.0f;

    public int maxLayer;
    public Meshless parent;

    public int parentIndex = -1;

    public int materialId = 0;

    public Node(float2 point, Meshless parent) {
        pos = point;
        originalPos = point;
        this.parent = parent;
        maxLayer = 0;
        materialId = parent != null ? parent.GetBaseMaterialId() : 0;
    }

    private static int GetRandomLayer(float ml = 0.6f) {
        return (int)(math.floor(math.log(1.0f / UnityEngine.Random.value) * ml) + 0.01f);
    }

    public bool AtLevel(int level) {
        return level <= maxLayer;
    }
}
