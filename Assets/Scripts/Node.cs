using System.Collections.Generic;
using Unity.Mathematics;

public class Node {
    public float2 pos;
    public float2 originalPos;

    public float invMass = 1.0f;
    public bool isFixed = false;

    public float restVolume = 0.0f;

    public int maxLayer;
    public Meshless parent;

    public int materialId = 0;

    public Node(float2 point, Meshless parent) {
        pos = point;
        originalPos = point;
        this.parent = parent;
        maxLayer = 0;
        materialId = parent != null ? parent.GetBaseMaterialId() : 0;
    }
}
