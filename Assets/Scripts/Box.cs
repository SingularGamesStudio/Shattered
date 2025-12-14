using Unity.Mathematics;
using UnityEngine;

public class Box : Meshless {
    public float2 center;
    public float2 size;

    [Header("Generator")]
    public int pointCount;
    public bool generateOnStart = false;

    void Start() {
        if (generateOnStart) Generate(pointCount, 0);
    }


    public void Generate(int count, short material) {
        if (nodes.Count == 0) {
            Add(center + size / 2);
            Add(center - size / 2);
            var size1 = new float2(size.x, -size.y);
            Add(center + size1 / 2);
            Add(center - size1 / 2);
        }
        uint seed = (uint)System.Guid.NewGuid().GetHashCode();
        if (seed == 0) seed = 1;
        var rnd = new Unity.Mathematics.Random(seed);
        for (int i = 0; i < count; i++) {
            Add(rnd.NextFloat2(center - size / 2, center + size / 2));
        }
        FixNode(0);
        //FixNode(3);
        Build();
    }
}
