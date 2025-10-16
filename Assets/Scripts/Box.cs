using Unity.Mathematics;
using UnityEngine;

public class Box : Meshless {
    public float2 center;
    public float2 size;

    [Header("Generator")]
    public int pointCount;

    public void Generate(int count, short material) {
        if (nodes.Count == 0) {
            Add(center + size / 2);
            Add(center - size / 2);
            var size1 = new float2(size.x, -size.y);
            Add(center + size1 / 2);
            Add(center - size1 / 2);
        }
        Unity.Mathematics.Random rnd = new Unity.Mathematics.Random((uint)(Time.time * 1000000));
        for (int i = 0; i < count; i++) {
            Add(rnd.NextFloat2(center - size / 2, center + size / 2));
        }
        FixNode(0);
        Build();
    }
}
