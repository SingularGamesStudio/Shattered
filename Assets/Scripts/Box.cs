using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Mathematics;

public class Box : Meshless
{
    public float2 center;
    public float2 size;

    [Header("Generator")]
    public int pointCount;

    public void Generate(int count, short material)
    {
        pos.Clear();
        vel.Clear();
        mat.Clear();
        pos.Add(center + size / 2);
		pos.Add(center - size / 2);
        var size1 = new float2(size.x, -size.y);
        pos.Add(center + size1 / 2);
		pos.Add(center - size1 / 2);
        Unity.Mathematics.Random rnd = new Unity.Mathematics.Random((uint)(Time.time*1000000));
        for (int i = 0; i<count-4; i++) {
            pos.Add(rnd.NextFloat2(center - size / 2, center + size / 2));
        }
        for(int i = 0; i<count;i++) {
            vel.Add(0);
            mat.Add(material);
        }
	}
}
