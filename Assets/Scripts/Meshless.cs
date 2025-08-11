using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.UIElements;

public class Meshless : MonoBehaviour
{
    public List<float2> pos;
	public List<float2> vel;
	public List<short> mat;
	public static void DrawCircle(float2 a, float r) => Gizmos.DrawWireSphere(new float3(a.x, a.y, 0), r);
	protected void OnDrawGizmos()
	{
		if(pos==null || pos.Count==0) {
			return;
		}
		float2 T(float2 x) => Application.isPlaying ? x : math.mul((float4x4)transform.localToWorldMatrix, math.float4(x, 1, 1)).xy;
		foreach (float2 p in pos) {
			DrawCircle(T(p), 0.1f);
		}
	}
}
