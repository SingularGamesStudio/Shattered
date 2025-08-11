using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.Mathematics;
using Unity.Collections;
using Unity.Burst;
using static UnityEditor.Experimental.GraphView.GraphView;

public static class HNSW
{
	public class Params
	{
		public int layers = 3;
		public int neighbors = 3;
		public int neighbors0 = 5;
		public float probaBase = 8;
		
		public Params(int layers = 3, int neighbors = 3, int neighbors0 = 5, float probaBase = 8)
		{
			this.layers = layers;
			this.neighbors = neighbors;
			this.neighbors0 = neighbors0;
			this.probaBase = probaBase;
		}
	}
	public class Instance
	{
		public Params param;
		// ordered by layer, starting from top
		public NativeList<int> neighbors;//TODO:
		public NativeList<float2> points;
		public NativeList<int> neighborsFrom;
		public NativeList<int> pointLayers;
		public NativeList<int> topLayer;
		public Instance(NativeArray<float2> points, Params param = null)
		{
			if (param == null) {
				this.param = new Params();
			} else {
				this.param = param;
			}
			foreach(float2 p in points) {
				Insert(this, p);
			}
		}
	}
	public static void Insert(Instance inst, float2 point)
	{
		int layer = (int)(math.floor(math.log(1.0 / (new Unity.Mathematics.Random().NextDouble(1))) / math.log(inst.param.probaBase))+0.01);
		inst.pointLayers.Add(layer);
		inst.points.Add(point);
		
	}
	public static int Find(Instance inst, float2 point)
	{
		int curPoint = inst.topLayer[new Unity.Mathematics.Random().NextInt(inst.topLayer.Length)];
		for (int layer = inst.param.layers; layer>=0; layer--) {
			curPoint = greedyNeighbor(inst, point, curPoint, layer);
		}
		return curPoint;
	}
	static int greedyNeighbor(Instance inst, float2 point, int origin, int layer) {
		int best = origin;
		int start = inst.neighborsFrom[origin] + (inst.pointLayers[origin] - layer) * inst.param.neighbors;
		int end = start + ((layer == 0) ? inst.param.neighbors0 : inst.param.neighbors);
		for (int i = start; i < end; i++) {
			int neighbor = inst.neighbors[i];
			if(neighbor==-1) {
				break;
			}
			if (math.lengthsq(inst.points[neighbor] - point) < math.lengthsq(inst.points[best] - point)) {
				best = neighbor;
			}
		}
		return best;
	}
}
