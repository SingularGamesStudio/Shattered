using System.Collections.Generic;
using System.Linq;
using Unity.Mathematics;
using UnityEngine;

public class Meshless: MonoBehaviour
{
	public void FixNode(int nodeIdx)
	{
		nodes[nodeIdx].isFixed = true;
		nodes[nodeIdx].invMass = 0.0f;
		nodes[nodeIdx].vel = float2.zero;
	}

	public HNSW hnsw;
	public List<Node> nodes = new List<Node>();
	[HideInInspector]
	public int maxLayer = -1;

	const float fixedTimeStep = 0.05f;
	const float gravity = -9.81f;
	const int constraintIters = 8;
	const int neighborCount = 6;

	public void Add(float2 pos)
	{
		Node newNode = new Node(pos, this);
		nodes.Add(newNode);
		if (newNode.maxLayer > maxLayer) {
			maxLayer = newNode.maxLayer;
		}
		nodes.Add(newNode);
	}

	public void Build()
	{
		nodes = nodes.OrderByDescending(node => node.maxLayer).ToList();
		hnsw = new HNSW(this);
	}

	private float keyHoldTime = 0.0f;
	const float holdThreshold = 0.2f; // seconds before considered held

	void Update()
	{
		if (Input.GetKeyDown(KeyCode.T)) {
			keyHoldTime = 0.0f; // Start timer on key down
		}

		if (Input.GetKey(KeyCode.T)) {
			keyHoldTime += Time.deltaTime;

			if (keyHoldTime >= holdThreshold) {
				// Held long enough: continuous stepping
				float delta = Time.deltaTime * 2;
				StepSimulation(delta);
			}
		}

		if (Input.GetKeyUp(KeyCode.T)) {
			if (keyHoldTime < holdThreshold) {
				// Considered a tap: single fixed step
				StepSimulation(fixedTimeStep);
			}

			keyHoldTime = 0.0f; // Reset timer on release
		}
	}

	// Manual simulation step triggered by button
	public void StepSimulation(float timeStep)
	{
		// 1. Apply external forces and predict positions
		foreach (var node in nodes) {
			if (node.isFixed) {
				node.predPos = node.pos;
				continue;
			}
			node.vel.y += gravity * timeStep;
			node.predPos = node.pos + node.vel * timeStep;
		}

		// Cache neighbors and their rest distances for this step
		List<int>[] cachedNeighbors = new List<int>[nodes.Count];
		List<float>[] cachedRestDistances = new List<float>[nodes.Count];

		for (int i = 0; i < nodes.Count; ++i) {
			var node = nodes[i];
			cachedNeighbors[i] = new List<int>();
			cachedRestDistances[i] = new List<float>();

			var neighbors = hnsw.SearchKnn(node.predPos, neighborCount);
			foreach (int nIdx in neighbors) {
				if (nIdx == i) continue;
				cachedNeighbors[i].Add(nIdx);

				float dist = math.distance(node.pos, nodes[nIdx].pos);
				cachedRestDistances[i].Add(dist);
			}
		}

		// 2. Constraint Solver Iterations
		for (int iter = 0; iter < constraintIters; ++iter) {
			for (int i = 0; i < nodes.Count; ++i) {
				Node nodeA = nodes[i];
				for (int n = 0; n < cachedNeighbors[i].Count; ++n) {
					int j = cachedNeighbors[i][n];
					Node nodeB = nodes[j];

					float2 delta = nodeA.predPos - nodeB.predPos;
					float dist = math.length(delta);
					if (dist < 1e-5f) continue;

					float restDistance = cachedRestDistances[i][n];
					float constraint = dist - restDistance;
					float wA = nodeA.invMass;
					float wB = nodeB.invMass;
					float wSum = wA + wB;
					if (wSum == 0) continue;

					float2 correction = (constraint / dist) * delta;
					if (!nodeA.isFixed)
						nodeA.predPos -= (wA / wSum) * correction;
					if (!nodeB.isFixed)
						nodeB.predPos += (wB / wSum) * correction;
				}
			}
		}


		// 3. Update velocities and positions
		for (int i = 0; i < nodes.Count; ++i) {
			if (nodes[i].isFixed) {
				nodes[i].vel = float2.zero;
				nodes[i].pos = nodes[i].predPos;
				continue;
			}
			float2 dampedPosition = math.lerp(nodes[i].pos, nodes[i].predPos, 0.9f);
			nodes[i].vel = (dampedPosition - nodes[i].pos) / timeStep * 0.9f;
			hnsw.Shift(i, dampedPosition);
		}
	}
}