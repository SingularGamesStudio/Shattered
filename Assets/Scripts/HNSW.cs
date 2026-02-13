using System;
using System.Collections.Generic;
using Unity.Mathematics;
using UnityEngine;

public class HNSW {
    public Meshless mesh;
    public List<int> instability;
    public int totalInstability = 0;
    private int M; // max connections per node per layer
    private int efConstruction;

    private int[] visitedStamp;
    private int currentStamp = 1;

    private Heap<float, int, AscendingFloatComparer> candidates;
    private Heap<float, int, DescendingFloatComparer> closest;

    private readonly List<int> entryPointsScratch = new List<int>(32);
    private readonly List<int> layerResultScratch = new List<int>(32);
    private readonly List<(float dist, int i)> distScratch = new List<(float, int)>(32);

    public HNSW(Meshless mesh, int maxConnections = 8, int efConstruction = 4) {
        instability = new List<int>(mesh.nodes.Count);
        M = maxConnections;
        this.efConstruction = efConstruction;
        this.mesh = mesh;

        EnsureVisitedCapacity(mesh.nodes.Count);
        candidates = new Heap<float, int, AscendingFloatComparer>(64);
        closest = new Heap<float, int, DescendingFloatComparer>(64);

        // init node 0
        instability.Add(0);
        mesh.nodes[0].HNSWNeighbors = new List<NeighborSet>(mesh.maxLayer + 1);
        for (int j = 0; j <= mesh.maxLayer; j++)
            mesh.nodes[0].HNSWNeighbors.Add(new NeighborSet(M * 2));

        for (int i = 1; i < mesh.nodes.Count; i++) {
            BuildNode(i);
        }
        //Debug.Log("HNSW Instability: " + totalInstability.ToString());
    }

    // Greedy search at single layer to find closest node to query starting from entry point
    private int SearchLayerSingle(float2 query, int enterNode, int layer) {
        int currentNode = enterNode;
        float currentDist = math.distancesq(mesh.nodes[currentNode].pos, query);
        bool improved;
        do {
            improved = false;
            foreach (var n in mesh.nodes[currentNode].HNSWNeighbors[layer]) {
                float dist = math.distancesq(mesh.nodes[n].pos, query);
                if (dist < currentDist) {
                    currentDist = dist;
                    currentNode = n;
                    improved = true;
                }
            }
        } while (improved);

        return currentNode;
    }

    private void EnsureVisitedCapacity(int nodeCount) {
        if (visitedStamp == null || visitedStamp.Length < nodeCount)
            visitedStamp = new int[nodeCount];
    }

    private void NextStamp() {
        currentStamp++;
        if (currentStamp == int.MaxValue) {
            Array.Clear(visitedStamp, 0, visitedStamp.Length);
            currentStamp = 1;
        }
    }

    private static int CompareDistId((float dist, int i) a, (float dist, int i) b) {
        int c = a.dist.CompareTo(b.dist);
        if (c != 0) return c;
        return a.i.CompareTo(b.i);
    }

    // Search for ef neighbors at a layer using a best-first search with candidate set
    private void SearchLayer(float2 query, List<int> entryPoints, int layer, int ef, List<int> result) {
        result.Clear();

        EnsureVisitedCapacity(mesh.nodes.Count);
        NextStamp();

        candidates.Clear();
        closest.Clear();

        for (int idx = 0; idx < entryPoints.Count; idx++) {
            int e = entryPoints[idx];
            if (e < 0 || e >= mesh.nodes.Count)
                continue;

            if (visitedStamp[e] == currentStamp)
                continue;

            visitedStamp[e] = currentStamp;
            float d = math.distancesq(mesh.nodes[e].pos, query);
            candidates.Push(d, e);
            closest.Push(d, e);
        }

        if (closest.Count == 0)
            return;

        while (candidates.Count > 0) {
            candidates.Pop(out float candDist, out int candId);

            if (closest.Count >= ef && candDist > closest.PeekKey())
                break; // stop search according to heuristic

            foreach (var neighbor in mesh.nodes[candId].HNSWNeighbors[layer]) {
                if (neighbor < 0 || neighbor >= mesh.nodes.Count)
                    continue;

                if (visitedStamp[neighbor] == currentStamp)
                    continue;

                visitedStamp[neighbor] = currentStamp;
                float dist = math.distancesq(mesh.nodes[neighbor].pos, query);

                if (closest.Count < ef || dist < closest.PeekKey()) {
                    candidates.Push(dist, neighbor);
                    closest.Push(dist, neighbor);
                    if (closest.Count > ef)
                        closest.Pop(out _, out _);
                }
            }
        }

        closest.CopyValuesTo(result);
    }

    // Prune neighbors keeping closest M neighbors to the node at given layer, disconnect others
    private void PruneNeighbors(int node, int layer) {
        var neighbors = mesh.nodes[node].HNSWNeighbors[layer];

        if (neighbors.Count <= M)
            return;

        distScratch.Clear();
        foreach (var i in neighbors)
            distScratch.Add((math.distancesq(mesh.nodes[node].pos, mesh.nodes[i].pos), i));

        distScratch.Sort(CompareDistId);

        for (int idx = M; idx < distScratch.Count; idx++) {
            int r = distScratch[idx].i;
            if (!neighbors.Remove(r))
                continue;

            mesh.nodes[r].HNSWNeighbors[layer].Remove(node);
            destabilize(r);
        }
    }

    void destabilize(int node) {
        instability[node]++;
        totalInstability++;

        if (instability[node] >= mesh.nodes[node].HNSWNeighbors[0].Count) {
            //TODO: rebuild node
            totalInstability -= instability[node];
            instability[node] = 0;
        }
        //TODO: if totalInstability is too big, rebuild the whole graph
    }

    // Add a point to the HNSW graph
    public void BuildNode(int i) {
        instability.Add(0);
        var nodeLayer = mesh.nodes[i].maxLayer;
        mesh.nodes[i].HNSWNeighbors = new List<NeighborSet>(nodeLayer + 1);
        for (int j = 0; j <= nodeLayer; j++)
            mesh.nodes[i].HNSWNeighbors.Add(new NeighborSet(M * 2));

        int layerEntry = 0;
        if (i == 0) {
            layerEntry = 1;
        }

        // Start from top layer, search for closest node to new point on each layer, descending layers until nodeLayer+1
        for (int layer = mesh.nodes[layerEntry].maxLayer; layer > nodeLayer; layer--) {
            layerEntry = SearchLayerSingle(mesh.nodes[i].pos, layerEntry, layer);
        }

        entryPointsScratch.Clear();
        entryPointsScratch.Add(layerEntry);

        for (int layer = Math.Min(nodeLayer, mesh.nodes[layerEntry].maxLayer); layer >= 0; layer--) {
            SearchLayer(mesh.nodes[i].pos, entryPointsScratch, layer, efConstruction, layerResultScratch);

            for (int n = 0; n < layerResultScratch.Count; n++) {
                int neighbor = layerResultScratch[n];
                mesh.nodes[i].HNSWNeighbors[layer].Add(neighbor);
                mesh.nodes[neighbor].HNSWNeighbors[layer].Add(i);

                if (mesh.nodes[neighbor].HNSWNeighbors[layer].Count > M) {
                    PruneNeighbors(neighbor, layer);
                }
            }

            entryPointsScratch.Clear();
            for (int n = 0; n < layerResultScratch.Count; n++)
                entryPointsScratch.Add(layerResultScratch[n]);
        }
    }

    // Move a Node into a new position. The further it moves, the greater the instability.
    public void Shift(int node, float2 newPos, bool rebuild = false) {
        mesh.nodes[node].pos = newPos;
        for (int layer = mesh.nodes[node].maxLayer; layer >= 0; layer--) {
            entryPointsScratch.Clear();
            foreach (var n in mesh.nodes[node].HNSWNeighbors[layer])
                entryPointsScratch.Add(n);

            SearchLayer(newPos, entryPointsScratch, layer, efConstruction + 1, layerResultScratch);

            for (int n = 0; n < layerResultScratch.Count; n++) {
                int neighbor = layerResultScratch[n];
                if (node == neighbor || mesh.nodes[node].HNSWNeighbors[layer].Contains(neighbor)) {
                    continue;
                }
                mesh.nodes[node].HNSWNeighbors[layer].Add(neighbor);
                mesh.nodes[neighbor].HNSWNeighbors[layer].Add(node);

                if (mesh.nodes[neighbor].HNSWNeighbors[layer].Count > M) {
                    PruneNeighbors(neighbor, layer);
                }
            }
            PruneNeighbors(node, layer);
        }
        //Debug.Log("HNSW Instability: " + totalInstability.ToString());
    }

    public void SearchKnn(float2 queryPoint, int k, List<int> results, int minLayer = 0, int efSearch = 16) {
        results.Clear();

        int currentNode = 0;
        for (int layer = mesh.maxLayer; layer >= minLayer + 1; layer--) {
            currentNode = SearchLayerSingle(queryPoint, currentNode, layer);
        }

        entryPointsScratch.Clear();
        entryPointsScratch.Add(currentNode);

        SearchLayer(queryPoint, entryPointsScratch, minLayer, efSearch, layerResultScratch);

        if (layerResultScratch.Count == 0)
            return;

        distScratch.Clear();
        for (int i = 0; i < layerResultScratch.Count; i++) {
            int id = layerResultScratch[i];
            distScratch.Add((math.distancesq(mesh.nodes[id].pos, queryPoint), id));
        }
        distScratch.Sort(CompareDistId);

        int outCount = Math.Min(k, distScratch.Count);
        for (int i = 0; i < outCount; i++)
            results.Add(distScratch[i].i);
    }

    // Convenience allocating overload (keep for non-hot usage).
    public List<int> SearchKnn(float2 queryPoint, int k, int minLayer = 0, int efSearch = 16) {
        var results = new List<int>(k);
        SearchKnn(queryPoint, k, results, minLayer, efSearch);
        return results;
    }
}
