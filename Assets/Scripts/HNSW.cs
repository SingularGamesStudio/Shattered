using System;
using System.Collections.Generic;
using System.Linq;
using Unity.Mathematics;
using UnityEngine;

public class HNSW {
    public Meshless mesh;
    public List<int> instability;
    public int totalInstability = 0;
    private int M; // max connections per node per layer
    private int efConstruction;

    public HNSW(Meshless mesh, int maxConnections = 8, int efConstruction = 4) {
        instability = new List<int>();
        M = maxConnections;
        this.efConstruction = efConstruction;
        this.mesh = mesh;

        // init node 0
        instability.Add(0);
        mesh.nodes[0].HNSWNeighbors = new List<HashSet<int>>(mesh.maxLayer + 1);
        for (int j = 0; j <= mesh.maxLayer; j++)
            mesh.nodes[0].HNSWNeighbors.Add(new HashSet<int>());

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

    // Search for ef neighbors at a layer using a best-first search with candidate set
    private List<int> SearchLayer(float2 query, List<int> entryPoints, int layer, int ef) {
        var visited = new HashSet<int>();
        var candidates = new SortedSet<(float dist, int i)>(Comparer<(float, int)>.Create((a, b) => {
            int c = a.Item1.CompareTo(b.Item1);
            if (c == 0) return a.Item2.CompareTo(b.Item2);
            return c;
        }));
        var closest = new SortedSet<(float dist, int i)>(Comparer<(float, int)>.Create((a, b) => {
            int c = a.Item1.CompareTo(b.Item1);
            if (c == 0) return a.Item2.CompareTo(b.Item2);
            return c;
        }));

        // Initialize candidate and closest sets with entry points (distinct)
        foreach (var e in entryPoints.Distinct()) {
            float d = math.distancesq(mesh.nodes[e].pos, query);
            candidates.Add((d, e));
            closest.Add((d, e));
            visited.Add(e);
        }

        while (candidates.Count > 0) {
            var current = candidates.Min; // candidate with smallest distance
            candidates.Remove(current);


            var worstClosest = closest.Max; // element with largest distance in closest set
            if (current.dist > worstClosest.dist && closest.Count >= ef)
                break; // stop search according to heuristic

            foreach (var neighbor in mesh.nodes[current.i].HNSWNeighbors[layer]) {
                if (visited.Contains(neighbor))
                    continue;
                visited.Add(neighbor);
                float dist = math.distancesq(mesh.nodes[neighbor].pos, query);

                var worst = closest.Max;

                if (closest.Count < ef || dist < worst.dist) {
                    candidates.Add((dist, neighbor));
                    closest.Add((dist, neighbor));
                    if (closest.Count > ef) {
                        closest.Remove(closest.Max);
                    }
                }
            }
        }
        return closest.Select(x => x.i).ToList();
    }

    // Prune neighbors keeping closest M neighbors to the node at given layer, disconnect others
    private void PruneNeighbors(int node, int layer) {
        var neighbors = mesh.nodes[node].HNSWNeighbors[layer];

        if (neighbors.Count <= M) {
            return;
        }

        var sortedByDist = neighbors
            .OrderBy(i => math.distancesq(mesh.nodes[node].pos, mesh.nodes[i].pos))
            .Take(M)
            .ToHashSet();

        var toRemove = neighbors.Except(sortedByDist).ToList();

        foreach (var r in toRemove) {
            neighbors.Remove(r);
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
        mesh.nodes[i].HNSWNeighbors = new List<HashSet<int>>(nodeLayer + 1);
        for (int j = 0; j <= nodeLayer; j++)
            mesh.nodes[i].HNSWNeighbors.Add(new HashSet<int>());

        int layerEntry = 0;
        if (i == 0) {
            layerEntry = 1;
        }

        // Start from top layer, search for closest node to new point on each layer, descending layers until nodeLayer+1
        for (int layer = mesh.nodes[layerEntry].maxLayer; layer > nodeLayer; layer--) {
            layerEntry = SearchLayerSingle(mesh.nodes[i].pos, layerEntry, layer);
        }

        // For all layers from min(nodeLayer, maxLayer) down to 0:
        // Use the neighbors found from previous layer as the entry points for layer search
        var prevLayerNeighbors = new List<int> { layerEntry };

        for (int layer = Math.Min(nodeLayer, mesh.nodes[layerEntry].maxLayer); layer >= 0; layer--) {
            // search efConstruction neighbors on this layer starting from prevLayerNeighbors
            var neighbors = SearchLayer(mesh.nodes[i].pos, prevLayerNeighbors, layer, efConstruction);
            // connect new node with neighbors on this layer
            foreach (var neighbor in neighbors) {
                mesh.nodes[i].HNSWNeighbors[layer].Add(neighbor);
                mesh.nodes[neighbor].HNSWNeighbors[layer].Add(i);

                // If neighbor exceeds M connections, prune
                if (mesh.nodes[neighbor].HNSWNeighbors[layer].Count > M) {
                    PruneNeighbors(neighbor, layer);
                }
            }
            // new node's neighbors become entry points for next lower layer
            prevLayerNeighbors = neighbors;
        }
    }

    // Move a Node into a new position. The further it moves, the greater the instability.
    public void Shift(int node, float2 newPos, bool rebuild = false) {
        mesh.nodes[node].pos = newPos;
        for (int layer = mesh.nodes[node].maxLayer; layer >= 0; layer--) {
            var neighbors = SearchLayer(newPos, mesh.nodes[node].HNSWNeighbors[layer].ToList(), layer, efConstruction + 1);

            foreach (var neighbor in neighbors) {
                if (node == neighbor || mesh.nodes[node].HNSWNeighbors[layer].Contains(neighbor)) {
                    continue;
                }
                mesh.nodes[node].HNSWNeighbors[layer].Add(neighbor);
                mesh.nodes[neighbor].HNSWNeighbors[layer].Add(node);

                // If neighbor exceeds M connections, prune
                if (mesh.nodes[neighbor].HNSWNeighbors[layer].Count > M) {
                    PruneNeighbors(neighbor, layer);
                }
            }
            PruneNeighbors(node, layer);
        }
        //Debug.Log("HNSW Instability: " + totalInstability.ToString());
    }

    // Approximate nearest neighbors search querying k closest neighbors
    public List<int> SearchKnn(float2 queryPoint, int k, int minLayer = 0, int efSearch = 16) {
        int currentNode = 0;
        // Search top-down greedy to get close entry point
        for (int layer = mesh.maxLayer; layer >= minLayer + 1; layer--) {
            currentNode = SearchLayerSingle(queryPoint, currentNode, layer);
        }
        // At base layer use full search with efSearch neighbors
        var candidateList = SearchLayer(queryPoint, new List<int> { currentNode }, minLayer, efSearch);
        // Take top k closest
        var sortedTopK = candidateList
            .OrderBy(i => math.distancesq(mesh.nodes[i].pos, queryPoint))
            .Take(k)
            .ToList();

        return sortedTopK;
    }
}