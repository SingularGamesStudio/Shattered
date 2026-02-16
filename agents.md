# Shattered (Unity) — Perplexity context (AGENTS.md)

Shattered is a Unity project for real-time 2D deformable simulation (meshless / particle-like nodes) with:
- Velocity-first constraint solving (XPBI-style).
- A hierarchical coarse-to-fine (V-cycle-like) solver schedule.
- Neighborhood queries backed by a spatial structure (HNSW) and optionally GPU Delaunay-based neighbors.

I use Perplexity as an assistant: it reads limited GitHub state, I review output, and I manually copy/paste code into Unity.


## 1) Non-negotiables (copy/paste safety)

When proposing code changes:

- Output either:
  - Full changed file(s), OR
  - Full changed method/class/struct definitions (entire definition).
- No placeholders like `/* unchanged */`.
- No “latest change / changed because …” commentary inside code comments.
- Never remove existing comments unless they are obsolete after your change.
- Avoid unnecessary variables when an expression is used once and stays readable.
- Keep diffs small and localized.

If you change any public surface (type / interface / serialized fields / public method signature),
call it out in your response text (not in code comments).


## 2) Hard constraints (perf + determinism)

Allocation discipline (steady-state):
- Avoid per-frame allocations in hot paths: solver iterations (`Relax`), neighbor loops, NodeBatch init, V-cycle loops, HNSW queries, visualization updates, GPU Delaunay maintenance / neighbor rebuild glue.

Determinism / stability:
- Prefer stable/deterministic behavior over micro-optimizations.
- Stabilize ordering where it affects accumulation; do not rely on hash iteration order.
- Neighbor accumulation must be deterministic (neighbors are angle-sorted).

Types:
- Prefer `Unity.Mathematics` (`float2`, `float2x2`, etc.) internally.
- Use `Vector2` only at Unity boundaries if required.


## 3) Project vocabulary (mental model)

Nodes:
- Each node has position/velocity and deformation state.
- Simulation updates velocities via constraints, then integrates positions once.

Neighborhood:
- Most operators depend on a fixed-size kNN neighborhood (k = `Const.NeighborCount`).
- Neighborhood ordering matters (angle-sorted for stable sums).

Hierarchy:
- Nodes are sorted by `maxLayer` descending.
- Level L uses a prefix of nodes (all nodes with `maxLayer >= L`).
- Each node stores `parentIndex` pointing to a single parent on the next coarser layer.


## 4) Solver: what happens each step (minimal)

High-level step:
1) Apply external forces to velocities.
2) Hierarchical solve (coarse → fine):
   - Expand active prefix for current level.
   - Initialise level caches.
   - Run solver iterations (fewer on coarse, more on fine).
   - Prolongate: add parent velocity delta to children (deltas only).
3) Commit deformation on the finest level (level 0).
4) Integrate positions once from corrected velocities.
5) Update spatial structure (HNSW `Shift`) for moved nodes.

Prolongation rule (important):
- Only propagate parent Δv (current parent vel minus saved parent vel); do not overwrite child velocity with parent velocity.


## 5) Invariants (do not break)

Hierarchy / indexing:
- Nodes must remain sorted by `maxLayer` descending.
- `levelEndIndex[L]` is a prefix count: nodes with `maxLayer >= L`.
- `parentIndex` must refer to a valid parent on the next layer.

Neighborhood:
- kNN size is fixed (`Const.NeighborCount`).
- Neighbors must be angle-sorted before any accumulation that affects physics.

Execution:
- Solver is velocity-first; do not switch to per-iteration position integration.
- Hot loops must be allocation-free in steady-state.


## 6) Where to edit (routing index)

Entry points / orchestration:
- `Assets/Scripts/SimulationController.cs`
  Timing, V-cycle schedule, prolongation, integration, HNSW shift calls.

Core data & hierarchy:
- `Assets/Scripts/Meshless.cs`
  Nodes container, hierarchy metadata (`levelEndIndex`), parameters.
- `Assets/Scripts/Node.cs`
  Node state (pos/vel/invMass/deformation/parentIndex/etc).

Neighborhood queries:
- `Assets/Scripts/HNSW.cs`
  kNN structure, queries, updates via `Shift`.

Physics core:
- `Assets/Scripts/Physics/NodeBatch.cs`
  Per-level caches: neighbors, per-node `h`, correction matrix `L`, cached `F0`, lambda, volumes, debug.
- `Assets/Scripts/Physics/Constraints/XPBIConstraint.cs`
  `Relax`, `CommitDeformation`, gradV-based velocity corrections.
- `Assets/Scripts/Physics/DeformationUtils.cs`
  Polar decomposition, Hencky/log strain, plastic return mapping, pseudo-inverse helpers.
- `Assets/Scripts/Physics/Constraints/SPHKernels.cs`
  Wendland C2 kernel + gradient (2D).
- `Assets/Scripts/Physics/Const.cs`
  Constants: neighbor count, iterations, eps, `KernelHScale`, rebuild intervals.

Debug / tooling:
- `Assets/Scripts/LoopProfiler.cs`
- `Assets/Scripts/MeshlessVisualiser.cs`
- `Assets/Scripts/Physics/DebugData.cs`
- `Assets/Scripts/Editor/ConstraintDebugWindow.cs`

## 7) GPU Delaunay neighbors (in-project notes)

This subsystem maintains a dynamic 2D triangulation on GPU and extracts a fixed-size neighbor list for real vertices.

Files:
- `Assets/Scripts/GPU/Delaunay/DelaunayGPU.cs`
  GPU driver / dispatch (`Init`, `Maintain`, `RebuildVertexAdjacency`, `UpdatePositionsFromNodes(...)`).
- `Assets/Scripts/GPU/Delaunay/DTBuilder.cs`
  CPU bootstrap triangulation + half-edge build.
- `Assets/Scripts/GPU/Delaunay/DelaunayTriangulation.compute`
  Kernels (`FixHalfEdges`, `LegalizeHalfEdges`, `BuildNeighbors`).
- `Assets/Scripts/GPU/Delaunay/DelaunayGpuTest.cs`
  Stress harness (not core).

Key design assumptions (don’t accidentally “simplify” them away):
- 3 “super” vertices are appended: `VertexCount = RealVertexCount + 3`.
- Triangles incident to super vertices are kept (not deleted).
- Predicates may run on normalized coordinates (uniform translate + scale) for conditioning.
- Maintenance is typically “fix” then “legalize”, then rebuild adjacency/neighbors.
- Neighbor extraction ignores super vertices.

If you propose changes here, be extra careful about:
- Allocation-free CPU glue.
- Stable neighbor list layout and bounds (fixed `_NeighborCount`).
- Locking / race safety (triangle-local locks / ownership rules).


## 8) Research pointers (for quick recall)

These are references used to guide implementation choices:
- XPBI: https://arxiv.org/html/2405.11694v2
- GPU-maintained 2D Delaunay triangulation for proximity queries:
  https://meshinglab.dcc.uchile.cl/publication/porro-hal-04029968/porro-hal-04029968.pdf