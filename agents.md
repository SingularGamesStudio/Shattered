# Shattered (Unity) — AGENTS.md (Perplexity assistant context)

Shattered is a Unity real-time deformables project focused on stable, controllable simulation of elastic + plastic behavior for gameplay.
The core solver is “velocity-first” (constraints apply Δv; positions integrate once per step) and is designed to stay stable under large time steps and limited iteration budgets.

Perplexity is used as a browser assistant:
- It can read limited GitHub repo state.
- I manually copy/paste code from Perplexity into Unity.
- I do not commit often mid-conversation.


## 0) What Shattered is trying to do (big picture)

Goals:
- Real-time soft bodies that are stable, controllable, and predictable in a game loop.
- Support both elastic response and long-term inelastic/plastic deformation without “rebuilding a mesh”.
- Keep per-step work bounded: fixed-k neighborhoods, limited solver iterations, allocation-free hot loops.
- Scale better when objects tear/merge: avoid global remeshing / global factorization.

Non-goals:
- Scientific-grade accuracy guarantees.
- Fully general FEM remeshing / fracture pipelines.
- Heavy grid-based methods that blow up memory/bandwidth budgets in typical Unity gameplay scenes.


## 1) Research baseline (what this is built from)

PBD:
- Classic Position-Based Dynamics manipulates positions via constraint projections for stability and controllability.

XPBD:
- XPBD extends PBD with compliant constraints so stiffness behaves consistently across time step and iteration count, instead of “getting stiffer” as you add iterations.

XPBI:
- XPBI explores extending XPBD-style ideas to continuum inelasticity using smoothing kernels and velocity-based formulations for updated Lagrangian deformation handling.

HPBD / hierarchical acceleration:
- Hierarchical Position-Based Dynamics (HPBD) uses a multilevel / multigrid-like hierarchy to improve convergence under tight iteration budgets.

Neighborhood search motivation:
- Shattered maintains a 2D Delaunay triangulation on GPU to keep per-node neighbor extraction predictable and to support a hierarchy of DTs.


## 2) Non-negotiables (output + paste safety)

When proposing code changes:
- Output either:
  - Full changed file(s), OR
  - Full changed method/class/struct definitions (entire definition).
- No placeholders like `/* ... unchanged ... */`.
- No partial snippets that require manual merging.
- Do not add “latest change / changed because …” commentary in code comments.
  Only add comments that explain non-obvious intent.
- Never remove existing comments unless they are obsolete after the change.
- Do not introduce unnecessary variables if a value is used once and stays readable.
- Keep diffs small and localized.

If you change any surface API (type, interface, serialized fields, public signatures), call it out in your response text (not in code comments).


## 3) Hard constraints (perf + sync)

Allocation discipline:
- Avoid per-frame allocations in hot paths: solver loops, neighbor loops/queries, hierarchy updates, visualization updates, DT glue.

GPU/CPU sync discipline:
- Avoid per-step GPU→CPU readbacks inside solver iteration loops.
- CPU readback of DT adjacency is allowed where DT adjacency is rebuilt anyway (DT rebuild increments an adjacency version and refreshes cached CPU arrays).

Determinism:
- Do not depend on hash iteration order anywhere that affects physics.


## 4) Core algorithm (what happens each step)

The simulation is velocity-first:
- External forces modify velocities.
- Constraints correct velocities (Δv).
- Positions integrate once per step from corrected velocities.

Hierarchical schedule (coarse → fine, GPU XPBI):
1) `SimulationController` advances ticks at `targetTPS` and calls GPU step for each active `Meshless`.
2) GPU solver uploads node state (pos/vel/invMass/flags/restVolume/parentIndex/F/Fp).
3) For each hierarchy level `maxLayer..0` (active prefix = `NodeCount(level)`):
   - Save velocity prefix (`SaveVelPrefix`).
   - Cache volumes hierarchically: clear active owners then accumulate leaves over total (`ClearCurrentVolume`, `CacheVolumesHierarchical`).
   - Cache per-node kernel radius `h` from DT neighbors (`CacheKernelH`).
   - Compute correction matrix `L` (Bonet–Lok style) from DT neighbors + volumes (`ComputeCorrectionL`).
   - Cache `F0` and reset `lambda` (`CacheF0AndResetLambda`).
   - Build / reuse 2-hop coloring using DT CPU neighbor arrays and `AdjacencyVersion` and upload `_ColorOrder` buffer if needed.
   - Run iterations (level 0 uses `Const.Iterations`, coarse levels use `Const.HPBDIterations`) by dispatching `RelaxColored` per color range.
   - Prolongate parent Δv into children prefix range (`Prolongate`) when `level > 0`.
   - Commit deformation only on level 0 (`CommitDeformation`).
4) Solver downloads vel/F/Fp back to `Meshless.nodes`.
5) CPU integrates positions once and updates DT positions + DT maintenance; DT readback is disabled in the per-tick path (`dtReadback: false`).
6) Periodically (every `Const.HierarchyRebuildInterval` ticks), `Meshless.BuildHierarchyWithDtReadback()` refreshes DT CPU adjacency and rebuilds `parentIndex`.


## 5) Invariants (do not break)

Hierarchy:
- Nodes must remain sorted by `maxLayer` descending after `Meshless.Build()`.
- `levelEndIndex[L]` is a prefix count: nodes with `maxLayer >= L`.
- `parentIndex` must refer to a valid parent on the next coarser level (or -1 if none).

Neighborhoods:
- Fixed-k neighborhood size is `Const.NeighborCount` (DT hierarchy is initialized with it).

Execution model:
- Solver is velocity-first; don’t switch to per-iteration position integration.
- Keep solve kernels present and named as required by `XPBI/XPBISolver.cs` (`HasAllKernels`).


## 6) File map (where to look)

Orchestration / entry points:
- `Assets/Scripts/SimulationController.cs`
  Timing, tick loop, GPU solver ownership/caching, integration, periodic hierarchy rebuild.
- `Assets/Scripts/Meshless.cs`
  Nodes container, hierarchy metadata (`levelEndIndex`), DT normalization, DT hierarchy build/maintenance/readback, parent rebuild.
- `Assets/Scripts/Node.cs`
  Node state: pos/vel/originalPos/invMass/isFixed, deformation (`F`, `Fp`), restVolume, `maxLayer`, `parentIndex`, `materialId`.

GPU XPBI solver:
- `Assets/Scripts/XPBI/XPBISolver.cs`
  CPU-side driver: GPU buffers, per-level dispatch schedule, 2-hop coloring build/cache keyed by `DT.AdjacencyVersion`, upload/download to Meshless.
- `Assets/Scripts/XPBI/Shaders/Solver.compute`
  Compute kernels for: forces, caching (volumes/h/L/F0/lambda), relax, prolongate, commit deformation. Only includes files, actual code is in `Solver.Relax.hlsl`, `Solver.Shared.hlsl`, `Solver.Cache.hlsl`
- `Assets/Scripts/XPBI/Shaders/Deformation.hlsl`
  Deformation utilities used by the compute solver (strain/plasticity helpers, etc.).
- `Assets/Scripts/XPBI/Shaders/Utils.hlsl`
  Shared math/utility helpers for the compute solver.

GPU Delaunay neighbors (hierarchical):
- `Assets/Scripts/DT/DT.cs`
  Per-level DT: GPU buffers for half-edge topology + adjacency, maintains adjacency, exposes neighbor buffers and CPU caches, bumps `AdjacencyVersion` on rebuild.
- `Assets/Scripts/DT/DTHierarchy.cs`
  Builds and maintains DT per hierarchy level; provides `ReadbackAllLevels()`, `FillNeighbors()`, and `FindNearestCoarseToFine()`.
- `Assets/Scripts/DT/DTBuilder.cs`
  CPU bootstrap triangulation + half-edge build routines.
- `Assets/Scripts/DT/DT.compute`
  DT compute kernels: fix edges, legalize, build adjacency, build render tri map.

Rendering / materials:
- `Assets/Scripts/Renderer.cs`
  Procedural draw of DT triangles (fill level 0; wire for all/selected levels), uses per-node material ids and rest positions in normalized DT space for UV anchoring.
- `Assets/Scripts/Material/MaterialLibrary.cs`
  Builds `Texture2DArray` for albedo visualization and a GPU buffer for physical params; maps `MaterialDef` to index.
- `Assets/Scripts/Material/MaterialDef.cs`
  Scriptable material definition: sprite + physical parameters.

Utilities:
- `Assets/Scripts/Const.cs`
  Simulation constants (iterations, neighbor count, rebuild cadence, eps, etc.).


## 7) GPU solver notes

Colored Gauss–Seidel:
- Solve uses per-level 2-hop coloring and dispatches `RelaxColored` per color; within each color, constraints update velocities directly without overlap.
- Coloring is rebuilt only when DT adjacency changes (`DT.AdjacencyVersion`) or when active prefix size changes.

Adjacency readback boundary:
- DT rebuilds adjacency on GPU and immediately reads neighbors + counts to CPU, and increments `AdjacencyVersion`.
- The XPBI solver relies on `NeighborsCpu` / `NeighborCountsCpu` for coloring, so ensure DT adjacency readback is performed whenever topology changes (currently handled by `DT.Maintain()` calling `RebuildVertexAdjacencyAndTriMap()`).

Kernel presence:
- If the compute shader is missing kernels, solver logs an error and returns early; keep kernel names stable when editing `Solver.compute`.


## 8) Rendering & materials

- The renderer uses “rest in normalized DT space” for UV anchoring (`Node.originalPos` normalized by `Meshless.DtNormCenter` / `DtNormInvHalfExtent`).
- Material sprites are baked into a `Texture2DArray` using a Blit→ReadPixels path in `MaterialLibrary.Rebuild()` to avoid format conversion issues.
- Coarse levels are wireframe-only (fill is drawn only for level 0).


## 9) References (primary)

- PBD: Müller et al., “Position Based Dynamics” (2007).
- XPBD: Macklin, Müller, Chentanez, “XPBD: Position-Based Simulation of Compliant Constrained Dynamics” (2016).
- XPBI: Yu et al., “XPBI: Position-Based Dynamics with Smoothing Kernels Handles Continuum Inelasticity” (2024).
- HPBD: Müller, “Hierarchical Position Based Dynamics” (HPBD).
- Dynamic Delaunay: Heinich Porro, Benoît Crespin. "Maintaining 2D Delaunay triangulations on the GPU for proximity queries of moving points"
