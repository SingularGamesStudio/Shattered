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
- Hierarchical Position-Based Dynamics (HPBD) uses a multilayer / multigrid-like hierarchy to improve convergence under tight iteration budgets.

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
- Prefer persistent CPU arrays and persistent ComputeBuffers.

GPU/CPU sync discipline:
- Avoid per-tick `ComputeBuffer.GetData` and avoid CPU-side blocking on GPU.
- Avoid per-tick DT adjacency readback; DT neighbor buffers should remain GPU-resident for the solver.
- CPU readbacks are allowed only as:
  - Periodic async snapshots for debug / AI / culling bounds (low rate).
  - Explicit hierarchy rebuild / debug paths.

Determinism:
- Do not depend on hash iteration order anywhere that affects physics.


## 4) Core algorithm (what happens each step)

The simulation is velocity-first:
- External forces modify velocities.
- Constraints correct velocities (Δv).
- Positions integrate once per step from corrected velocities.

GPU-truth execution model:
- GPU buffers hold the authoritative simulation state (pos/vel/invMass/flags/restVolume/parentIndex/F/Fp).
- CPU `Meshless.nodes[*].pos` is NOT updated every tick in the GPU-truth path.
- Rendering uses DT buffers (`DT.PositionsBuffer`, `DT.HalfEdgesBuffer`, `DT.TriToHEBuffer`) and is independent of CPU node positions.

Time stepping:
- `SimulationController` runs a fixed tick dt = `1/targetTPS`.
- If the simulation cannot keep up, it slows down: it runs up to `maxTicksPerFrame` ticks and discards backlog (no death-spiral catch-up).

Per tick (per Meshless):
1) Apply gameplay forces (optional, event-based) to velocities on GPU.
2) Apply external forces on GPU.
3) Solve constraints hierarchically on GPU:
   - Layers are processed coarse → fine (`maxLayer..0`) when hierarchical is enabled, else layer 0 only.
   - Active prefix size is `NodeCount(layer)` (layer 0 uses `total`).
   - Cache volumes hierarchically (leaf → owner prefix), cache kernel radii, compute correction matrix `L`, reset lambdas, then run relax iterations using GPU 2-hop coloring + indirect dispatch.
   - Prolongate parent Δv to fine prefix when `layer > 0`.
   - Commit deformation on layer 0.
4) Integrate positions on GPU once (`pos += vel * dt`) for all nodes.
5) Update DT positions on GPU and maintain DT topology:
   - DT position buffers are updated from solver `pos` (normalized DT space).
   - DT maintenance runs every tick for ALL DT layers that exist (0..maxLayer), independent of whether the solver is using hierarchy, because rendering may draw coarse DT layers.
   - DT adjacency rebuild increments `AdjacencyVersion`.
   - DT neighbor/count CPU readback is disabled in the per-tick path unless explicitly requested.
6) Periodic maintenance (optional):
   - Parent index rebuild can run on GPU periodically for hierarchical solve.
   - CPU async position snapshots can be scheduled at a low rate (every N ticks) for debug/AI/culling bounds.


## 5) Invariants (do not break)

Hierarchy:
- Nodes must remain sorted by `maxLayer` descending after `Meshless.Build()`.
- `layerEndIndex[L]` is a prefix count: nodes with `maxLayer >= L`.
- `parentIndex` must refer to a valid parent on the next coarser layer (or -1 if none).

Neighborhoods:
- Fixed-k neighborhood size is `Const.NeighborCount` (DT hierarchy is initialized with it).
- Solver uses DT neighbor buffers on GPU; it must not require CPU neighbor arrays.

Execution model:
- Solver is velocity-first; don’t switch to per-iteration position integration.


## 6) File map (where to look)

Orchestration / entry points:
- `Assets/Scripts/SimulationController.cs`
  Tick loop (fixed dt, drop-backlog), GPU solver ownership/caching, periodic async snapshots, TPS overlay.
- `Assets/Scripts/Meshless.cs`
  Nodes container, hierarchy metadata (`layerEndIndex`), DT normalization, DT hierarchy build/maintenance/readback, CPU-side hierarchy build helpers.
- `Assets/Scripts/Node.cs`
  Node state: pos/vel/originalPos/invMass/isFixed, deformation (`F`, `Fp`), restVolume, `maxLayer`, `parentIndex`, `materialId`.

GPU XPBI solver:
- `Assets/Scripts/XPBI/XPBISolver.cs`
  GPU buffers (persistent), per-layer dispatch schedule, 2-hop coloring build/cache keyed by `DT.AdjacencyVersion`, GPU integration, GPU DT position update, optional gameplay force events, optional GPU parent rebuild.
- `Assets/Scripts/XPBI/Shaders/Solver.compute`
  Compute kernel entrypoints; includes shared code in:
  - `Solver.Shared.hlsl`
  - `Solver.Cache.hlsl`
  - `Solver.Relax.hlsl`
  - `Solver.Coloring.hlsl` (GPU coloring + indirect args build)
- `Assets/Scripts/XPBI/Shaders/Deformation.hlsl`
  Deformation utilities used by the compute solver (strain/plasticity helpers, etc.).
- `Assets/Scripts/XPBI/Shaders/Utils.hlsl`
  Shared math/utility helpers for the compute solver.

GPU Delaunay neighbors (hierarchical):
- `Assets/Scripts/DT/DT.cs`
  Per-layer DT: GPU buffers for half-edge topology + adjacency, maintains adjacency, exposes neighbor buffers and optional CPU caches, bumps `AdjacencyVersion` on rebuild.
- `Assets/Scripts/DT/DTHierarchy.cs`
  Builds DT per hierarchy layer, updates positions from node prefix, maintains, optional readback for CPU features.
- `Assets/Scripts/DT/DTBuilder.cs`
  CPU bootstrap triangulation + half-edge build routines.
- `Assets/Scripts/DT/DT.compute`
  DT compute kernels: fix edges, legalize, build adjacency, build render tri map.

Rendering / materials:
- `Assets/Scripts/Renderer.cs`
  Procedural draw of DT triangles (fill layer 0; wire for coarse layers), uses rest positions in normalized DT space for UV anchoring.
- `Assets/Shaders/Triangulation.shader`
  Fill shader, converts normalized DT space to world via `_NormCenter` / `_NormInvHalfExtent`.
- `Assets/Shaders/Wireframe.shader`
  Wireframe shader, same normalization path as fill.
- `Assets/Scripts/Material/MaterialLibrary.cs`
  Builds `Texture2DArray` for albedo visualization and a GPU buffer for physical params; maps `MaterialDef` to index.
- `Assets/Scripts/Material/MaterialDef.cs`
  Scriptable material definition: sprite + physical parameters.

Utilities:
- `Assets/Scripts/Const.cs`
  Simulation constants (iterations, neighbor count, rebuild cadence, eps, etc.).


## 7) GPU solver notes

Colored Gauss–Seidel:
- Solve uses per-layer 2-hop coloring and dispatches `RelaxColored` per color; within each color, constraints update velocities directly without overlap.
- Coloring is rebuilt only when DT adjacency changes (`DT.AdjacencyVersion`) or when active prefix size changes.
- Per-color scheduling does not use GPU→CPU readbacks; `DispatchIndirect` uses GPU-built args buffers.

Adjacency readback boundary:
- DT rebuilds adjacency on GPU and may read neighbors + counts to CPU for CPU-side features/debug.
- XPBI coloring and solve must not depend on CPU adjacency arrays.

Kernel presence:
- If the compute shader is missing kernels, solver logs an error and returns early; keep kernel names stable when editing `Solver.compute`.


## 8) Rendering & materials

- Both fill and wireframe shaders interpret DT positions in normalized DT space and convert to world using `_NormCenter` and `_NormInvHalfExtent`.
- Rest UV anchoring uses `Node.originalPos` normalized into DT space, so textures remain “painted” while nodes move.
- Coarse layers are wireframe-only (fill is drawn only for layer 0).


## 9) References (primary)

- PBD: Müller et al., “Position Based Dynamics” (2007).
- XPBD: Macklin, Müller, Chentanez, “XPBD: Position-Based Simulation of Compliant Constrained Dynamics” (2016).
- XPBI: Yu et al., “XPBI: Position-Based Dynamics with Smoothing Kernels Handles Continuum Inelasticity” (2024).
- HPBD: Müller, “Hierarchical Position Based Dynamics” (HPBD).
- Dynamic Delaunay: Heinich Porro, Benoît Crespin. "Maintaining 2D Delaunay triangulations on the GPU for proximity queries of moving points"
