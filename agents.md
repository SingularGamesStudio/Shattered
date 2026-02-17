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
- Original XPBI uses grid-based neighbor rebuilds; Shattered targets cases where coarse hierarchy levels become sparse and where tearing/merging makes global rebuilds expensive/unpleasant.
- For GPU neighbor extraction / proximity queries, Shattered can use a maintained 2D Delaunay triangulation to keep per-node cost predictable.


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


## 3) Hard constraints (perf + determinism)

Allocation discipline:
- Avoid per-frame allocations in hot paths: solver loops, neighbor loops/queries, NodeBatch init, V-cycle, visualization updates, GPU Delaunay glue.

GPU/CPU sync discipline:
- Avoid per-step GPU→CPU readbacks in hot paths (solver iterations, per-level relax loops).
- If readback is needed, do it where topology is rebuilt anyway (DT adjacency rebuild), then reuse cached CPU arrays.

Determinism:
- Prefer stable/deterministic behavior over micro-optimizations.
- Do not depend on hash iteration order anywhere that affects physics.
- Neighborhood accumulation order must be deterministic (neighbors angle-sorted).

Math/types:
- Prefer `Unity.Mathematics` types (`float2`, `float2x2`, etc.) internally.
- Use `Vector2` only at Unity boundaries if required.


## 4) Core algorithm (what happens each step)

The simulation is velocity-first:
- Constraints correct velocities (Δv).
- Positions integrate once per step from corrected velocities.

Hierarchical schedule (V-cycle-ish coarse → fine):
1) External forces modify velocities.
2) For each hierarchy level (coarse → fine):
   - Expand active prefix to include all nodes up to this level.
   - Initialise per-level caches (neighbors, h, correction matrices, volumes, cached reference deformation).
   - Run solver iterations (GPU colored Gauss–Seidel).
   - Prolongate parent velocity delta into children (deltas only).
3) Commit deformation on finest level (level 0).
4) Integrate positions once.
5) Update proximity structure / GPU Delaunay hierarchy after integration (positions upload + maintenance).

Prolongation rule (important):
- Only propagate parent Δv (current parent vel minus saved parent vel); do not overwrite child velocity with parent velocity.


## 5) Invariants (do not break)

Hierarchy:
- Nodes must remain sorted by `maxLayer` descending.
- `levelEndIndex[L]` is a prefix count: nodes with `maxLayer >= L`.
- `parentIndex` must refer to a valid parent on the next coarser level.

Neighborhoods:
- Fixed-k neighborhood size: `Const.NeighborCount`.
- Neighbor order must be deterministic before any accumulation (angle sort).

Execution model:
- Solver is velocity-first; don’t switch to per-iteration position integration.
- Hot loops must be allocation-free in steady-state.
- Do not introduce per-step GPU→CPU readback inside solver iterations.


## 6) File map (where to look)

Orchestration / entry points:
- `Assets/Scripts/SimulationController.cs`
  Timing, V-cycle schedule, prolongation, integration, spatial updates.
- `Assets/Scripts/Meshless.cs`
  Nodes container, hierarchy metadata (`levelEndIndex`), parameters, rebuild cadence.
- `Assets/Scripts/Node.cs`
  Node state: pos/vel/invMass/deformation/parentIndex/etc.

Physics core:
- `Assets/Scripts/Physics/NodeBatch.cs`
  Per-level caches: neighbors, per-node `h`, correction `L`, cached `F0`, lambda, effective volumes, debug.
- `Assets/Scripts/Physics/Constraints/XPBIConstraint.cs`
  `Relax`, `CommitDeformation`, gradV-based velocity corrections.
- `Assets/Scripts/Physics/DeformationUtils.cs`
  Polar decomposition, Hencky/log strain, plasticity helpers, pseudo-inverse helpers.
- `Assets/Scripts/Physics/Constraints/SPHKernels.cs`
  Wendland C2 kernel + gradient (2D).
- `Assets/Scripts/Physics/Const.cs`
  Constants: eps, iterations, neighbor count, `KernelHScale`, rebuild cadence.

Debug / tooling:
- `Assets/Scripts/LoopProfiler.cs`
- `Assets/Scripts/Physics/DebugData.cs`
- `Assets/Scripts/Editor/ConstraintDebugWindow.cs`

GPU XPBI solver (this thread):
- `Assets/Scripts/GPU/Solver/XPBI/XPBISolver.compute`
  Compute kernels for caching (volumes/h/L/F0/lambda), relax, prolongate, commit deformation, external forces.
  Colored GS: `RelaxColored` uses `_ColorOrder` + per-color ranges (`_ColorStart`, `_ColorCount`) and writes velocities directly.
- `Assets/Scripts/GPU/Solver/XPBIGPUSolver.cs`
  CPU-side driver: binds buffers, runs hierarchical schedule, builds/caches coloring, dispatches `RelaxColored` per color.

GPU Delaunay neighbors:
- `Assets/Scripts/GPU/Delaunay/DelaunayGPU.cs`
  GPU driver: init/maintain, adjacency rebuild, upload node positions.
  Neighbor caching/versioning for solver: keep CPU neighbor arrays synchronized on adjacency rebuild and expose an adjacency version counter.
- `Assets/Scripts/GPU/Delaunay/DelaunayHierarchyGPU.cs`
  Builds and maintains per-level DTs; adjacency rebuild happens after fix/legalize.
- `Assets/Scripts/GPU/Delaunay/DTBuilder.cs`
  CPU bootstrap triangulation + half-edge build.
- `Assets/Scripts/GPU/Delaunay/DelaunayTriangulation.compute`
  Kernels: fix edges, legalize, build neighbors.

Rendering / debug visualization:
- `Assets/Scripts/MeshlessTriangulationRenderer.cs`
  Procedural DT debug draw (fill + wire).
- `Assets/Shaders/MeshlessTriangulation.shader`
  DT fill shader (samples material texture array).
- `Assets/Shaders/MeshlessTriangulationWire.shader`
  DT wire-only overlay shader.
- `Assets/Scripts/MeshlessMaterialLibrary.cs`
  Bakes per-material sprites into a Texture2DArray for DT visualization.


## 7) GPU solver notes

Jacobi vs Gauss–Seidel on GPU:
- Old path: relax scatters Δv into a buffer using atomics, then applies Δv (Jacobi-style).
- New path: colored Gauss–Seidel: partition constraint-centers into colors and run colors sequentially; within each color, update velocities directly in parallel.

Coloring rule:
- A constraint centered at i writes to i and its neighborhood, so a conservative “2-hop conflict” rule is used for coloring (neighbors + neighbors-of-neighbors) to avoid write overlap within a color.
- Color count may be high (small per-color batches); treat as correctness-first, tune later.

No per-step readbacks:
- Coloring must not call `ComputeBuffer.GetData()` during solve iterations.
- Use CPU-cached neighbor arrays that are refreshed only when DT adjacency is rebuilt.

Per-level coloring cache:
- Cache per-level color metadata (`colorCount`, `colorStarts[]`, `colorCounts[]`) and the GPU `_ColorOrder` buffer contents.
- Rebuild only when topology changes (adjacency version changes) or when active prefix size changes for that level.


## 8) Rendering & materials

- Texture baking: sprite → Texture2DArray baking should not rely on `Graphics.CopyTexture` because format conversions/cropping can fail; prefer a bake path that supports conversion and cropping.
- Procedural draw: `Graphics.DrawProcedural` does not select shader passes; the `layer` argument is a Unity render layer. If you need “fill-only” vs “wire-only”, use separate shaders/materials.
- Coarse levels: higher DT levels are for wireframe visualization only; do not draw textured fill for them.
- Wireframe: render wire as an overlay (`ZWrite Off`), and keep thickness in pixel units to avoid “level 0 is huge” artifacts; if shared edges get darker, avoid alpha blending for the wire overlay.
- UV anchoring: to make the texture move with the object (stable under deformation), compute UVs from stored rest positions (rest in normalized DT space) rather than current/world positions.


## 9) References (primary)

- PBD: Müller et al., “Position Based Dynamics” (2007).
- XPBD: Macklin, Müller, Chentanez, “XPBD: Position-Based Simulation of Compliant Constrained Dynamics” (2016).
- XPBI: Yu et al., “XPBI: Position-Based Dynamics with Smoothing Kernels Handles Continuum Inelasticity” (2024).
- HPBD: Müller, “Hierarchical Position Based Dynamics” (HPBD).
- Dynamic Delaunay: Heinich Porro, Benoît Crespin. "Maintaining 2D Delaunay triangulations on the GPU for proximity queries of moving points"
