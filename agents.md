# Shattered (Unity) — AGENTS.md

This repo is not worked on by autonomous AI agents.
I use Perplexity as an assistant: it reads the repository state (GitHub), I review its output, and I manually copy/paste the produced code.

These instructions exist to make Perplexity’s code output match this project’s conventions and be easy to paste into Unity.

## Output rules (strict)

When you propose code changes:

- Do not add “latest change”, “changed because…”, or similar commentary in code comments. Only add comments that are necessary to understand non-obvious intent.
- Do not introduce unnecessary variables if a value is used once; inline expressions when it stays readable.
- Output either:
  - The full changed file(s), or
  - The full changed function(s) / structure(s) (entire method/class/struct definition),
  with no placeholders like `/* ... unchanged ... */` or “rest of file omitted”, to make copy/paste safe.

## Change discipline

- Keep diffs small and localized.
- Avoid per-frame allocations in hot paths (`Relax`, neighbor loops, visualization updates).
- Feel free to alter existing types or interfaces if its required, but notify (in responce text, not comments when you do so)
- Prefer deterministic math and stable behavior over micro-optimizations.

## Repository map

Entry points:
- `Assets/Scripts/Meshless.cs`: Step loop (forces → solver iterations → commit deformation → integrate positions → update HNSW).
- `Assets/Scripts/Node.cs`: Per-node state (2D pos/vel, inverse mass, deformation state).
- `Assets/Scripts/HNSW.cs`: Approximate kNN structure (used for neighborhoods and updated via `Shift` when nodes move).

Physics core:
- `Assets/Scripts/Physics/NodeBatch.cs`: Per-step caches (neighbors, correction matrix `L`, cached `F0`, lambda, debug).
- `Assets/Scripts/Physics/DeformationUtils.cs`: Hencky/log strain model, polar decomposition, plastic return mapping.
- `Assets/Scripts/Physics/Constraints/XPBIConstraint.cs`: Velocity-based XPBD-style solve with constitutive constraint.
- `Assets/Scripts/Physics/Const.cs`: Global constants (eps, neighbor count, solver iterations).

Debug tooling:
- `Assets/Scripts/MeshlessVisualiser.cs`: Runtime visualization helper.
- `Assets/Scripts/Physics/DebugData.cs`: Per-node debug accumulation.
- `Assets/Scripts/Editor/ConstraintDebugWindow.cs`: Editor window for constraint/debug state.

## What the solver actually does (current implementation)

### Neighborhoods (fixed-k kNN, cached per step)
- Neighborhood size is `Const.NeighborCount` (currently 6).
- Neighbors are queried once per `NodeBatch` (per `StepSimulation`) and reused for all iterations in that step.
- After integrating positions, HNSW is updated via `hnsw.Shift(i, nodes[i].pos)`.

### Velocity-first corrections
- The constraint solver updates **velocities** directly (`Δv`).
- Positions are integrated once per step from corrected velocities.

### Trial deformation and cached reference
- At the start of the step, `XPBIConstraint.Initialise` caches `F0 = node.F` per node.
- Each iteration forms:
  - `Ftrial = (I + dt * gradV) * F0`
  where `gradV` is estimated from neighbor velocity differences.

### Correction matrix L (world-space, geometry-only)
- `L` is computed from world-space neighbor offsets `xij = xj - xi` in `NodeBatch.ComputeCorrectionMatrices()`.
- `L` depends only on neighborhood geometry, not on `F`/`Fp`.

### Constitutive model (Hencky/log strain)
- Energy/gradient uses polar decomposition and `log(stretch)` (Hencky strain), not Green strain `sym(F^T F - I)`.
- Constraint value is `C(F) = sqrt(2 * PsiHencky(F))`.

### Plasticity (Hencky-space return mapping + Fp update)
- Plastic projection is a return mapping in Hencky space (deviatoric yield + volumetric clamp).
- `CommitDeformation()` updates `Fp` based on `Ftrial` and the projected elastic `Fel`, then assigns `node.F = Fel`.

## Project knobs

- `Const.SolverIterations`: number of solver iterations per step.
- `Const.NeighborCount`: k in kNN neighborhood.
- `Meshless.compliance`: passed into `XPBIConstraint.Relax` and used to build the XPBD-style compliance term inside the solver.

Material-ish constants (currently global, in code):
- `XPBIConfig.YoungsModulus`, `XPBIConfig.PoissonsRatio`
- `XPBIConfig.YieldHencky`, `XPBIConfig.VolumetricHenckyLimit`

## Common pitfalls

- Neighbor caching: `NodeBatch` caches neighbors and `L`. If a change needs mid-step neighborhood refresh, you must invalidate/rebuild caches explicitly.
- Hot allocations: avoid allocating arrays/lists inside per-node/per-iteration loops.
- Unity.Mathematics: prefer `float2`/`float2x2` consistently; avoid mixing with `Vector2` unless necessary at Unity boundaries.
