# Shattered (Unity) — AGENTS.md

This repo is not worked on by autonomous AI agents.
I use Perplexity as an assistant: it reads the repository state (GitHub), I review its output, and I manually copy/paste the produced code.

These instructions exist to make Perplexity's code output match this project's conventions and be easy to paste into Unity.

## Output rules (strict)

When you propose code changes:

- Do not add "latest change", "changed because…", or similar commentary in code comments. Only add comments that are necessary to understand non-obvious intent.
- But never remove existing comments, unless they became obsolete after the change.
- Do not introduce unnecessary variables if a value is used once; inline expressions when it stays readable.
- Output either:
  - The full changed file(s), or
  - The full changed function(s) / structure(s) (entire method/class/struct definition),
  with no placeholders like `/* ... unchanged ... */`, to make copy/paste safe.

## Change discipline

- Keep diffs small and localized.
- Avoid per-frame allocations in hot paths (`Relax`, neighbor loops, visualization updates, hierarchical V-cycle).
- Feel free to alter existing types or interfaces if required, but notify (in response text, not comments) when you do so.
- Prefer deterministic math and stable behavior over micro-optimizations.

## Repository map

Entry points:
- `Assets/Scripts/SimulationController.cs`: Top-level orchestrator (timing, hierarchical V-cycle, force application, integration).
- `Assets/Scripts/Meshless.cs`: Per-object data container (nodes, HNSW, hierarchy metadata, parameters).
- `Assets/Scripts/Node.cs`: Per-node state (2D pos/vel, inverse mass, deformation state, hierarchy parent).
- `Assets/Scripts/HNSW.cs`: Approximate kNN structure (used for neighborhoods, hierarchy parent lookup, updated via `Shift`).

Physics core:
- `Assets/Scripts/Physics/NodeBatch.cs`: Per-step caches (neighbors, correction matrix `L`, cached `F0`, lambda, current volume, debug). Expandable batch reused across hierarchical levels.
- `Assets/Scripts/Physics/DeformationUtils.cs`: Hencky/log strain model, polar decomposition, plastic return mapping.
- `Assets/Scripts/Physics/Constraints/XPBIConstraint.cs`: Static class with velocity-based XPBD-style solve (`Relax`, `CommitDeformation`).
- `Assets/Scripts/Physics/Const.cs`: Global constants (eps, neighbor count, solver iterations, HPBD iterations).

Debug tooling:
- `Assets/Scripts/MeshlessVisualiser.cs`: Runtime visualization helper.
- `Assets/Scripts/Physics/DebugData.cs`: Per-node debug accumulation.
- `Assets/Scripts/Editor/ConstraintDebugWindow.cs`: Editor window for constraint/debug state.

## Architecture: Hierarchical solver

### High-level flow (SimulationController)
The simulation uses a **hierarchical position-based dynamics (HPBD)** approach with V-cycle multigrid structure:

1. **External forces** applied to all nodes (gravity)
2. **V-cycle solve** from coarsest to finest level:
   - For each level L (from `maxLayer` down to 0):
     - Save current positions
     - Expand batch to include all nodes up to level L
     - Initialize batch (neighbors, correction matrices, volumes, F0)
     - Run solver iterations (2 for coarse levels, 8 for finest)
     - **Prolongate corrections**: propagate parent position changes to child nodes
3. **Commit deformation**: Update `F`, `Fp` on finest level (level 0)
4. **Integrate positions** from corrected velocities
5. **Update HNSW** via `Shift` for each moved node

### Hierarchy structure (Meshless)
- Nodes are sorted descending by `maxLayer` (HNSW layer assignment)
- `levelEndIndex[L]` = number of nodes with `maxLayer >= L` (prefix of nodes list)
- Each node stores `parentIndex` = single closest parent at `maxLayer + 1`
- Hierarchy rebuilt every `Const.HierarchyRebuildInterval` frames

### Batch reuse across levels (NodeBatch)
- Single `NodeBatch` created with max capacity = total node count
- `ExpandTo(nodeCount)` increases active `Count` without reallocation
- `Initialise()` called per level:
  - **Volumes**: computed only for newly added nodes (`lastInitializedCount` to `Count`)
  - **Neighbors**: recomputed for ALL (topology changes with new nodes)
  - **Correction matrices L**: recomputed for ALL (depend on neighbors)
  - **F0, lambda**: reset for all active nodes

### Position correction prolongation
After solving at level L > 0, propagate corrections to finer nodes:
```
for each node i in (levelEndIndex[L], levelEndIndex[L-1]):
    parentCorrection = parent.pos - saved_parent.pos
    node.pos += parentCorrection
```
Only **corrections** (deltas) are propagated, not absolute positions, preserving fine-scale detail.

## What the solver actually does (current implementation)

### Neighborhoods (fixed-k kNN, cached per level)
- Neighborhood size is `Const.NeighborCount` (currently 6)
- Neighbors queried once per level via HNSW layer 0 (finest)
- Recomputed at each hierarchical level since topology changes

### Velocity-first corrections
- The constraint solver updates **velocities** directly (`Δv`)
- Positions integrated once per step from corrected velocities
- Hierarchical corrections modify positions between levels, then velocities refined at finer level

### Trial deformation and cached reference
- At level initialization, `batch.Initialise()` caches `F0 = node.F` per node
- Each iteration forms:
  - `Ftrial = (I + dt * gradV) * F0`
  where `gradV` is estimated from neighbor velocity differences

### Correction matrix L (world-space, geometry-only)
- `L` is computed from world-space neighbor offsets `xij = xj - xi` in `NodeBatch.ComputeCorrectionMatrices()`
- Uses simplified 1/r² weighting (not SPH kernels yet)
- `L` depends only on neighborhood geometry, not on `F`/`Fp`

### Constitutive model (Hencky/log strain)
- Energy/gradient uses polar decomposition and `log(stretch)` (Hencky strain)
- Constraint value is `C(F) = sqrt(2 * PsiHencky(F))`

### Plasticity (Hencky-space return mapping + Fp update)
- Plastic projection is a return mapping in Hencky space (deviatoric yield + volumetric clamp)
- `CommitDeformation()` updates `Fp` based on `Ftrial` and projected elastic `Fel`, then assigns `node.F = Fel`
- Only runs on finest level (level 0) after V-cycle completes

## Project knobs

Solver parameters:
- `Const.Iterations`: solver iterations for finest level (level 0), default 8
- `Const.HPBDIterations`: iterations per coarse level, default 2
- `Const.NeighborCount`: k in kNN neighborhood, default 6
- `Meshless.compliance`: passed to `XPBIConstraint.Relax` for XPBD compliance term

Hierarchy parameters:
- `SimulationController.useHierarchicalSolver`: toggle V-cycle on/off
- `Const.HierarchyRebuildInterval`: frames between hierarchy updates, default 60

Material constants (global, in code):
- `XPBIConfig.YoungsModulus`, `XPBIConfig.PoissonsRatio`
- `XPBIConfig.YieldHencky`, `XPBIConfig.VolumetricHenckyLimit`

## Common pitfalls

- **Batch expansion**: `NodeBatch.ExpandTo()` increases `Count`, but caches must be explicitly invalidated/recomputed via `Initialise()`.
- **Hot allocations**: Avoid allocating arrays/lists inside per-node/per-iteration/per-level loops. Batch and saved positions array are pre-allocated and reused.
- **Hierarchy consistency**: Parent indices can become stale if HNSW layers change; rebuild happens periodically but not every frame.
- **Level ordering**: Nodes must stay sorted by `maxLayer` descending for `levelEndIndex` prefixes to work correctly.
- **Unity.Mathematics**: Prefer `float2`/`float2x2` consistently; avoid mixing with `Vector2` unless necessary at Unity boundaries.

## Future work: SPH kernels

Current implementation uses simplified 1/r² weighting. Plan to add proper SPH kernels:
- Wendland C2 kernel evaluation
- Per-node effective radius based on representative volume (for hierarchical sparse levels)
- Volume-weighted correction matrix: `L = (Σ V_j ∇W(r_ij, h) ⊗ x_ij)^(-1)`
- This will make the correction matrix physically accurate for both dense (fine) and sparse (coarse) levels