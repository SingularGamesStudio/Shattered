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
- `Assets/Scripts/Physics/NodeBatch.cs`: Per-level caches (neighbors, kernel radius `h`, correction matrix `L`, cached `F0`, lambda, current volume, debug). Designed to be expanded across hierarchy levels.
- `Assets/Scripts/Physics/DeformationUtils.cs`: Hencky/log strain model, polar decomposition, plastic return mapping, pseudo-inverse helpers.
- `Assets/Scripts/Physics/Constraints/XPBIConstraint.cs`: Velocity-based XPBD/XPBI-style solve (`Relax`, `CommitDeformation`) using kernel-estimated `gradV`.
- `Assets/Scripts/Physics/Constraints/SPHKernels.cs`: Wendland C2 kernel + gradient (2D) used by corrected kernel operators.
- `Assets/Scripts/Physics/Const.cs`: Global constants (eps, neighbor count, solver iterations, HPBD iterations, `KernelHScale`).

Debug tooling:
- `Assets/Scripts/MeshlessVisualiser.cs`: Runtime visualization helper.
- `Assets/Scripts/Physics/DebugData.cs`: Per-node debug accumulation.
- `Assets/Scripts/Editor/ConstraintDebugWindow.cs`: Editor window for constraint/debug state.

## Architecture: Hierarchical solver

### High-level flow (SimulationController)

The simulation uses a hierarchical solver with a V-cycle-like coarse-to-fine schedule:

1. **External forces** applied to all nodes (gravity modifies velocities).
2. **V-cycle solve** from coarsest to finest level:
   - For each level L (from `maxLayer` down to 0):
     - Expand batch to include all nodes up to level L (active prefix).
     - Initialize batch (volumes, neighbors, kernel radii `h`, correction matrices `L`, cache `F0`, reset lambda/debug).
     - Run solver iterations (few on coarse levels, more on finest).
     - **Prolongate corrections (velocity-based)**: propagate parent velocity deltas to child nodes.
3. **Commit deformation**: Update `F`, `Fp` on the finest level (level 0).
4. **Integrate positions** once from corrected velocities.
5. **Update HNSW** via `Shift` for each moved node.

### Hierarchy structure (Meshless)

- Nodes are sorted descending by `maxLayer` (HNSW layer assignment).
- `levelEndIndex[L]` = number of nodes with `maxLayer >= L` (prefix of nodes list).
- Each node stores `parentIndex` = single closest parent at `maxLayer + 1`.
- Hierarchy rebuilt every `Const.HierarchyRebuildInterval` frames.

### Batch reuse across levels (NodeBatch)

- The intended usage is one `NodeBatch` allocated at max capacity and reused each step.
- `ExpandTo(nodeCount)` increases active `Count` without reallocation (except if node count grows).
- `Initialise()` is called per level and (currently) does:
  - **Volumes**: recomputed for the active prefix using hierarchical aggregation (see below).
  - **Neighbors**: recomputed for all active nodes (kNN).
  - **Kernel radii (`h`)**: recomputed for all active nodes.
  - **Correction matrices `L`**: recomputed for all active nodes (depend on neighbors, `h`, and volumes).
  - **Lambda/debug**: reset for all active nodes.
  - **F0 cache**: updated for newly-added nodes when expanding within a step.

### Velocity correction prolongation

After solving at level L > 0, propagate velocity deltas to finer nodes:
```
for each node i in (levelEndIndex[L], levelEndIndex[L-1]):
    parentDeltaV = parent.vel - saved_parent.vel
    node.vel += parentDeltaV
```
Only **deltas** are propagated, not absolute values.

## What the solver actually does (current implementation)

### Neighborhoods (fixed-k kNN, cached per level)

- Neighborhood size is `Const.NeighborCount` (currently 6).
- Neighbors queried via HNSW (kNN) each level.
- Neighbors are sorted by angle for deterministic accumulation order.

### Velocity-first corrections

- The constraint solver updates **velocities** directly (`Δv`).
- Positions are integrated once per step from corrected velocities.
- Hierarchical coupling is done by prolongating **velocity** deltas from coarse parents to fine children.

### Trial deformation and cached reference

- At level initialization, `NodeBatch` caches `F0 = node.F`.
- Each iteration forms:
  - `Ftrial = (I + dt * gradV) * F0`
  where `gradV` is estimated from neighbor velocity differences using kernel operators.

### Kernel operators (Wendland C2 + corrected gradients)

- Wendland C2 kernel and gradient are implemented in `Physics/Constraints/SPHKernels.cs`.
- Each node has a smoothing length `h` (support radius is `2h`).
- `h` is chosen per node from neighbor distances (median distance scaled by `Const.KernelHScale`), so it naturally increases on sparse hierarchy levels.

### Correction matrix L (XPBI-style)

- `L` is computed per node using a corrected kernel gradient matrix of the form:
  - `L = (Σ V_b * (∇W_b ⊗ x_pb))^(-1)`
- The inverse uses a pseudo-inverse (SVD) for stability.
- `L` is then used to correct kernel gradients (`L * ∇W`) when estimating `gradV` and constraint gradients.

### Constitutive model (Hencky/log strain)

- Energy/gradient uses polar decomposition and `log(stretch)` (Hencky strain).
- Constraint value is `C(F) = sqrt(2 * PsiHencky(F))`.

### Plasticity (Hencky-space return mapping + Fp update)

- Plastic projection is a return mapping in Hencky space (deviatoric yield + volumetric clamp).
- `CommitDeformation()` updates `Fp` based on `Ftrial` and projected elastic `Fel`, then assigns `node.F = Fel`.
- Only runs on finest level (level 0) after the V-cycle completes.

### Effective volumes (hierarchy-aware)

Kernel sums use an effective per-node volume `V_b`:

- Finest level: `V = restVolume * |det(F)|`.
- Coarse levels: each active node represents a cluster; its `currentVolume` is the sum of descendant leaf deformed volumes.
- Volume mapping to the active prefix uses parentIndex-chain ownership with DSU-style path compression to avoid repeated long walks.

## Project knobs

Solver parameters:
- `Const.Iterations`: solver iterations for finest level (level 0), default 8.
- `Const.HPBDIterations`: iterations per coarse level, default 2.
- `Const.NeighborCount`: k in kNN neighborhood, default 6.
- `Meshless.compliance`: passed to `XPBIConstraint.Relax` for XPBD compliance term.
- `Const.KernelHScale`: scaling applied to median neighbor distance to get per-node `h`.

Hierarchy parameters:
- `SimulationController.useHierarchicalSolver`: toggle V-cycle on/off.
- `Const.HierarchyRebuildInterval`: frames between hierarchy updates, default 60.

Material constants (global, in code):
- `XPBIConfig.YoungsModulus`, `XPBIConfig.PoissonsRatio`.
- `XPBIConfig.YieldHencky`, `XPBIConfig.VolumetricHenckyLimit`.

## Common pitfalls

- **Hot allocations**: Avoid allocating arrays/lists inside per-node/per-iteration/per-level loops.
- **Level ordering**: Nodes must stay sorted by `maxLayer` descending for `levelEndIndex` prefixes to work correctly.
- **Unity.Mathematics**: Prefer `float2`/`float2x2` consistently; avoid mixing with `Vector2` unless necessary at Unity boundaries.