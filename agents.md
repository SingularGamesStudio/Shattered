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
