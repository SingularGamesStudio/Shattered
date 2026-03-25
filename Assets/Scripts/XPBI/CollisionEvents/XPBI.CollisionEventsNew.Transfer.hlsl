// =============================================================================
// ContactPropagation.compute
//
// Propagates layer-0 (fine) contacts up to coarser multigrid layers.
//
// Dispatch per coarse layer L (C# loop, see bottom of file):
//   1. ClearCoarseContacts        – ceil(_MaxCoarseContacts / 64) groups
//   2. PropagateVertexContacts    – ceil(_FineContactCount * _CoarseParentsPerNode / 64)
//   3. PropagateEdgeContacts      – ceil(_FineContactCount * 4 / 64)
// =============================================================================

// Must match targetParentCount used in RebuildParentsAtLayer.
#define MAX_PROPAGATION_PARENTS 2

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

// Layout must match what EmitContact writes in the collision pass.
struct FineContact
{
    uint   ownerA;
    uint   ownerB;
    uint   vGiA;     // INVALID_U32 for edge-edge contacts
    uint   heA;      // half-edge on A (both vertex- and edge-contacts)
    uint   featB;    // vertex contact: SDF feature id on B
                     // edge   contact: half-edge id on B (heB)
    float2 n;        // contact normal, pointing B → A
    float  pen;      // penetration depth (> 0 = overlapping)
    float2 x;        // contact world position
    float2 cpB;      // closest point on B surface
};

// Coarse contact produced by this pass.
struct CoarseContact
{
    uint   ownerA;
    uint   ownerB;
    uint   coarseGiA;  // coarse node on A side
    uint   coarseGiB;  // coarse node on B side; INVALID_U32 if unresolved
    float2 n;
    float  pen;        // fine pen × combined parent weight
    float  weight;     // wA × wB; solver scales constraint by this
    float2 x;
};

// ---------------------------------------------------------------------------
// Uniforms
// ---------------------------------------------------------------------------

uint  _FineContactCount;        // snapshot of _ContactCount[0]
uint  _MaxCoarseContacts;       // output buffer capacity
uint  _CoarseParentsPerNode;    // = targetParentCount from RebuildParentsAtLayer

// ---------------------------------------------------------------------------
// Buffers
// ---------------------------------------------------------------------------

// Fine contacts (written by QueryVertexContacts / QueryEdgeEdgeContacts)
StructuredBuffer<FineContact>       _FineContacts;
StructuredBuffer<uint>              _FineContactCountBuffer;

// Parent map for this coarse layer – re-bound each dispatch by C#.
// Layout: [gi * _CoarseParentsPerNode + slot]  (matches RebuildParentsAtLayer output)
StructuredBuffer<int>               _LayerParentIndices;
StructuredBuffer<float>             _LayerParentWeights;

// Output
RWStructuredBuffer<CoarseContact>   _CoarseContacts;
RWBuffer<uint>                      _CoarseContactCount;  // [1], atomic

uint ReadFineContactCount() 
{
    return min(_FineContactCount, _FineContactCountBuffer[0]);
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

bool GetCoarseParent(uint fineGi, uint slot, out int coarseGi, out float w)
{
    coarseGi = -1;
    w        = 0.0;
    if (fineGi == INVALID_U32 || slot >= _CoarseParentsPerNode)
        return false;

    uint idx = fineGi * _CoarseParentsPerNode + slot;
    coarseGi = _LayerParentIndices[idx];
    w        = _LayerParentWeights [idx];
    return (coarseGi >= 0) && (w > 1e-9);
}

void EmitCoarseContact(uint ownerA, uint ownerB,
                       int  coarseGiA, int coarseGiB,
                       float2 n, float pen, float weight, float2 x)
{
    uint slot;
    InterlockedAdd(_CoarseContactCount[0], 1u, slot);
    if (slot >= _MaxCoarseContacts)
        return;

    CoarseContact cc;
    cc.ownerA    = ownerA;
    cc.ownerB    = ownerB;
    cc.coarseGiA = (uint)coarseGiA;
    cc.coarseGiB = (coarseGiB >= 0) ? (uint)coarseGiB : INVALID_U32;
    cc.n         = n;
    cc.pen       = pen * weight;
    cc.weight    = weight;
    cc.x         = x;
    _CoarseContacts[slot] = cc;
}

// Resolve B-side fine vertex from a SDF feature id.
// featB is either a half-edge id (edge feature) or a vertex half-edge id.
uint ResolveBSideFineVertex(uint featB, uint subVert)
{
    // Try half-edge endpoints first (covers both edge and vertex features,
    // since vertex features are also stored with a half-edge slot).
    uint vGiB = (subVert == 0u) ? _BoundaryEdgeV0Gi[featB]
                                 : _BoundaryEdgeV1Gi[featB];
    if (vGiB == INVALID_U32 && subVert == 0u)
        vGiB = _BoundaryVertexGi[featB];   // pure vertex feature fallback
    return vGiB;
}

// ---------------------------------------------------------------------------
// ClearCoarseContacts
// ---------------------------------------------------------------------------

[numthreads(64, 1, 1)]
void ClearCoarseContacts(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x == 0u)
        _CoarseContactCount[0] = 0u;

    if (tid.x < _MaxCoarseContacts)
    {
        CoarseContact cc = (CoarseContact)0;
        cc.ownerA    = INVALID_U32;
        cc.ownerB    = INVALID_U32;
        cc.coarseGiA = INVALID_U32;
        cc.coarseGiB = INVALID_U32;
        _CoarseContacts[tid.x] = cc;
    }
}

// ---------------------------------------------------------------------------
// PropagateVertexContacts
// ---------------------------------------------------------------------------
// Handles vertex-edge fine contacts.
//
// Thread layout:
//   workIdx = contactIdx * _CoarseParentsPerNode + pA
//
// A-side: expands over all _CoarseParentsPerNode parents of vGiA so every
//         coarse node that influences the fine contact gets a constraint copy,
//         weighted by its parent contribution.
//
// B-side: uses only the best (slot-0) coarse parent of the B-boundary vertex.
//         Expanding B-side too would create O(parents²) coarse contacts per
//         fine contact; empirically the nearest-parent approximation on B is
//         sufficient because the B boundary is the reference surface.
// ---------------------------------------------------------------------------

[numthreads(64, 1, 1)]
void PropagateVertexContacts(uint3 tid : SV_DispatchThreadID)
{
    uint workIdx    = tid.x;
    uint contactIdx = workIdx / _CoarseParentsPerNode;
    uint pA         = workIdx - contactIdx * _CoarseParentsPerNode;

    if (contactIdx >= ReadFineContactCount()) return;

    FineContact fc = _FineContacts[contactIdx];
    if (fc.vGiA == INVALID_U32)                       return; // edge-edge, skip
    if (fc.ownerA == INVALID_U32 || fc.ownerB == INVALID_U32) return;

    // ── A-side: pA-th coarse parent of the contacting fine vertex ────────────
    int   coarseGiA;
    float wA;
    if (!GetCoarseParent(fc.vGiA, pA, coarseGiA, wA))
        return;

    // ── B-side: nearest coarse parent of the B boundary vertex ───────────────
    // Use v0 of the feature half-edge; v1 as fallback for interior params.
    uint fineGiB = ResolveBSideFineVertex(fc.featB, 0u);

    int   coarseGiB = -1;
    float wB        = 1.0;
    if (fineGiB != INVALID_U32)
    {
        int   g; float w;
        if (GetCoarseParent(fineGiB, 0u, g, w))
        {
            coarseGiB = g;
            wB        = w;
        }
    }

    float weight = wA * wB;
    if (weight <= 1e-9) return;

    EmitCoarseContact(fc.ownerA, fc.ownerB,
                      coarseGiA, coarseGiB,
                      fc.n, fc.pen, weight, fc.x);
}

// ---------------------------------------------------------------------------
// PropagateEdgeContacts
// ---------------------------------------------------------------------------
// Handles edge-edge fine contacts.
//
// Thread layout:
//   workIdx = contactIdx * 4 + pairIdx
//   pairIdx ∈ {0,1,2,3} → (pA, pB) = (0,0),(0,1),(1,0),(1,1)
//
// Each edge endpoint on A is paired with each endpoint on B.
// Weight 0.25 normalises so the 4 sub-contacts together represent the full
// fine contact (assuming equal endpoint contributions at both ends).
//
// We only use slot-0 parents here; edge-edge contacts are naturally lower-
// frequency events, so a single dominant parent per endpoint is adequate.
// ---------------------------------------------------------------------------

[numthreads(64, 1, 1)]
void PropagateEdgeContacts(uint3 tid : SV_DispatchThreadID)
{
    const uint stride = 4u;

    uint workIdx    = tid.x;
    uint contactIdx = workIdx / stride;
    uint pairIdx    = workIdx - contactIdx * stride;
    uint pA         = pairIdx >> 1u;
    uint pB         = pairIdx &  1u;

    if (contactIdx >= ReadFineContactCount()) return;

    FineContact fc = _FineContacts[contactIdx];
    if (fc.vGiA != INVALID_U32)                       return; // vertex-edge, skip
    if (fc.ownerA == INVALID_U32 || fc.ownerB == INVALID_U32) return;

    // ── Resolve fine vertex GIs for the chosen endpoints ─────────────────────
    uint fineGiA = (pA == 0u) ? _BoundaryEdgeV0Gi[fc.heA]    : _BoundaryEdgeV1Gi[fc.heA];
    uint fineGiB = (pB == 0u) ? _BoundaryEdgeV0Gi[fc.featB]  : _BoundaryEdgeV1Gi[fc.featB];

    if (fineGiA == INVALID_U32 || fineGiB == INVALID_U32) return;

    // ── Slot-0 coarse parents for both endpoints ──────────────────────────────
    int   coarseGiA; float wA;
    if (!GetCoarseParent(fineGiA, 0u, coarseGiA, wA)) return;

    int   coarseGiB = -1;
    float wB        = 1.0; 
    {
        int g; float w;
        if (GetCoarseParent(fineGiB, 0u, g, w))
        {
            coarseGiB = g;
            wB        = w;
        }
    }

    // 0.25 = 1/4 endpoints; total weight across all 4 sub-contacts sums to wA*wB
    float weight = 0.25 * wA * wB;
    if (weight <= 1e-9) return;

    EmitCoarseContact(fc.ownerA, fc.ownerB,
                      coarseGiA, coarseGiB,
                      fc.n, fc.pen, weight, fc.x);
}