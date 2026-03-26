// ============================================================================
// ContactPropagation.NodeOwned.compute
//
// Propagates fine slave-owned node contacts to a coarser layer as
// slave-owned node contacts again.
//
// This file is for the unified fully parallel collision architecture:
// - fine layer uses per-node slave-owned manifolds
// - coarse layers also use per-node slave-owned manifolds
// - the same collision solve include can run on either layer by rebinding
//   the current layer's contact buffers
//
// Expected dispatch order per coarse layer:
//   1. ClearCoarseNodeContacts
//      groups = ceil(max(_ActiveCount, _ActiveCount * _CoarseNodeContactStride) / 64)
//   2. PropagateFineContactsToCoarse
//      groups = ceil(_FineContactCount * _CoarseParentsPerNode / 64)
//
// Assumptions:
// - fine contacts are already one-sided slave-owned node contacts
// - if symmetric response is desired, detection emitted both directions
// - parent map buffers are rebound for the current target coarse layer
// ============================================================================

#ifndef CONTACT_PROPAGATION_NODE_OWNED_INCLUDED
#define CONTACT_PROPAGATION_NODE_OWNED_INCLUDED

#define INVALID_U32 0xffffffffu

#ifndef COARSE_MANIFOLD_CAP
#define COARSE_MANIFOLD_CAP 4u
#endif

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

struct XPBI_NodeContact
{
    uint   ownerSlave;
    uint   ownerMaster;
    uint   slaveGi;

    uint   masterGi0;
    uint   masterGi1;
    float  masterW1;

    float2 n;
    float  pen;
    float  scale;
    float2 x;
    uint   flags;
};

// ---------------------------------------------------------------------------
// Uniforms
// ---------------------------------------------------------------------------

uint _FineContactCount;
uint _CoarseParentsPerNode;
uint _CoarseNodeContactStride;

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

// Layout: [fineGi * _CoarseParentsPerNode + slot]
StructuredBuffer<int>             _LayerParentIndices;
StructuredBuffer<float>           _LayerParentWeights;

// ---------------------------------------------------------------------------
// Outputs
// ---------------------------------------------------------------------------

// Per coarse-layer node-owned manifold.
RWStructuredBuffer<XPBI_NodeContact> _CoarseNodeContacts;
RWBuffer<uint>                       _CoarseNodeContactCount;
RWBuffer<uint>                       _CoarseNodeContactOverflow;
RWStructuredBuffer<float>            _CoarseNodeContactLambda;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

uint ReadFineNodeContactCount()
{
    return min(_FineContactCount, min(_FineNodeContactCount[0], _MaxFineNodeContacts));
}

uint LocalIndexFromGlobalTransfer(uint gi)
{
    if (_UseDtGlobalNodeMap != 0u)
    {
        int li = _DtGlobalToLayerLocalMap[gi];
        return (li >= 0) ? (uint)li : INVALID_U32;
    }

    int liUnmapped = (int)gi - (int)_DtLocalBase;
    return (liUnmapped >= 0) ? (uint)liUnmapped : INVALID_U32;
}

float SafeRsqrtProp(float x)
{
    return rsqrt(max(x, 1e-20));
}

float2 SafeNormalizeProp(float2 v)
{
    return v * SafeRsqrtProp(dot(v, v));
}

uint CoarseNodeContactIndex(uint li, uint slot)
{
    return li * _CoarseNodeContactStride + slot;
}

void ClearNodeContact(out XPBI_NodeContact c)
{
    c.ownerSlave  = INVALID_U32;
    c.ownerMaster = INVALID_U32;
    c.slaveGi     = INVALID_U32;

    c.masterGi0   = INVALID_U32;
    c.masterGi1   = INVALID_U32;
    c.masterW1    = 0.0;

    c.n           = 0.0;
    c.pen         = 0.0;
    c.scale       = 0.0;
    c.x           = 0.0;
    c.flags       = 0u;
}

bool GetCoarseParent(uint fineGi, uint slot, out int coarseGi, out float w)
{
    coarseGi = -1;
    w = 0.0;

    if (fineGi == INVALID_U32 || slot >= _CoarseParentsPerNode)
        return false;

    uint idx = fineGi * _CoarseParentsPerNode + slot;
    coarseGi = _LayerParentIndices[idx];
    w        = _LayerParentWeights[idx];

    return (coarseGi >= 0) && (w > 1e-9);
}

// Build one coarse master representation for the fine contact.
// The result is still node-to-surface:
// - either one coarse master node
// - or a coarse master segment with two coarse nodes + masterW1
//
// masterScale is the attenuation induced by mapping the fine master feature
// to the current coarse layer.
bool ResolveMasterCoarseContact(
    FineNodeContact fc,
    out int masterGi0,
    out int masterGi1,
    out float masterW1,
    out float masterScale)
{
    masterGi0 = -1;
    masterGi1 = -1;
    masterW1 = 0.0;
    masterScale = 0.0;

    if (fc.masterGi0 == INVALID_U32)
        return false;

    // Single-node master feature.
    if (fc.masterGi1 == INVALID_U32)
    {
        int p0;
        float wp0;
        if (!GetCoarseParent(fc.masterGi0, 0u, p0, wp0))
            return false;

        masterGi0 = p0;
        masterGi1 = -1;
        masterW1 = 0.0;
        masterScale = wp0;
        return (masterScale > 1e-9);
    }

    // Two-node master feature.
    float w1fine = saturate(fc.masterW1);
    float w0fine = 1.0 - w1fine;

    int p0, p1;
    float wp0, wp1;
    bool have0 = GetCoarseParent(fc.masterGi0, 0u, p0, wp0);
    bool have1 = GetCoarseParent(fc.masterGi1, 0u, p1, wp1);

    float c0 = have0 ? (w0fine * wp0) : 0.0;
    float c1 = have1 ? (w1fine * wp1) : 0.0;

    if (c0 <= 1e-9 && c1 <= 1e-9)
        return false;

    if (have0 && have1)
    {
        if (p0 == p1)
        {
            masterGi0 = p0;
            masterGi1 = -1;
            masterW1 = 0.0;
            masterScale = c0 + c1;
            return (masterScale > 1e-9);
        }

        masterGi0 = p0;
        masterGi1 = p1;
        masterScale = c0 + c1;
        if (!(masterScale > 1e-9))
            return false;

        masterW1 = c1 / masterScale;
        return true;
    }

    if (c0 > 1e-9)
    {
        masterGi0 = p0;
        masterGi1 = -1;
        masterW1 = 0.0;
        masterScale = c0;
        return true;
    }

    masterGi0 = p1;
    masterGi1 = -1;
    masterW1 = 0.0;
    masterScale = c1;
    return true;
}

void AppendCoarseNodeContact(XPBI_NodeContact c)
{
    if (c.ownerSlave == INVALID_U32 || c.ownerMaster == INVALID_U32)
        return;
    if (c.slaveGi == INVALID_U32)
        return;
    if (!(c.pen > 0.0) || !(c.scale > 1e-9))
        return;

    float n2 = dot(c.n, c.n);
    if (!(n2 > 1e-20))
        return;
    c.n *= rsqrt(n2);

    uint li = LocalIndexFromGlobalTransfer(c.slaveGi);
    if (li == INVALID_U32 || li >= _ActiveCount)
        return;

    uint slot;
    InterlockedAdd(_CoarseNodeContactCount[li], 1u, slot);

    if (slot >= _CoarseNodeContactStride)
    {
        _CoarseNodeContactOverflow[li] = 1u;
        return;
    }

    uint idx = CoarseNodeContactIndex(li, slot);
    _CoarseNodeContacts[idx] = c;
}

// ---------------------------------------------------------------------------
// ClearCoarseNodeContacts
// ---------------------------------------------------------------------------

[numthreads(64, 1, 1)]
void ClearCoarseNodeContacts(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;

    if (i < _ActiveCount)
    {
        _CoarseNodeContactCount[i] = 0u;
        _CoarseNodeContactOverflow[i] = 0u;
    }

    uint total = _ActiveCount * _CoarseNodeContactStride;
    if (i < total)
    {
        XPBI_NodeContact c;
        ClearNodeContact(c);
        _CoarseNodeContacts[i] = c;
        _CoarseNodeContactLambda[i] = 0.0;
    }
}

// ---------------------------------------------------------------------------
// PropagateFineContactsToCoarse
// ---------------------------------------------------------------------------
// Thread layout:
//   workIdx = fineIdx * _CoarseParentsPerNode + pS
//
// Each thread expands one fine slave-owned contact through one slave-parent slot.
// The master side is mapped once to a coarse node or coarse segment and the
// resulting coarse node-owned contact is appended directly to the slave coarse
// node manifold.
// ---------------------------------------------------------------------------

[numthreads(64, 1, 1)]
void PropagateFineContactsToCoarse(uint3 tid : SV_DispatchThreadID)
{
    uint workIdx = tid.x;

    if (_CoarseParentsPerNode == 0u)
        return;

    uint fineIdx = workIdx / _CoarseParentsPerNode;
    uint pS      = workIdx - fineIdx * _CoarseParentsPerNode;

    uint fineCount = ReadFineNodeContactCount();
    if (fineIdx >= fineCount)
        return;

    FineNodeContact fc = _FineNodeContacts[fineIdx];

    if (fc.ownerSlave == INVALID_U32 || fc.ownerMaster == INVALID_U32)
        return;
    if (fc.ownerSlave == fc.ownerMaster)
        return;
    if (fc.slaveGi == INVALID_U32)
        return;
    if (!(fc.pen > 0.0) || !(fc.scale > 1e-9))
        return;

    float n2 = dot(fc.n, fc.n);
    if (!(n2 > 1e-20))
        return;

    int coarseSlaveGi;
    float wS;
    if (!GetCoarseParent(fc.slaveGi, pS, coarseSlaveGi, wS))
        return;

    int coarseMasterGi0, coarseMasterGi1;
    float coarseMasterW1, masterScale;
    if (!ResolveMasterCoarseContact(
            fc,
            coarseMasterGi0, coarseMasterGi1,
            coarseMasterW1, masterScale))
        return;

    float scale = fc.scale * wS * masterScale;
    if (!(scale > 1e-9))
        return;

    XPBI_NodeContact cc;
    ClearNodeContact(cc);

    cc.ownerSlave  = fc.ownerSlave;
    cc.ownerMaster = fc.ownerMaster;
    cc.slaveGi     = (uint)coarseSlaveGi;

    cc.masterGi0   = (uint)coarseMasterGi0;
    cc.masterGi1   = (coarseMasterGi1 >= 0) ? (uint)coarseMasterGi1 : INVALID_U32;
    cc.masterW1    = coarseMasterW1;

    cc.n           = SafeNormalizeProp(fc.n);
    cc.pen         = fc.pen;
    cc.scale       = scale;
    cc.x           = fc.x;
    cc.flags       = fc.srcKind;

    AppendCoarseNodeContact(cc);
}

#endif