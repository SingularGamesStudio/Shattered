// ============================================================================
// CollisionStencilUtils.hlsl
// Shared declarations + generic helpers for collision stencils.
// No XPBI_* macros, no solve-path defines.
// ============================================================================

#ifndef COLLISION_STENCIL_UTILS_INCLUDED
#define COLLISION_STENCIL_UTILS_INCLUDED

static const uint COLLISION_INVALID_U32 = 0xffffffffu;
static const uint COLLISION_MAX_NODES   = 4u;

struct CollisionFineContact
{
    uint ownerA;
    uint ownerB;
    uint vGiA;
    uint heA;
    uint featB;
    float2 n;
    float pen;
    float2 x;
    float2 cpB;
};

struct CollisionCoarseContact
{
    uint ownerA;
    uint ownerB;
    uint coarseGiA;
    uint coarseGiB;
    float2 n;
    float pen;
    float weight;
    float2 x;
};

// ---------------------------------------------------------------------------
// Per-node -> contact CSR
// Each contact is owned by exactly one anchor node.
// _NodeCollisionRefStart is size (_ActiveCount + 1) for the current layer.
// ---------------------------------------------------------------------------
StructuredBuffer<uint> _NodeCollisionRefStart;
StructuredBuffer<uint> _NodeCollisionRefs;

// ---------------------------------------------------------------------------
// Per-contact stencil data for the CURRENT solve layer.
// Layer 0  : built from raw contacts.
// Layer >0 : built from propagated coarse contacts.
// ---------------------------------------------------------------------------
StructuredBuffer<uint>  _ColAnchorGi;

StructuredBuffer<uint>  _ColNodeGi0;
StructuredBuffer<uint>  _ColNodeGi1;
StructuredBuffer<uint>  _ColNodeGi2;
StructuredBuffer<uint>  _ColNodeGi3;

StructuredBuffer<float> _ColBeta0;
StructuredBuffer<float> _ColBeta1;
StructuredBuffer<float> _ColBeta2;
StructuredBuffer<float> _ColBeta3;

StructuredBuffer<float> _ColNX;
StructuredBuffer<float> _ColNY;

StructuredBuffer<float> _ColPen;      // raw penetration cap for this stencil
StructuredBuffer<float> _ColScale;    // fine: 1.0, coarse: sqrt(weight)

StructuredBuffer<uint>  _ColOwnerA;
StructuredBuffer<uint>  _ColOwnerB;

StructuredBuffer<CollisionFineContact> _FineContacts;
StructuredBuffer<uint> _FineContactCountBuffer;

StructuredBuffer<CollisionCoarseContact> _CoarseContacts;
StructuredBuffer<uint> _CoarseContactCountBuffer;

StructuredBuffer<uint> _BoundaryEdgeV0Gi;
StructuredBuffer<uint> _BoundaryEdgeV1Gi;
StructuredBuffer<uint> _BoundaryVertexGi;

uint _CoarseContactCapacity;

// ---------------------------------------------------------------------------
// Generic readers
// ---------------------------------------------------------------------------
bool ColReadNode(uint cid, uint slot, out uint nodeGi, out float beta)
{
    nodeGi = COLLISION_INVALID_U32;
    beta   = 0.0;

    switch (slot)
    {
        case 0u: nodeGi = _ColNodeGi0[cid]; beta = _ColBeta0[cid]; return true;
        case 1u: nodeGi = _ColNodeGi1[cid]; beta = _ColBeta1[cid]; return true;
        case 2u: nodeGi = _ColNodeGi2[cid]; beta = _ColBeta2[cid]; return true;
        case 3u: nodeGi = _ColNodeGi3[cid]; beta = _ColBeta3[cid]; return true;
        default: return false;
    }
}

float2 ColReadNormal(uint cid)
{
    return float2(_ColNX[cid], _ColNY[cid]);
}

bool ColReadNormalizedNormal(uint cid, out float2 nrm)
{
    nrm = ColReadNormal(cid);
    float n2 = dot(nrm, nrm);
    if (!(n2 > 1e-12))
    {
        nrm = 0.0;
        return false;
    }

    nrm *= rsqrt(n2);
    return true;
}

float ColReadScale(uint cid)
{
    return max(_ColScale[cid], 0.0);
}

float ColReadPen(uint cid)
{
    return max(_ColPen[cid], 0.0);
}

uint ColRefBegin(uint li)
{
    return _NodeCollisionRefStart[li];
}

uint ColRefEnd(uint li)
{
    return _NodeCollisionRefStart[li + 1u];
}

uint ColFineCount()
{
    uint count = _FineContactCountBuffer[0u];
    return min(count, _CollisionEventCapacity);
}

uint ColCoarseCount()
{
    uint count = _CoarseContactCountBuffer[0u];
    return min(count, _CoarseContactCapacity);
}

#endif