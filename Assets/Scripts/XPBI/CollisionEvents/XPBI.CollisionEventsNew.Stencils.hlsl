// ============================================================================
// CollisionStencilBuild.hlsl
// Transform raw fine/coarse contacts into per-contact solve stencils + anchor CSR.
// ============================================================================

#ifndef COLLISION_STENCIL_BUILD_INCLUDED
#define COLLISION_STENCIL_BUILD_INCLUDED

static const uint COLLISION_INVALID_U32 = 0xffffffffu;
static const uint COLLISION_MAX_NODES   = 4u;
static const float COL_STENCIL_GEOM_EPS = 1e-12;

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

StructuredBuffer<uint> _CoarseContactCountBuffer;

uint LocalIndexFromGlobal(uint gi)
{
    if (_UseDtGlobalNodeMap != 0u)
    {
        int li = _DtGlobalToLayerLocalMap[gi];
        return (li >= 0) ? (uint)li : COLLISION_INVALID_U32;
    }

    int liUnmapped = (int)gi - (int)_DtLocalBase;
    return (liUnmapped >= 0) ? (uint)liUnmapped : COLLISION_INVALID_U32;
}

uint ColExpandedFineBase()
{
    return 0u;
}

uint ColExpandedCoarseBase()
{
    return _CollisionEventCapacity;
}

uint ColExpandedFineCid(uint fineLocalCid)
{
    return ColExpandedFineBase() + fineLocalCid;
}

uint ColExpandedCoarseCid(uint coarseLocalCid)
{
    return ColExpandedCoarseBase() + coarseLocalCid;
}

void ColWriteSlot(uint cid, uint slot, uint gi, float beta)
{
    switch (slot)
    {
        case 0u: _ColNodeGi0[cid] = gi; _ColBeta0[cid] = beta; break;
        case 1u: _ColNodeGi1[cid] = gi; _ColBeta1[cid] = beta; break;
        case 2u: _ColNodeGi2[cid] = gi; _ColBeta2[cid] = beta; break;
        case 3u: _ColNodeGi3[cid] = gi; _ColBeta3[cid] = beta; break;
        default: break;
    }
}

void ColInvalidate(uint cid)
{
    _ColAnchorGi[cid] = COLLISION_INVALID_U32;

    _ColOwnerA[cid] = COLLISION_INVALID_U32;
    _ColOwnerB[cid] = COLLISION_INVALID_U32;

    _ColNX[cid] = 0.0;
    _ColNY[cid] = 0.0;

    _ColPen[cid] = 0.0;
    _ColScale[cid] = 0.0;

    [unroll]
    for (uint i = 0u; i < COLLISION_MAX_NODES; i++)
        ColWriteSlot(cid, i, COLLISION_INVALID_U32, 0.0);
}

void FinalizeAndStoreStencil(
    uint cid,
    uint ownerA,
    uint ownerB,
    float2 nIn,
    float penRaw,
    float scale,
    uint rawGi0,
    uint rawGi1,
    uint rawGi2,
    uint rawGi3,
    float rawBeta0,
    float rawBeta1,
    float rawBeta2,
    float rawBeta3)
{
    ColInvalidate(cid);

    if (ownerA == ownerB)
        return;
    if (ownerA == COLLISION_INVALID_U32 || ownerB == COLLISION_INVALID_U32)
        return;

    float n2 = dot(nIn, nIn);
    if (!(n2 > COL_STENCIL_GEOM_EPS))
        return;

    if (!(scale > 0.0))
        return;

    float2 nrm = nIn * rsqrt(n2);

    uint rawGi[COLLISION_MAX_NODES];
    float rawBeta[COLLISION_MAX_NODES];

    rawGi[0] = rawGi0; rawGi[1] = rawGi1; rawGi[2] = rawGi2; rawGi[3] = rawGi3;
    rawBeta[0] = rawBeta0; rawBeta[1] = rawBeta1; rawBeta[2] = rawBeta2; rawBeta[3] = rawBeta3;

    uint mergedGi[COLLISION_MAX_NODES];
    float mergedBeta[COLLISION_MAX_NODES];

    [unroll]
    for (uint i = 0u; i < COLLISION_MAX_NODES; i++)
    {
        mergedGi[i] = COLLISION_INVALID_U32;
        mergedBeta[i] = 0.0;
    }

    [unroll]
    for (uint s = 0u; s < COLLISION_MAX_NODES; s++)
    {
        uint g = rawGi[s];
        float b = rawBeta[s];

        if (g == COLLISION_INVALID_U32)
            continue;
        if (abs(b) <= COL_STENCIL_GEOM_EPS)
            continue;

        [unroll]
        for (uint m = 0u; m < COLLISION_MAX_NODES; m++)
        {
            if (mergedGi[m] == g)
            {
                mergedBeta[m] += b;
                break;
            }

            if (mergedGi[m] == COLLISION_INVALID_U32)
            {
                mergedGi[m] = g;
                mergedBeta[m] = b;
                break;
            }
        }
    }

    uint outCount = 0u;
    uint anchorGi = COLLISION_INVALID_U32;

    uint outGi[COLLISION_MAX_NODES];
    float outBeta[COLLISION_MAX_NODES];

    [unroll]
    for (uint i = 0u; i < COLLISION_MAX_NODES; i++)
    {
        outGi[i] = COLLISION_INVALID_U32;
        outBeta[i] = 0.0;
    }

    [unroll]
    for (uint s = 0u; s < COLLISION_MAX_NODES; s++)
    {
        uint g = mergedGi[s];
        float b = mergedBeta[s];

        if (g == COLLISION_INVALID_U32)
            continue;
        if (abs(b) <= COL_STENCIL_GEOM_EPS)
            continue;

        uint li = LocalIndexFromGlobal(g);
        if (li == COLLISION_INVALID_U32 || li >= _ActiveCount)
            continue;

        outGi[outCount] = g;
        outBeta[outCount] = b;
        outCount++;

        if (anchorGi == COLLISION_INVALID_U32 || g < anchorGi)
            anchorGi = g;
    }

    if (outCount < 2u)
        return;
    if (anchorGi == COLLISION_INVALID_U32)
        return;

    _ColOwnerA[cid] = ownerA;
    _ColOwnerB[cid] = ownerB;
    _ColNX[cid] = nrm.x;
    _ColNY[cid] = nrm.y;
    _ColPen[cid] = max(penRaw, 0.0);
    _ColScale[cid] = scale;
    _ColAnchorGi[cid] = anchorGi;

    [unroll]
    for (uint i = 0u; i < COLLISION_MAX_NODES; i++)
        ColWriteSlot(cid, i, outGi[i], outBeta[i]);

    uint anchorLi = LocalIndexFromGlobal(anchorGi);
    if (anchorLi == COLLISION_INVALID_U32 || anchorLi >= _ActiveCount)
    {
        ColInvalidate(cid);
        return;
    }

    InterlockedAdd(_NodeCollisionRefCount[anchorLi], 1u);
}

[numthreads(64, 1, 1)]
void ClearNodeCollisionRefAux(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i < _ActiveCount)
    {
        _NodeCollisionRefCount[i] = 0u;
        _NodeCollisionRefWrite[i] = 0u;
    }

    if (i == 0u)
        _NodeCollisionRefStart[_ActiveCount] = 0u;
}

[numthreads(64, 1, 1)]
void BuildFineContactStencils(uint3 tid : SV_DispatchThreadID)
{
    uint fineCid = tid.x;
    uint count = min(_ContactCount[0u], _CollisionEventCapacity);
    if (fineCid >= count)
        return;

    uint cid = ColExpandedFineCid(fineCid);
    Contact c = _Contacts[fineCid];

    FinalizeAndStoreStencil(
        cid,
        c.ownerA,
        c.ownerB,
        c.n,
        c.pen,
        1.0,
        c.nodeGi0,
        c.nodeGi1,
        c.nodeGi2,
        c.nodeGi3,
        c.beta0,
        c.beta1,
        c.beta2,
        c.beta3);
}

[numthreads(64, 1, 1)]
void BuildCoarseContactStencils(uint3 tid : SV_DispatchThreadID)
{
    uint coarseCid = tid.x;
    uint count = min(_CoarseContactCountBuffer[0u], _CoarseContactCapacity);
    if (coarseCid >= count)
        return;

    uint cid = ColExpandedCoarseCid(coarseCid);
    CollisionCoarseContact c = _CoarseContacts[coarseCid];

    float w = max(c.weight, 1e-12);

    FinalizeAndStoreStencil(
        cid,
        c.ownerA,
        c.ownerB,
        c.n,
        max(c.pen, 0.0),
        sqrt(w),
        c.coarseGiA,
        c.coarseGiB,
        COLLISION_INVALID_U32,
        COLLISION_INVALID_U32,
        -1.0,
        1.0,
        0.0,
        0.0);
}

[numthreads(1, 1, 1)]
void ExclusiveScanNodeCollisionRefCount(uint3 tid : SV_DispatchThreadID)
{
    if (tid.x != 0u || tid.y != 0u || tid.z != 0u)
        return;

    uint running = 0u;
    [loop]
    for (uint i = 0u; i < _ActiveCount; i++)
    {
        _NodeCollisionRefStart[i] = running;
        running += _NodeCollisionRefCount[i];
    }

    _NodeCollisionRefStart[_ActiveCount] = running;
}

[numthreads(64, 1, 1)]
void ClearNodeCollisionRefWrite(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i < _ActiveCount)
        _NodeCollisionRefWrite[i] = 0u;
}

[numthreads(64, 1, 1)]
void ScatterFineContactRefs(uint3 tid : SV_DispatchThreadID)
{
    uint fineCid = tid.x;
    uint count = min(_ContactCount[0u], _CollisionEventCapacity);
    if (fineCid >= count)
        return;

    uint cid = ColExpandedFineCid(fineCid);

    uint anchorGi = _ColAnchorGi[cid];
    if (anchorGi == COLLISION_INVALID_U32)
        return;

    uint anchorLi = LocalIndexFromGlobal(anchorGi); 
    if (anchorLi == COLLISION_INVALID_U32 || anchorLi >= _ActiveCount)
        return;

    uint dst;
    InterlockedAdd(_NodeCollisionRefWrite[anchorLi], 1u, dst);
    _NodeCollisionRefs[_NodeCollisionRefStart[anchorLi] + dst] = cid;
}

[numthreads(64, 1, 1)]
void ScatterCoarseContactRefs(uint3 tid : SV_DispatchThreadID)
{
    uint coarseCid = tid.x;
    uint count = min(_CoarseContactCountBuffer[0u], _CoarseContactCapacity);
    if (coarseCid >= count)
        return;

    uint cid = ColExpandedCoarseCid(coarseCid);

    uint anchorGi = _ColAnchorGi[cid];
    if (anchorGi == COLLISION_INVALID_U32)
        return;

    uint anchorLi = LocalIndexFromGlobal(anchorGi);
    if (anchorLi == COLLISION_INVALID_U32 || anchorLi >= _ActiveCount)
        return;

    uint dst;
    InterlockedAdd(_NodeCollisionRefWrite[anchorLi], 1u, dst);
    _NodeCollisionRefs[_NodeCollisionRefStart[anchorLi] + dst] = cid;
}

#endif
