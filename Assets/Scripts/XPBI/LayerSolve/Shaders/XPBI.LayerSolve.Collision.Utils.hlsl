#ifndef XPBI_COLLISION_NODE_OWNED_HELPERS_INCLUDED
#define XPBI_COLLISION_NODE_OWNED_HELPERS_INCLUDED

static const uint COLLISION_INVALID_U32 = 0xffffffffu;

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

StructuredBuffer<XPBI_NodeContact> _ColNodeContacts;
StructuredBuffer<uint>             _ColNodeContactCount;
RWStructuredBuffer<float>          _ColNodeContactLambda;

StructuredBuffer<float2>           _ColReadPos;
StructuredBuffer<float2>           _ColReadVel;

uint _ColNodeContactStride;

float XPBI_ColSafeRsqrt(float x)
{
    return rsqrt(max(x, 1e-20));
}

float2 XPBI_ColSafeNormalize(float2 v)
{
    return v * XPBI_ColSafeRsqrt(dot(v, v));
}

uint XPBI_ColNodeContactIndex(uint li, uint slot)
{
    return li * _ColNodeContactStride + slot;
}

uint XPBI_ColNodeContactCountOf(uint li)
{
    return min(_ColNodeContactCount[li], _ColNodeContactStride);
}

bool XPBI_ColReadContact(uint li, uint slot, out XPBI_NodeContact c)
{
    c = (XPBI_NodeContact)0;

    if (slot >= _ColNodeContactStride)
        return false;

    uint idx = XPBI_ColNodeContactIndex(li, slot);
    c = _ColNodeContacts[idx];

    if (c.ownerSlave == COLLISION_INVALID_U32) return false;
    if (c.ownerMaster == COLLISION_INVALID_U32) return false;
    if (c.slaveGi == COLLISION_INVALID_U32) return false;
    if (!(c.pen > 0.0)) return false;
    if (!(c.scale > 1e-9)) return false;

    return true;
}

float XPBI_ColReadLambda(uint li, uint slot)
{
    return _ColNodeContactLambda[XPBI_ColNodeContactIndex(li, slot)];
}

void XPBI_ColWriteLambda(uint li, uint slot, float value)
{
    _ColNodeContactLambda[XPBI_ColNodeContactIndex(li, slot)] = value;
}

float2 XPBI_ColReadFrozenPos(uint gi)
{
    return _ColReadPos[gi];
}

float2 XPBI_ColReadFrozenVel(uint gi)
{
    return _ColReadVel[gi];
}

float2 XPBI_ColReadFrozenPredPos(uint gi, float dt)
{
    return XPBI_ColReadFrozenPos(gi) + XPBI_ColReadFrozenVel(gi) * dt;
}

bool XPBI_ColReadNormal(XPBI_NodeContact c, out float2 nrm)
{
    float n2 = dot(c.n, c.n);
    if (!(n2 > 1e-20))
    {
        nrm = 0.0;
        return false;
    }

    nrm = c.n * rsqrt(n2);
    return true;
}

float2 XPBI_ColSampleMasterPredPos(XPBI_NodeContact c, float dt)
{
    if (c.masterGi0 == COLLISION_INVALID_U32)
        return c.x;

    if (c.masterGi1 == COLLISION_INVALID_U32)
        return XPBI_ColReadFrozenPredPos(c.masterGi0, dt);

    float w1 = saturate(c.masterW1);
    float w0 = 1.0 - w1;

    float2 x0 = XPBI_ColReadFrozenPredPos(c.masterGi0, dt);
    float2 x1 = XPBI_ColReadFrozenPredPos(c.masterGi1, dt);
    return w0 * x0 + w1 * x1;
}

float2 XPBI_ColSampleMasterVel(XPBI_NodeContact c)
{
    if (c.masterGi0 == COLLISION_INVALID_U32)
        return 0.0;

    if (c.masterGi1 == COLLISION_INVALID_U32)
        return XPBI_ColReadFrozenVel(c.masterGi0);

    float w1 = saturate(c.masterW1);
    float w0 = 1.0 - w1;

    float2 v0 = XPBI_ColReadFrozenVel(c.masterGi0);
    float2 v1 = XPBI_ColReadFrozenVel(c.masterGi1);
    return w0 * v0 + w1 * v1;
}

float2 XPBI_ColClampDeltaV(float2 dv, float maxDv, float eps)
{
    if (!(maxDv > eps))
        return dv;

    float dv2 = dot(dv, dv);
    float maxDv2 = maxDv * maxDv;

    if (dv2 > maxDv2)
        dv *= maxDv * rsqrt(max(dv2, eps * eps));

    return dv;
}

#endif