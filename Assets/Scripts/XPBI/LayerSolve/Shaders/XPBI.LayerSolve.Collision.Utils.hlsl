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

struct XPBICohesiveShellEval
{
    float active;    // 0 or 1
    float s;         // r / support
    float shell;     // normalized shell coordinate in [0, 1]
    float traction;  // bilinear traction shape in [0, 1]
};

float XPBI_DegradeDamage(float d, float residualStiffness)
{
    float a = 1.0 - saturate(d);
    return a * a + residualStiffness;
}

float XPBI_DamageFromKappaExp(float kappa, float onset, float softening)
{
    if (!(kappa > onset)) return 0.0;
    return saturate(1.0 - exp(-(kappa - onset) / max(softening, EPS)));
}

void XPBI_SymEigenValues2x2(float a, float b, float d, out float l0, out float l1)
{
    float tr = a + d;
    float disc = sqrt(max((a - d) * (a - d) + 4.0 * b * b, 0.0));
    l0 = 0.5 * (tr + disc);
    l1 = 0.5 * (tr - disc);
}

float XPBI_TensileHenckyEnergy2D(Mat2 F, float mu, float lambda, float stretchEps)
{
    Mat2 FT = TransposeMat2(F);
    Mat2 C = MulMat2(FT, F);

    float a = C.c0.x;
    float b = 0.5 * (C.c0.y + C.c1.x);
    float d = C.c1.y;

    float l0 = 0.0, l1 = 0.0;
    XPBI_SymEigenValues2x2(a, b, d, l0, l1);

    float s0 = sqrt(max(l0, stretchEps * stretchEps));
    float s1 = sqrt(max(l1, stretchEps * stretchEps));

    float e0 = log(max(s0, stretchEps));
    float e1 = log(max(s1, stretchEps));

    float ep0 = max(e0, 0.0);
    float ep1 = max(e1, 0.0);
    float trp = max(e0 + e1, 0.0);

    return mu * (ep0 * ep0 + ep1 * ep1) + 0.5 * lambda * trp * trp;
}

XPBICohesiveShellEval XPBI_EvalCohesiveOuterShell(float r, float support, float onsetRatio, float peakRatio)
{
    XPBICohesiveShellEval o;
    o.active = 0.0;
    o.s = 0.0;
    o.shell = 0.0;
    o.traction = 0.0;

    if (!(support > EPS)) return o;

    float s0 = saturate(onsetRatio);
    float sp = saturate(peakRatio);
    sp = max(sp, s0 + 1e-4);

    float s = r / support;
    o.s = s;

    if (s <= s0 || s >= 1.0) return o;

    o.active = 1.0;
    o.shell = saturate((s - s0) / max(1.0 - s0, EPS));

    if (s <= sp)
    {
        o.traction = saturate((s - s0) / max(sp - s0, EPS));
    }
    else
    {
        o.traction = saturate((1.0 - s) / max(1.0 - sp, EPS));
    }

    return o;
}

float XPBI_ClampSymmetricPairScale(float pairScale)
{
    return saturate(pairScale);
}

#endif