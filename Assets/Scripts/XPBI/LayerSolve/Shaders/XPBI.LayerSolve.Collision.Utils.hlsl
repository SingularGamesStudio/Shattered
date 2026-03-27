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
    uint nodeGi0;
    uint nodeGi1;
    uint nodeGi2;
    uint nodeGi3;
    float beta0;
    float beta1;
    float beta2;
    float beta3;
    float2 n;
    float pen;
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

uint ColExpandedFineBase()
{
    return 0u;
}

uint ColExpandedCoarseBase()
{
    return _CollisionEventCapacity;
}

uint ColExpandedCapacity()
{
    return _CollisionEventCapacity + _CoarseContactCapacity;
}

bool ColExpandedIsFine(uint cid)
{
    return cid < ColExpandedCoarseBase();
}

uint ColExpandedToLocal(uint cid)
{
    return ColExpandedIsFine(cid) ? cid : (cid - ColExpandedCoarseBase());
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