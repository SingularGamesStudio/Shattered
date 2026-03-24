// ============================================================================
// CollisionSolveBlock.hlsl
//
// Expected locals already in scope:
//   uint  li, gi, active, dtLi
//   float support
//
// Expected macros/helpers in the solve path:
//   LocalIndexFromGlobal()
//   XPBI_POS(li, gi)
//   XPBI_VEL(li, gi)
//   XPBI_SET_VEL(li, gi, v)
//   XPBI_INV_MASS(li, gi)
//   XPBI_NEIGHBOR_FIXED(li, gi)
//   EPS
//   _Dt
//   _CollisionCompliance
//   _CollisionSupportScale
// ============================================================================

int ownerICollision = (li < active) ? _DtCollisionOwnerByLocal[li] : -1;
if (ownerICollision < 0)
    return;

float dt = max(_Dt, EPS);
float dt2 = dt * dt;

float alphaCollision = max(_CollisionCompliance, 0.0);
float targetSeparation = max(EPS, _CollisionSupportScale * support);

bool useFineContacts = (_ActiveCount == _FineCount);
uint contactCount = useFineContacts ? ColFineCount() : ColCoarseCount();
uint lambdaBase = useFineContacts ? 0u : _CollisionEventCapacity;

[loop]
for (uint cid = 0u; cid < contactCount; cid++)
{
    uint ownerA = COLLISION_INVALID_U32;
    uint ownerB = COLLISION_INVALID_U32;
    float2 nrm = 0.0;
    float penRaw = 0.0;
    float scale = 1.0;

    uint nodeGiIn[COLLISION_MAX_NODES];
    float betaIn[COLLISION_MAX_NODES];

    [unroll]
    for (uint sInit = 0u; sInit < COLLISION_MAX_NODES; sInit++)
    {
        nodeGiIn[sInit] = COLLISION_INVALID_U32;
        betaIn[sInit] = 0.0;
    }

    if (useFineContacts)
    {
        CollisionFineContact c = _FineContacts[cid];
        ownerA = c.ownerA;
        ownerB = c.ownerB;
        nrm = c.n;
        penRaw = max(c.pen, 0.0);
        scale = 1.0;

        if (c.vGiA != COLLISION_INVALID_U32)
        {
            nodeGiIn[0] = c.vGiA;
            betaIn[0] = -1.0;

            uint b0 = _BoundaryEdgeV0Gi[c.featB];
            uint b1 = _BoundaryEdgeV1Gi[c.featB];
            if (b0 == COLLISION_INVALID_U32)
                b0 = _BoundaryVertexGi[c.featB];

            if (b0 != COLLISION_INVALID_U32 && b1 != COLLISION_INVALID_U32)
            {
                uint b0Li = LocalIndexFromGlobal(b0);
                uint b1Li = LocalIndexFromGlobal(b1);
                if (b0Li != COLLISION_INVALID_U32 && b0Li < active && b1Li != COLLISION_INVALID_U32 && b1Li < active)
                {
                    float2 b0Cand = XPBI_POS(b0Li, b0) + XPBI_VEL(b0Li, b0) * dt;
                    float2 b1Cand = XPBI_POS(b1Li, b1) + XPBI_VEL(b1Li, b1) * dt;
                    float2 edge = b1Cand - b0Cand;
                    float edgeLen2 = dot(edge, edge);
                    float t = 0.0;
                    if (edgeLen2 > EPS)
                        t = saturate(dot(c.cpB - b0Cand, edge) / edgeLen2);

                    nodeGiIn[1] = b0;
                    nodeGiIn[2] = b1;
                    betaIn[1] = (1.0 - t);
                    betaIn[2] = t;
                }
                else
                {
                    nodeGiIn[1] = b0;
                    betaIn[1] = 1.0;
                }
            }
            else if (b0 != COLLISION_INVALID_U32)
            {
                nodeGiIn[1] = b0;
                betaIn[1] = 1.0;
            }
        }
        else
        {
            uint a0 = _BoundaryEdgeV0Gi[c.heA];
            uint a1 = _BoundaryEdgeV1Gi[c.heA];
            uint b0 = _BoundaryEdgeV0Gi[c.featB];
            uint b1 = _BoundaryEdgeV1Gi[c.featB];

            if (a0 != COLLISION_INVALID_U32 && a1 != COLLISION_INVALID_U32 && b0 != COLLISION_INVALID_U32 && b1 != COLLISION_INVALID_U32)
            {
                uint a0Li = LocalIndexFromGlobal(a0);
                uint a1Li = LocalIndexFromGlobal(a1);
                uint b0Li = LocalIndexFromGlobal(b0);
                uint b1Li = LocalIndexFromGlobal(b1);
                if (a0Li != COLLISION_INVALID_U32 && a0Li < active &&
                    a1Li != COLLISION_INVALID_U32 && a1Li < active &&
                    b0Li != COLLISION_INVALID_U32 && b0Li < active &&
                    b1Li != COLLISION_INVALID_U32 && b1Li < active)
                {
                    float2 a0Cand = XPBI_POS(a0Li, a0) + XPBI_VEL(a0Li, a0) * dt;
                    float2 a1Cand = XPBI_POS(a1Li, a1) + XPBI_VEL(a1Li, a1) * dt;
                    float2 b0Cand = XPBI_POS(b0Li, b0) + XPBI_VEL(b0Li, b0) * dt;
                    float2 b1Cand = XPBI_POS(b1Li, b1) + XPBI_VEL(b1Li, b1) * dt;

                    float2 aEdge = a1Cand - a0Cand;
                    float2 bEdge = b1Cand - b0Cand;
                    float aLen2 = dot(aEdge, aEdge);
                    float bLen2 = dot(bEdge, bEdge);

                    float s = 0.5;
                    float t = 0.5;
                    if (aLen2 > EPS)
                        s = saturate(dot(c.x - a0Cand, aEdge) / aLen2);
                    if (bLen2 > EPS)
                        t = saturate(dot(c.x - b0Cand, bEdge) / bLen2);

                    nodeGiIn[0] = a0;
                    nodeGiIn[1] = a1;
                    nodeGiIn[2] = b0;
                    nodeGiIn[3] = b1;
                    betaIn[0] = -(1.0 - s);
                    betaIn[1] = -s;
                    betaIn[2] = (1.0 - t);
                    betaIn[3] = t;
                }
            }
        }
    }
    else
    {
        CollisionCoarseContact c = _CoarseContacts[cid];
        ownerA = c.ownerA;
        ownerB = c.ownerB;
        nrm = c.n;
        penRaw = max(c.pen, 0.0);
        scale = sqrt(max(c.weight, 0.0));

        nodeGiIn[0] = c.coarseGiA;
        nodeGiIn[1] = c.coarseGiB;
        betaIn[0] = -1.0;
        betaIn[1] = 1.0;
    }

    if (ownerA == ownerB)
        continue;
    if (ownerA == COLLISION_INVALID_U32 || ownerB == COLLISION_INVALID_U32)
        continue;

    float n2 = dot(nrm, nrm);
    if (!(n2 > 1e-12))
        continue;
    nrm *= rsqrt(n2);

    if (!(scale > EPS))
        continue;

    uint anchorGi = COLLISION_INVALID_U32;
    [unroll]
    for (uint sAnchor = 0u; sAnchor < COLLISION_MAX_NODES; sAnchor++)
    {
        uint g = nodeGiIn[sAnchor];
        if (g == COLLISION_INVALID_U32)
            continue;
        if (anchorGi == COLLISION_INVALID_U32 || g < anchorGi)
            anchorGi = g;
    }
    if (anchorGi == COLLISION_INVALID_U32 || anchorGi != gi)
        continue;

    float projected = 0.0;
    float wTerm = 0.0;
    uint usedCount = 0u;

    uint  nodeLiCache[COLLISION_MAX_NODES];
    uint  nodeGiCache[COLLISION_MAX_NODES];
    float betaCache[COLLISION_MAX_NODES];
    float invMassCache[COLLISION_MAX_NODES];
    bool  fixedCache[COLLISION_MAX_NODES];

    [unroll]
    for (uint sInit2 = 0u; sInit2 < COLLISION_MAX_NODES; sInit2++)
    {
        nodeLiCache[sInit2] = COLLISION_INVALID_U32;
        nodeGiCache[sInit2] = COLLISION_INVALID_U32;
        betaCache[sInit2] = 0.0;
        invMassCache[sInit2] = 0.0;
        fixedCache[sInit2] = true;
    }

    [unroll]
    for (uint s = 0u; s < COLLISION_MAX_NODES; s++)
    {
        uint nodeGi = nodeGiIn[s];
        float beta = betaIn[s];

        if (nodeGi == COLLISION_INVALID_U32)
            continue;
        if (abs(beta) <= 1e-12)
            continue;

        uint nodeLi = LocalIndexFromGlobal(nodeGi);
        if (nodeLi == COLLISION_INVALID_U32 || nodeLi >= active)
            continue;

        bool fixedNode = XPBI_NEIGHBOR_FIXED(nodeLi, nodeGi);
        float invMass = fixedNode ? 0.0 : XPBI_INV_MASS(nodeLi, nodeGi);

        float betaScaled = scale * beta;
        float2 xCand = XPBI_POS(nodeLi, nodeGi) + XPBI_VEL(nodeLi, nodeGi) * dt;

        projected += betaScaled * dot(nrm, xCand);
        wTerm += dt2 * invMass * betaScaled * betaScaled;

        nodeLiCache[s] = nodeLi;
        nodeGiCache[s] = nodeGi;
        betaCache[s] = betaScaled;
        invMassCache[s] = invMass;
        fixedCache[s] = fixedNode;
        usedCount++;
    }

    if (usedCount < 2u)
        continue;

    float Cn = projected - scale * targetSeparation;

    if (penRaw > EPS)
    {
        float penScaled = scale * penRaw;
        Cn = min(Cn, -penScaled);
    }

    if (Cn >= 0.0)
        continue;

    if (wTerm <= EPS && alphaCollision <= EPS)
        continue;

    uint lambdaIdx = lambdaBase + cid;
    float lambdaPrev = _CollisionLambda[lambdaIdx];
    float dLambda = -(Cn + alphaCollision * lambdaPrev) / max(wTerm + alphaCollision, EPS);
    float lambdaNew = max(0.0, lambdaPrev + dLambda);
    float appliedDLambda = lambdaNew - lambdaPrev;
    if (appliedDLambda <= 0.0)
        continue;

    [unroll]
    for (uint sApply = 0u; sApply < COLLISION_MAX_NODES; sApply++)
    {
        uint nodeGi = nodeGiCache[sApply];
        if (nodeGi == COLLISION_INVALID_U32)
            continue;
        if (fixedCache[sApply])
            continue;

        uint nodeLi = nodeLiCache[sApply];
        float invMass = invMassCache[sApply];
        float betaScaled = betaCache[sApply];

        float2 dV = invMass * betaScaled * dt * appliedDLambda * nrm;
        if (!all(isfinite(dV)))
            continue;

        XPBI_SET_VEL(nodeLi, nodeGi, XPBI_VEL(nodeLi, nodeGi) + dV);
    }

    _CollisionLambda[lambdaIdx] = lambdaNew;
} 