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
if (ownerICollision >= 0)
{
float dt = max(_Dt, EPS);
float dt2 = dt * dt;

float alphaCollision = max(_CollisionCompliance, 0.0);
float targetSeparation = max(EPS, _CollisionSupportScale * support);

bool useFineContacts = (_ActiveCount == _FineCount);

uint contactCount = useFineContacts ? ColFineCount() : ColCoarseCount();
uint lambdaBase   = useFineContacts ? 0u : _CollisionEventCapacity;

const float COL_GEOM_EPS = 1e-12;
const float COL_PARAM_EPS = 1e-4;

[loop]
for (uint cid = 0u; cid < contactCount; cid++)
{
    uint ownerA = COLLISION_INVALID_U32;
    uint ownerB = COLLISION_INVALID_U32;
    float2 nrm = 0.0;
    float penRaw = 0.0;
    float scale = 1.0;

    uint  nodeGiRaw[COLLISION_MAX_NODES];
    float betaRaw[COLLISION_MAX_NODES];

    [unroll]
    for (uint iInit = 0u; iInit < COLLISION_MAX_NODES; iInit++)
    {
        nodeGiRaw[iInit] = COLLISION_INVALID_U32;
        betaRaw[iInit] = 0.0;
    }

    // --------------------------------------------------------------------
    // Decode current event into a raw local stencil.
    // --------------------------------------------------------------------
    if (useFineContacts)
    {
        CollisionFineContact c = _FineContacts[cid];

        ownerA = c.ownerA;
        ownerB = c.ownerB;
        nrm = c.n;
        penRaw = max(c.pen, 0.0);
        scale = 1.0;

        // Fine vertex-vs-feature event.
        if (c.vGiA != COLLISION_INVALID_U32)
        {
            nodeGiRaw[0] = c.vGiA;
            betaRaw[0] = -1.0;

            uint b0 = _BoundaryEdgeV0Gi[c.featB];
            uint b1 = _BoundaryEdgeV1Gi[c.featB];
            uint bv = _BoundaryVertexGi[c.featB];

            bool haveB0 = (b0 != COLLISION_INVALID_U32);
            bool haveB1 = (b1 != COLLISION_INVALID_U32);

            // Prefer the boundary edge interpretation when both endpoints exist.
            // Collapse to point-point when projection lands on an endpoint.
            if (haveB0 && haveB1 && b0 != b1)
            {
                uint b0Li = LocalIndexFromGlobal(b0);
                uint b1Li = LocalIndexFromGlobal(b1);

                bool b0Active = (b0Li != COLLISION_INVALID_U32 && b0Li < active);
                bool b1Active = (b1Li != COLLISION_INVALID_U32 && b1Li < active);

                if (b0Active && b1Active)
                {
                    float2 b0Cand = XPBI_POS(b0Li, b0) + XPBI_VEL(b0Li, b0) * dt;
                    float2 b1Cand = XPBI_POS(b1Li, b1) + XPBI_VEL(b1Li, b1) * dt;
                    float2 e = b1Cand - b0Cand;
                    float e2 = dot(e, e);

                    if (e2 > COL_GEOM_EPS)
                    {
                        float t = saturate(dot(c.cpB - b0Cand, e) / e2);

                        if (t <= COL_PARAM_EPS)
                        {
                            nodeGiRaw[1] = b0;
                            betaRaw[1] = 1.0;
                        }
                        else if (t >= (1.0 - COL_PARAM_EPS))
                        {
                            nodeGiRaw[1] = b1;
                            betaRaw[1] = 1.0;
                        }
                        else
                        {
                            nodeGiRaw[1] = b0;
                            nodeGiRaw[2] = b1;
                            betaRaw[1] = 1.0 - t;
                            betaRaw[2] = t;
                        }
                    }
                    else
                    {
                        float d0 = dot(c.cpB - b0Cand, c.cpB - b0Cand);
                        float d1 = dot(c.cpB - b1Cand, c.cpB - b1Cand);

                        nodeGiRaw[1] = (d0 <= d1) ? b0 : b1;
                        betaRaw[1] = 1.0;
                    }
                }
                else if (b0Active || b1Active)
                {
                    nodeGiRaw[1] = b0Active ? b0 : b1;
                    betaRaw[1] = 1.0;
                }
                else if (bv != COLLISION_INVALID_U32)
                {
                    nodeGiRaw[1] = bv;
                    betaRaw[1] = 1.0;
                }
            }
            else if (bv != COLLISION_INVALID_U32)
            {
                nodeGiRaw[1] = bv;
                betaRaw[1] = 1.0;
            }
            else if (haveB0 || haveB1)
            {
                nodeGiRaw[1] = haveB0 ? b0 : b1;
                betaRaw[1] = 1.0;
            }
        }
        // Fine edge-edge event.
        else
        {
            uint a0 = _BoundaryEdgeV0Gi[c.heA];
            uint a1 = _BoundaryEdgeV1Gi[c.heA];
            uint b0 = _BoundaryEdgeV0Gi[c.featB];
            uint b1 = _BoundaryEdgeV1Gi[c.featB];

            if (a0 != COLLISION_INVALID_U32 && a1 != COLLISION_INVALID_U32 &&
                b0 != COLLISION_INVALID_U32 && b1 != COLLISION_INVALID_U32)
            {
                uint a0Li = LocalIndexFromGlobal(a0);
                uint a1Li = LocalIndexFromGlobal(a1);
                uint b0Li = LocalIndexFromGlobal(b0);
                uint b1Li = LocalIndexFromGlobal(b1);

                bool a0Active = (a0Li != COLLISION_INVALID_U32 && a0Li < active);
                bool a1Active = (a1Li != COLLISION_INVALID_U32 && a1Li < active);
                bool b0Active = (b0Li != COLLISION_INVALID_U32 && b0Li < active);
                bool b1Active = (b1Li != COLLISION_INVALID_U32 && b1Li < active);

                if (a0Active && a1Active && b0Active && b1Active)
                {
                    float2 a0Cand = XPBI_POS(a0Li, a0) + XPBI_VEL(a0Li, a0) * dt;
                    float2 a1Cand = XPBI_POS(a1Li, a1) + XPBI_VEL(a1Li, a1) * dt;
                    float2 b0Cand = XPBI_POS(b0Li, b0) + XPBI_VEL(b0Li, b0) * dt;
                    float2 b1Cand = XPBI_POS(b1Li, b1) + XPBI_VEL(b1Li, b1) * dt;

                    float2 aEdge = a1Cand - a0Cand;
                    float2 bEdge = b1Cand - b0Cand;
                    float aLen2 = dot(aEdge, aEdge);
                    float bLen2 = dot(bEdge, bEdge);

                    // Degenerate A edge -> point-edge against B.
                    if (aLen2 <= COL_GEOM_EPS && bLen2 > COL_GEOM_EPS)
                    {
                        float t = saturate(dot(c.x - b0Cand, bEdge) / bLen2);

                        nodeGiRaw[0] = a0;
                        betaRaw[0] = -1.0;

                        if (t <= COL_PARAM_EPS)
                        {
                            nodeGiRaw[1] = b0;
                            betaRaw[1] = 1.0;
                        }
                        else if (t >= (1.0 - COL_PARAM_EPS))
                        {
                            nodeGiRaw[1] = b1;
                            betaRaw[1] = 1.0;
                        }
                        else
                        {
                            nodeGiRaw[1] = b0;
                            nodeGiRaw[2] = b1;
                            betaRaw[1] = 1.0 - t;
                            betaRaw[2] = t;
                        }
                    }
                    // Degenerate B edge -> edge-point.
                    else if (bLen2 <= COL_GEOM_EPS && aLen2 > COL_GEOM_EPS)
                    {
                        float s = saturate(dot(c.x - a0Cand, aEdge) / aLen2);

                        if (s <= COL_PARAM_EPS)
                        {
                            nodeGiRaw[0] = a0;
                            betaRaw[0] = -1.0;
                        }
                        else if (s >= (1.0 - COL_PARAM_EPS))
                        {
                            nodeGiRaw[0] = a1;
                            betaRaw[0] = -1.0;
                        }
                        else
                        {
                            nodeGiRaw[0] = a0;
                            nodeGiRaw[1] = a1;
                            betaRaw[0] = -(1.0 - s);
                            betaRaw[1] = -s;
                        }

                        nodeGiRaw[2] = b0;
                        betaRaw[2] = 1.0;
                    }
                    // Both degenerate -> point-point.
                    else if (aLen2 <= COL_GEOM_EPS && bLen2 <= COL_GEOM_EPS)
                    {
                        nodeGiRaw[0] = a0;
                        nodeGiRaw[1] = b0;
                        betaRaw[0] = -1.0;
                        betaRaw[1] = 1.0;
                    }
                    // Full edge-edge stencil.
                    else
                    {
                        float s = saturate(dot(c.x - a0Cand, aEdge) / aLen2);
                        float t = saturate(dot(c.x - b0Cand, bEdge) / bLen2);

                        nodeGiRaw[0] = a0;
                        nodeGiRaw[1] = a1;
                        nodeGiRaw[2] = b0;
                        nodeGiRaw[3] = b1;

                        betaRaw[0] = -(1.0 - s);
                        betaRaw[1] = -s;
                        betaRaw[2] = 1.0 - t;
                        betaRaw[3] = t;
                    }
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

        float w = max(c.weight, EPS);
        penRaw = max(c.pen, 0.0) / w;
        scale  = sqrt(w);

        nodeGiRaw[0] = c.coarseGiA;
        nodeGiRaw[1] = c.coarseGiB;
        betaRaw[0] = -1.0;
        betaRaw[1] = 1.0;
    }

    if (ownerA == ownerB)
        continue;
    if (ownerA == COLLISION_INVALID_U32 || ownerB == COLLISION_INVALID_U32)
        continue;

    float n2 = dot(nrm, nrm);
    if (!(n2 > COL_GEOM_EPS))
        continue;
    nrm *= rsqrt(n2);

    if (!(scale > EPS))
        continue;

    // --------------------------------------------------------------------
    // Merge duplicate raw participants by GI.
    // This avoids overcounting when a feature degenerates or collapses.
    // --------------------------------------------------------------------
    uint  nodeGiMerged[COLLISION_MAX_NODES];
    float betaMerged[COLLISION_MAX_NODES];

    [unroll]
    for (uint mInit = 0u; mInit < COLLISION_MAX_NODES; mInit++)
    {
        nodeGiMerged[mInit] = COLLISION_INVALID_U32;
        betaMerged[mInit] = 0.0;
    }

    [unroll]
    for (uint sRaw = 0u; sRaw < COLLISION_MAX_NODES; sRaw++)
    {
        uint g = nodeGiRaw[sRaw];
        float b = betaRaw[sRaw];

        if (g == COLLISION_INVALID_U32)
            continue;
        if (abs(b) <= COL_GEOM_EPS)
            continue;

        bool merged = false;

        [unroll]
        for (uint m = 0u; m < COLLISION_MAX_NODES; m++)
        {
            if (nodeGiMerged[m] == g)
            {
                betaMerged[m] += b;
                merged = true;
                break;
            }

            if (nodeGiMerged[m] == COLLISION_INVALID_U32)
            {
                nodeGiMerged[m] = g;
                betaMerged[m] = b;
                merged = true;
                break;
            }
        }

        if (!merged)
        {
            // Should not happen with max-4 input, but keep silent and skip.
        }
    }

    // --------------------------------------------------------------------
    // Validate merged participants, build active stencil, and choose anchor
    // FROM VALIDATED PARTICIPANTS ONLY.
    // --------------------------------------------------------------------
    float projected = 0.0;
    float wTerm = 0.0;
    uint usedCount = 0u;
    uint anchorGi = COLLISION_INVALID_U32;

    uint  nodeLiCache[COLLISION_MAX_NODES];
    uint  nodeGiCache[COLLISION_MAX_NODES];
    float betaCache[COLLISION_MAX_NODES];
    float invMassCache[COLLISION_MAX_NODES];
    bool  fixedCache[COLLISION_MAX_NODES];

    [unroll]
    for (uint cInit = 0u; cInit < COLLISION_MAX_NODES; cInit++)
    {
        nodeLiCache[cInit] = COLLISION_INVALID_U32;
        nodeGiCache[cInit] = COLLISION_INVALID_U32;
        betaCache[cInit] = 0.0;
        invMassCache[cInit] = 0.0;
        fixedCache[cInit] = true;
    }

    [unroll]
    for (uint s = 0u; s < COLLISION_MAX_NODES; s++)
    {
        uint nodeGi = nodeGiMerged[s];
        float beta = betaMerged[s];

        if (nodeGi == COLLISION_INVALID_U32)
            continue;
        if (abs(beta) <= COL_GEOM_EPS)
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

        if (anchorGi == COLLISION_INVALID_U32 || nodeGi < anchorGi)
            anchorGi = nodeGi;
    }

    if (usedCount < 2u)
        continue;
    if (anchorGi == COLLISION_INVALID_U32 || anchorGi != gi)
        continue;

    float Cn = projected - scale * targetSeparation;

    if (penRaw > EPS)
    {
        float penScaled = scale * penRaw;
        Cn = max(Cn, -penScaled);
    }

    if (Cn >= 0.0)
        continue;

    if (wTerm <= EPS && alphaCollision <= EPS)
        continue;

    uint lambdaIdx = lambdaBase + cid;
    float lambdaPrev = XPBI_COL_READ_LAMBDA(lambdaIdx);
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

        float2 dV = -invMass * betaScaled * dt * appliedDLambda * nrm;
        if (!all(isfinite(dV)))
            continue;

        XPBI_COL_APPLY_DV(nodeLi, nodeGi, dV);
    }

    XPBI_COL_WRITE_LAMBDA(lambdaIdx, lambdaNew);
}
}