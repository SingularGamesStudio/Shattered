// ============================================================================
// CollisionSolveBlock.hlsl
//
// Fast referenced-contact solve path.
//
// Keeps the newer collision improvements, but only processes contacts
// referenced by the current node instead of scanning all events.
//
// Expected locals already in scope:
//   uint  li, gi, active, dtLi
//   float support
//
// Expected macros/helpers in the solve path:
//   LocalIndexFromGlobal()
//   ColRefBegin(li)
//   ColRefEnd(li)
//   ColReadNormalizedNormal(cid, out nrm)
//   ColReadScale(cid)
//   ColReadPen(cid)
//   ColReadNode(cid, s, out nodeGi, out beta)
//   XPBI_POS(li, gi)
//   XPBI_VEL(li, gi)
//   XPBI_INV_MASS(li, gi)
//   XPBI_NEIGHBOR_FIXED(li, gi)
//   XPBI_COL_APPLY_DV(li, gi, dv)
//   XPBI_COL_READ_LAMBDA(idx)
//   XPBI_COL_WRITE_LAMBDA(idx, value)
//   EPS
//   _Dt
//   _CollisionCompliance
//   _CollisionSkinScale
//   _CollisionSlop
//   _CollisionPenBias
//   _CollisionMaxBias
//   _CollisionMaxPush
//   _CollisionRelaxation
//   _CollisionRestitution
//   _CollisionRestitutionThreshold
//   _CollisionFriction
//   _CollisionMaxDv
//
// Optional existing buffers:
//   _DtCollisionOwnerByLocal
//   _NodeCollisionRefs
//   _ColOwnerA
//   _ColOwnerB
// ============================================================================

if (li < active)
{
    float dt  = max(_Dt, EPS);
    float dt2 = dt * dt;

    float alphaCollision  = max(_CollisionCompliance, 0.0);
    float targetSeparation = max(EPS, _CollisionSkinScale * support);

    const float COL_GEOM_EPS = 1e-12;

    uint refBegin = ColRefBegin(li);
    uint refEnd   = ColRefEnd(li);

    [loop]
    for (uint refIt = refBegin; refIt < refEnd; refIt++)
    {
        uint cid = _NodeCollisionRefs[refIt];

        if (cid >= ColExpandedCapacity())
            continue;

        uint ownerA = _ColOwnerA[cid];
        uint ownerB = _ColOwnerB[cid];

        if (ownerA == ownerB)
            continue;
        if (ownerA == COLLISION_INVALID_U32 || ownerB == COLLISION_INVALID_U32)
            continue;

        float2 nrm;
        if (!ColReadNormalizedNormal(cid, nrm))
            continue;

        float scale = ColReadScale(cid);
        if (!(scale > EPS))
            continue;

        float penRaw = max(ColReadPen(cid), 0.0);

        // --------------------------------------------------------------------
        // Read raw stencil from the referenced contact.
        // Assumes ColReadNode() exposes the per-contact participants as
        // unscaled beta weights; scale is applied in solve.
        // --------------------------------------------------------------------
        uint  nodeGiRaw[COLLISION_MAX_NODES];
        float betaRaw[COLLISION_MAX_NODES];

        [unroll]
        for (uint iInit = 0u; iInit < COLLISION_MAX_NODES; iInit++)
        {
            nodeGiRaw[iInit] = COLLISION_INVALID_U32;
            betaRaw[iInit]   = 0.0;
        }

        [unroll]
        for (uint sRead = 0u; sRead < COLLISION_MAX_NODES; sRead++)
        {
            ColReadNode(cid, sRead, nodeGiRaw[sRead], betaRaw[sRead]);
        }

        // --------------------------------------------------------------------
        // Merge duplicate raw participants by GI.
        // Preserves the newer safeguard against collapsed/degenerate features.
        // --------------------------------------------------------------------
        uint  nodeGiMerged[COLLISION_MAX_NODES];
        float betaMerged[COLLISION_MAX_NODES];

        [unroll]
        for (uint mInit = 0u; mInit < COLLISION_MAX_NODES; mInit++)
        {
            nodeGiMerged[mInit] = COLLISION_INVALID_U32;
            betaMerged[mInit]   = 0.0;
        }

        [unroll]
        for (uint sRaw = 0u; sRaw < COLLISION_MAX_NODES; sRaw++)
        {
            uint  g = nodeGiRaw[sRaw];
            float b = betaRaw[sRaw];

            if (g == COLLISION_INVALID_U32)
                continue;
            if (abs(b) <= COL_GEOM_EPS)
                continue;

            [unroll]
            for (uint m = 0u; m < COLLISION_MAX_NODES; m++)
            {
                if (nodeGiMerged[m] == g)
                {
                    betaMerged[m] += b;
                    break;
                }

                if (nodeGiMerged[m] == COLLISION_INVALID_U32)
                {
                    nodeGiMerged[m] = g;
                    betaMerged[m]   = b;
                    break;
                }
            }
        }

        // --------------------------------------------------------------------
        // Validate merged participants, build active stencil, and choose anchor
        // from validated participants only.
        // --------------------------------------------------------------------
        float projected = 0.0;
        float wTerm     = 0.0;
        uint  usedCount = 0u;
        uint  anchorGi  = COLLISION_INVALID_U32;

        uint  nodeLiCache[COLLISION_MAX_NODES];
        uint  nodeGiCache[COLLISION_MAX_NODES];
        float betaScaledCache[COLLISION_MAX_NODES];
        float betaUnscaledCache[COLLISION_MAX_NODES];
        float invMassCache[COLLISION_MAX_NODES];
        bool  fixedCache[COLLISION_MAX_NODES];

        [unroll]
        for (uint cInit = 0u; cInit < COLLISION_MAX_NODES; cInit++)
        {
            nodeLiCache[cInit]         = COLLISION_INVALID_U32;
            nodeGiCache[cInit]         = COLLISION_INVALID_U32;
            betaScaledCache[cInit]     = 0.0;
            betaUnscaledCache[cInit]   = 0.0;
            invMassCache[cInit]        = 0.0;
            fixedCache[cInit]          = true;
        }

        [unroll]
        for (uint s = 0u; s < COLLISION_MAX_NODES; s++)
        {
            uint  nodeGi = nodeGiMerged[s];
            float beta   = betaMerged[s];

            if (nodeGi == COLLISION_INVALID_U32)
                continue;
            if (abs(beta) <= COL_GEOM_EPS)
                continue;

            uint nodeLi = LocalIndexFromGlobal(nodeGi);
            if (nodeLi == COLLISION_INVALID_U32 || nodeLi >= active)
                continue;

            bool  fixedNode = XPBI_NEIGHBOR_FIXED(nodeLi, nodeGi);
            float invMass   = fixedNode ? 0.0 : XPBI_INV_MASS(nodeLi, nodeGi);

            float  betaScaled = scale * beta;
            float2 xCand      = XPBI_POS(nodeLi, nodeGi) + XPBI_VEL(nodeLi, nodeGi) * dt;

            projected += betaScaled * dot(nrm, xCand);
            wTerm     += dt2 * invMass * betaScaled * betaScaled;

            nodeLiCache[s]       = nodeLi;
            nodeGiCache[s]       = nodeGi;
            betaScaledCache[s]   = betaScaled;
            betaUnscaledCache[s] = beta;
            invMassCache[s]      = invMass;
            fixedCache[s]        = fixedNode;

            usedCount++;

            if (anchorGi == COLLISION_INVALID_U32 || nodeGi < anchorGi)
                anchorGi = nodeGi;
        }

        if (usedCount < 2u)
            continue;
        if (anchorGi == COLLISION_INVALID_U32 || anchorGi != gi)
            continue;

        float slop = max(_CollisionSlop, 0.0) * support;
        if (penRaw <= slop)
            continue;

        float penEff  = max(penRaw - slop, 0.0);
        float bias    = min(max(_CollisionPenBias, 0.0) * penEff, max(_CollisionMaxBias, 0.0));
        float maxPush = max(_CollisionMaxPush, 0.0);

        float Cn = projected - scale * targetSeparation;
        Cn -= scale * bias;
        Cn = max(Cn, -scale * (penEff + maxPush));

        if (Cn >= 0.0)
            continue;

        if (wTerm <= EPS && alphaCollision <= EPS)
            continue;

        // Lambda uses the same expanded CID space as the stencils/refs.
        uint  lambdaIdx  = cid;
        float lambdaPrev = XPBI_COL_READ_LAMBDA(lambdaIdx);

        float dLambda = -(Cn + alphaCollision * lambdaPrev) / max(wTerm + alphaCollision, EPS);
        float lambdaNewRaw = max(0.0, lambdaPrev + dLambda);

        float appliedDLambda = lambdaNewRaw - lambdaPrev;
        float relax = saturate(_CollisionRelaxation);
        appliedDLambda *= relax;

        if (appliedDLambda <= 0.0)
            continue;

        float lambdaNew = lambdaPrev + appliedDLambda;

        // --------------------------------------------------------------------
        // Relative velocity terms for restitution / friction.
        // Matches the newer path: use unscaled stencil weights here.
        // --------------------------------------------------------------------
        float2 vRel = 0.0;
        float  wEff = 0.0;

        [unroll]
        for (uint sRel = 0u; sRel < COLLISION_MAX_NODES; sRel++)
        {
            uint relGi = nodeGiCache[sRel];
            if (relGi == COLLISION_INVALID_U32)
                continue;

            uint  relLi         = nodeLiCache[sRel];
            float betaUnscaled  = betaUnscaledCache[sRel];

            vRel += betaUnscaled * XPBI_VEL(relLi, relGi);
            wEff += invMassCache[sRel] * betaUnscaled * betaUnscaled;
        }

        float  vn    = dot(vRel, nrm);
        float2 vt    = vRel - vn * nrm;
        float  vtLen = length(vt);

        float restitution          = saturate(_CollisionRestitution);
        float restitutionThreshold = max(_CollisionRestitutionThreshold, 0.0);
        float restitutionDeltaVn   = restitution * max(-vn - restitutionThreshold, 0.0);

        float friction = saturate(_CollisionFriction);
        float maxDv    = max(_CollisionMaxDv, 0.0);

        float  wEffSafe           = max(wEff, EPS);
        float  restitutionImpulse = (wEff > EPS) ? (restitutionDeltaVn / wEffSafe) : 0.0;
        float  maxFrictionDeltaV  = friction * max(-vn, 0.0);
        float  frictionImpulse    = (wEff > EPS) ? (min(vtLen, maxFrictionDeltaV) / wEffSafe) : 0.0;
        float2 vtDir              = (vtLen > EPS) ? (vt / vtLen) : 0.0;

        [unroll]
        for (uint sApply = 0u; sApply < COLLISION_MAX_NODES; sApply++)
        {
            uint nodeGi = nodeGiCache[sApply];
            if (nodeGi == COLLISION_INVALID_U32)
                continue;
            if (fixedCache[sApply])
                continue;

            uint  nodeLi       = nodeLiCache[sApply];
            float invMass      = invMassCache[sApply];
            float betaScaled   = betaScaledCache[sApply];
            float betaUnscaled = betaUnscaledCache[sApply];

            float2 dV = -invMass * betaScaled * dt * appliedDLambda * nrm;
            dV += invMass * betaUnscaled * restitutionImpulse * nrm;

            if (frictionImpulse > EPS && vtLen > EPS)
                dV += -invMass * betaUnscaled * frictionImpulse * vtDir;

            if (maxDv > EPS)
            {
                float dVLen = length(dV);
                if (dVLen > maxDv)
                    dV *= maxDv / dVLen;
            }

            if (!all(isfinite(dV)))
                continue;

            XPBI_COL_APPLY_DV(nodeLi, nodeGi, dV);
        }

        XPBI_COL_WRITE_LAMBDA(lambdaIdx, lambdaNew);
    }
}