// ============================================================================
// CollisionSolveBlock.NodeOwned.hlsl
//
// Unified fully parallel collision solve for slave-owned node contacts.
// Works for either fine or coarse layers by rebinding:
//
//   _ColNodeContacts
//   _ColNodeContactCount
//   _ColNodeContactLambda
//   _ColNodeContactStride
//
// Each thread owns one slave node (li, gi), processes only its manifold,
// writes only its own velocity and its own lambda slots, and reads the
// master side from frozen snapshot buffers.
// ============================================================================

if (li < active)
{
    float dt  = max(_Dt, EPS);
    float dt2 = dt * dt;

    if (gi == COLLISION_INVALID_U32)
        return;

    if (XPBI_NEIGHBOR_FIXED(li, gi))
        return;

    float invMassI = XPBI_INV_MASS(li, gi);
    if (!(invMassI > EPS))
        return;

    float alphaCollision  = max(_CollisionCompliance, 0.0);
    float targetSeparation = max(EPS, _CollisionSkinScale * support);
    float slop = max(_CollisionSlop, 0.0) * support;
    float maxDv = max(_CollisionMaxDv, 0.0);

    float2 xI = XPBI_POS(li, gi);
    float2 vI = XPBI_VEL(li, gi);

    uint contactCount = XPBI_ColNodeContactCountOf(li);

    [loop]
    for (uint slot = 0u; slot < _ColNodeContactStride; slot++)
    {
        if (slot >= contactCount)
            break;

        XPBI_NodeContact c;
        if (!XPBI_ColReadContact(li, slot, c))
            continue;

        if (c.slaveGi != gi)
            continue;

        if (c.ownerSlave == c.ownerMaster)
            continue;

        float scale = c.scale;
        if (!(scale > EPS))
            continue;

        float penRaw = max(c.pen, 0.0);
        if (penRaw <= slop)
            continue;

        float2 nrm;
        if (!XPBI_ColReadNormal(c, nrm))
            continue;

        float2 xMasterPred = XPBI_ColSampleMasterPredPos(c, dt);
        float2 xSelfPred   = xI + vI * dt;

        float penEff = max(penRaw - slop, 0.0);
        float bias   = min(max(_CollisionPenBias, 0.0) * penEff, max(_CollisionMaxBias, 0.0));
        float maxPush = max(_CollisionMaxPush, 0.0);

        float Cbase = dot(nrm, xSelfPred - xMasterPred) - targetSeparation;
        float Cn = scale * (Cbase - bias);
        Cn = max(Cn, -scale * (penEff + maxPush));

        if (Cn >= 0.0)
            continue;

        float wTerm = dt2 * invMassI * (scale * scale);
        if (wTerm <= EPS && alphaCollision <= EPS)
            continue;

        float lambdaPrev = XPBI_ColReadLambda(li, slot);

        float dLambda = -(Cn + alphaCollision * lambdaPrev) / max(wTerm + alphaCollision, EPS);
        if (!isfinite(dLambda))
            continue;

        float lambdaNewRaw = max(0.0, lambdaPrev + dLambda);
        float appliedDLambda = (lambdaNewRaw - lambdaPrev) * saturate(_CollisionRelaxation);

        if (appliedDLambda <= 0.0)
            continue;

        float lambdaNew = lambdaPrev + appliedDLambda;

        float2 vMaster = XPBI_ColSampleMasterVel(c);
        float2 vRel = vI - vMaster;

        float vn = dot(vRel, nrm);
        float2 vt = vRel - vn * nrm;
        float vtLen = length(vt);
        float2 vtDir = (vtLen > EPS) ? (vt / vtLen) : 0.0;

        float restitution = saturate(_CollisionRestitution); 
        float restitutionThreshold = max(_CollisionRestitutionThreshold, 0.0);
        float restitutionDeltaVn = restitution * max(-vn - restitutionThreshold, 0.0);

        float friction = saturate(_CollisionFriction);
        float maxFrictionDeltaV = friction * max(-vn, 0.0);
        float frictionDeltaV = min(vtLen, maxFrictionDeltaV);

        float2 dV = -(invMassI * scale * dt * appliedDLambda) * nrm;
        dV += restitutionDeltaVn * nrm;

        if (frictionDeltaV > EPS && vtLen > EPS)
            dV += -frictionDeltaV * vtDir;

        dV = XPBI_ColClampDeltaV(dV, maxDv, EPS);

        if (!all(isfinite(dV)))
            continue;

        vI += dV;
        XPBI_ColWriteLambda(li, slot, lambdaNew);
    }

    XPBI_SET_VEL(li, gi, vI);
}