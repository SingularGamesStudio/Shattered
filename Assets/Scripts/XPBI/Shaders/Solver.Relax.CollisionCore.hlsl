// Expected locals in scope:
//   uint li, gi, active, dtLi
//   float support

int ownerICollision = (li < active) ? _DtOwnerByLocal[li] : -1;
if (ownerICollision >= 0 && _ActiveCount == _FineCount)
{
    float targetSeparation = max(EPS, _CollisionSupportScale * support);
    float targetSeparationSq = targetSeparation * targetSeparation;
    float alphaCollision = _CollisionCompliance / max(_Dt * _Dt, EPS);
    float invDtCollision = 1.0 / max(_Dt, EPS);
    float maxDeltaVPerContact = targetSeparation * invDtCollision;
    float maxSpeedCollision = 4.0 * targetSeparation * invDtCollision;
    float maxSpeedCollision2 = maxSpeedCollision * maxSpeedCollision;

    uint baseIdxCol = dtLi * _DtNeighborCount;
    uint rawCountCol = _DtNeighborCounts[dtLi];
    uint nCountCol = min(rawCountCol, _DtNeighborCount);
    nCountCol = min(nCountCol, targetNeighborCount);

    uint contactBase = dtLi * _DtNeighborCount;

    bool fixedI = XPBI_NEIGHBOR_FIXED(li, gi);
    float invMassICol = fixedI ? 0.0 : XPBI_INV_MASS(li, gi);

    [loop] for (uint kCol = 0u; kCol < nCountCol; kCol++)
    {
        uint gjLi = _DtNeighbors[baseIdxCol + kCol];
        if (gjLi == ~0u || gjLi >= active) continue;

        int ownerJ = _DtOwnerByLocal[gjLi];
        if (ownerJ < 0 || ownerJ == ownerICollision) continue;

        uint gj = XPBI_GET_GJ(gjLi);
        if (gj == ~0u || gj <= gi) continue;

        bool fixedJ = XPBI_NEIGHBOR_FIXED(gjLi, gj);
        if (fixedI && fixedJ) continue;

        uint lambdaIdx = contactBase + kCol;
        float lambdaPrev = _CollisionLambda[lambdaIdx];
        float invMassJCol = fixedJ ? 0.0 : XPBI_INV_MASS(gjLi, gj);

        if (_UseTransferredCollisions != 0u)
        {
            uint slot = contactBase + kCol;
            uint c = _XferColCount[slot];
            if (c == 0u) continue;

            float penSum = asfloat(_XferColPenBits[slot]);
            float pen = penSum / max((float)c, 1.0);
            if (!(pen > EPS)) continue;

            float2 nSum = float2(asfloat(_XferColNXBits[slot]), asfloat(_XferColNYBits[slot]));
            float n2 = dot(nSum, nSum);
            if (!(n2 > 1e-12)) continue;
            float2 nrm = nSum * rsqrt(n2);

            float Cn = max(-pen, -targetSeparation);
            float denomCol = invMassICol + invMassJCol;
            if (denomCol <= EPS) continue;

            float dLambdaCol = -(Cn + alphaCollision * lambdaPrev) / (denomCol + alphaCollision);
            float lambdaNew = max(0.0, lambdaPrev + dLambdaCol);
            dLambdaCol = lambdaNew - lambdaPrev;
            if (dLambdaCol <= 0.0) continue;

            float2 velI = XPBI_VEL(li, gi);
            float2 velJ = XPBI_VEL(gjLi, gj);

            float2 relVelBefore = velJ - velI;
            float relNormalBefore = dot(relVelBefore, nrm);

            float impulseVel = dLambdaCol * invDtCollision;
            float maxInvMassCol = max(invMassICol, invMassJCol);
            float impulseCap = maxDeltaVPerContact / max(maxInvMassCol, EPS);
            if (impulseVel > impulseCap) impulseVel = impulseCap;

            if (!fixedI) velI -= invMassICol * impulseVel * nrm;
            if (!fixedJ) velJ += invMassJCol * impulseVel * nrm;

            if (_CollisionFriction > 0.0)
            {
                float2 relVel = velJ - velI;
                float relNormal = dot(relVel, nrm);
                float2 tangentVel = relVel - relNormal * nrm;
                float tangentLen = length(tangentVel);
                if (tangentLen > EPS)
                {
                    float2 tangentDir = tangentVel / tangentLen;
                    float frictionImpulseVel = -tangentLen / max(denomCol, EPS);
                    float frictionLimit = _CollisionFriction * impulseVel;
                    frictionImpulseVel = clamp(frictionImpulseVel, -frictionLimit, frictionLimit);

                    if (!fixedI) velI -= invMassICol * frictionImpulseVel * tangentDir;
                    if (!fixedJ) velJ += invMassJCol * frictionImpulseVel * tangentDir;
                }
            }

            if (_CollisionRestitution > 0.0 && relNormalBefore < -_CollisionRestitutionThreshold)
            {
                float restitutionImpulseVel =
                (-(1.0 + _CollisionRestitution) * relNormalBefore) / max(denomCol, EPS);
                restitutionImpulseVel = min(restitutionImpulseVel, impulseCap);

                if (!fixedI) velI -= invMassICol * restitutionImpulseVel * nrm;
                if (!fixedJ) velJ += invMassJCol * restitutionImpulseVel * nrm;
            }

            if (!all(isfinite(velI))) velI = 0.0;
            if (!all(isfinite(velJ))) velJ = 0.0;

            float vI2 = dot(velI, velI);
            if (vI2 > maxSpeedCollision2) velI *= maxSpeedCollision * rsqrt(max(vI2, EPS * EPS));
            float vJ2 = dot(velJ, velJ);
            if (vJ2 > maxSpeedCollision2) velJ *= maxSpeedCollision * rsqrt(max(vJ2, EPS * EPS));

            XPBI_SET_VEL(li, gi, velI);
            XPBI_SET_VEL(gjLi, gj, velJ);
            _CollisionLambda[lambdaIdx] = lambdaNew;
            continue;
        }

        float2 dx = XPBI_POS(gjLi, gj) - XPBI_POS(li, gi);
        float distSq = dot(dx, dx);
        if (distSq <= EPS * EPS || distSq > targetSeparationSq) continue;

        float dist = sqrt(distSq);
        float Cn = max(dist - targetSeparation, -targetSeparation);
        if (Cn >= 0.0) continue;

        float2 nrm = dx / max(dist, EPS);
        float denomCol = invMassICol + invMassJCol;
        if (denomCol <= EPS) continue;

        float dLambdaCol = -(Cn + alphaCollision * lambdaPrev) / (denomCol + alphaCollision);
        float lambdaNew = max(0.0, lambdaPrev + dLambdaCol);
        dLambdaCol = lambdaNew - lambdaPrev;
        if (dLambdaCol <= 0.0) continue;

        float2 velI = XPBI_VEL(li, gi);
        float2 velJ = XPBI_VEL(gjLi, gj);

        float2 relVelBefore = velJ - velI;
        float relNormalBefore = dot(relVelBefore, nrm);

        float impulseVel = dLambdaCol * invDtCollision;
        float maxInvMassCol = max(invMassICol, invMassJCol);
        float impulseCap = maxDeltaVPerContact / max(maxInvMassCol, EPS);
        if (impulseVel > impulseCap) impulseVel = impulseCap;

        if (!fixedI) velI -= invMassICol * impulseVel * nrm;
        if (!fixedJ) velJ += invMassJCol * impulseVel * nrm;

        if (_CollisionFriction > 0.0)
        {
            float2 relVel = velJ - velI;
            float relNormal = dot(relVel, nrm);
            float2 tangentVel = relVel - relNormal * nrm;
            float tangentLen = length(tangentVel);
            if (tangentLen > EPS)
            {
                float2 tangentDir = tangentVel / tangentLen;
                float frictionImpulseVel = -tangentLen / max(denomCol, EPS);
                float frictionLimit = _CollisionFriction * impulseVel;
                frictionImpulseVel = clamp(frictionImpulseVel, -frictionLimit, frictionLimit);

                if (!fixedI) velI -= invMassICol * frictionImpulseVel * tangentDir;
                if (!fixedJ) velJ += invMassJCol * frictionImpulseVel * tangentDir;
            }
        }

        if (_CollisionRestitution > 0.0 && relNormalBefore < -_CollisionRestitutionThreshold)
        {
            float restitutionImpulseVel =
            (-(1.0 + _CollisionRestitution) * relNormalBefore) / max(denomCol, EPS);
            restitutionImpulseVel = min(restitutionImpulseVel, impulseCap);

            if (!fixedI) velI -= invMassICol * restitutionImpulseVel * nrm;
            if (!fixedJ) velJ += invMassJCol * restitutionImpulseVel * nrm;
        }

        if (!all(isfinite(velI))) velI = 0.0;
        if (!all(isfinite(velJ))) velJ = 0.0;

        float vI2 = dot(velI, velI);
        if (vI2 > maxSpeedCollision2) velI *= maxSpeedCollision * rsqrt(max(vI2, EPS * EPS));
        float vJ2 = dot(velJ, velJ);
        if (vJ2 > maxSpeedCollision2) velJ *= maxSpeedCollision * rsqrt(max(vJ2, EPS * EPS));

        XPBI_SET_VEL(li, gi, velI);
        XPBI_SET_VEL(gjLi, gj, velJ);
        _CollisionLambda[lambdaIdx] = lambdaNew;
    }
}