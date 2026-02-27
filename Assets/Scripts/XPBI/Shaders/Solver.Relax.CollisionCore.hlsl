// Expected locals in scope:
//   uint li, gi, active, dtLi
//   float support

int ownerICollision = (li < active) ? _DtOwnerByLocal[li] : -1;
if (ownerICollision >= 0)
{
    float targetSeparation = max(EPS, _CollisionSupportScale * support);
    float targetSeparationSq = targetSeparation * targetSeparation;
    float alphaCollision = _CollisionCompliance / max(_Dt * _Dt, EPS);

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

        float2 dx = XPBI_POS(gjLi, gj) - XPBI_POS(li, gi);
        float distSq = dot(dx, dx);
        if (distSq <= EPS * EPS || distSq > targetSeparationSq) continue;

        float dist = sqrt(distSq);
        float Cn = dist - targetSeparation;
        if (Cn >= 0.0) continue;

        float2 nrm = dx / max(dist, EPS);

        float invMassJCol = fixedJ ? 0.0 : XPBI_INV_MASS(gjLi, gj);
        float denomCol = invMassICol + invMassJCol;
        if (denomCol <= EPS) continue;

        uint lambdaIdx = contactBase + kCol;
        float lambdaPrev = _CollisionLambda[lambdaIdx];

        float dLambdaCol = -(Cn + alphaCollision * lambdaPrev) / (denomCol + alphaCollision);
        float lambdaNew = max(0.0, lambdaPrev + dLambdaCol);
        dLambdaCol = lambdaNew - lambdaPrev;
        if (dLambdaCol <= 0.0) continue;

        float2 velI = XPBI_VEL(li, gi);
        float2 velJ = XPBI_VEL(gjLi, gj);

        float2 relVelBefore = velJ - velI;
        float relNormalBefore = dot(relVelBefore, nrm);

        float impulseVel = dLambdaCol / max(_Dt, EPS);

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

            if (!fixedI) velI -= invMassICol * restitutionImpulseVel * nrm;
            if (!fixedJ) velJ += invMassJCol * restitutionImpulseVel * nrm;
        }

        XPBI_SET_VEL(li, gi, velI);
        XPBI_SET_VEL(gjLi, gj, velJ);
        _CollisionLambda[lambdaIdx] = lambdaNew;
    }
}