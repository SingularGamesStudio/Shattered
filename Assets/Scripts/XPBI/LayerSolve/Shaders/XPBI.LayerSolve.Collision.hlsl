// Expected locals in scope:
//   uint li, gi, active, dtLi
//   float support

int ownerICollision = (li < active) ? _DtCollisionOwnerByLocal[li] : -1;
if (ownerICollision >= 0 && _ActiveCount == _FineCount)
{
    float targetSeparation = max(EPS, _CollisionSupportScale * support);
    float targetSeparationSq = targetSeparation * targetSeparation;
    float dt = max(_Dt, EPS);
    float dt2 = dt * dt;
    float alphaCollision = max(_CollisionCompliance, 0.0);

    uint baseIdxCol = dtLi * _DtNeighborCount;
    uint rawCountCol = _DtNeighborCounts[dtLi];
    uint nCountCol = min(rawCountCol, _DtNeighborCount);
    nCountCol = min(nCountCol, targetNeighborCount);

    uint contactBase = dtLi * _DtNeighborCount * xferColManifoldSlots;

    bool fixedI = XPBI_NEIGHBOR_FIXED(li, gi);
    float invMassICol = fixedI ? 0.0 : XPBI_INV_MASS(li, gi);

    [loop] for (uint kCol = 0u; kCol < nCountCol; kCol++)
    {
        uint gjLi = _DtNeighbors[baseIdxCol + kCol];
        if (gjLi == ~0u || gjLi >= active) continue;

        int ownerJ = _DtCollisionOwnerByLocal[gjLi];
        if (ownerJ < 0 || ownerJ == ownerICollision) continue;

        uint gj = XPBI_GET_GJ(gjLi);
        if (gj == ~0u || gj <= gi) continue;

        bool fixedJ = XPBI_NEIGHBOR_FIXED(gjLi, gj);
        if (fixedI && fixedJ) continue;

        float invMassJCol = fixedJ ? 0.0 : XPBI_INV_MASS(gjLi, gj);
        float wTerm = dt2 * (invMassICol + invMassJCol);
        if (wTerm <= EPS && alphaCollision <= EPS) continue;

        uint pairBase = contactBase + kCol * xferColManifoldSlots;

        if (_UseTransferredCollisions != 0u)
        {
            [unroll]
            for (uint slotOffset = 0u; slotOffset < xferColManifoldSlots; slotOffset++)
            {
                uint slot = pairBase + slotOffset;
                if (_XferColCount[slot] == 0u)
                    continue;

                float pen = asfloat(_XferColPenBits[slot]);
                if (!(pen > EPS))
                    continue;

                float2 nrm = float2(asfloat(_XferColNXBits[slot]), asfloat(_XferColNYBits[slot]));
                float n2 = dot(nrm, nrm);
                if (!(n2 > 1e-12))
                    continue;
                nrm *= rsqrt(n2);

                float2 xiCand = XPBI_POS(li, gi) + XPBI_VEL(li, gi) * dt;
                float2 xjCand = XPBI_POS(gjLi, gj) + XPBI_VEL(gjLi, gj) * dt;
                float Cn = dot(nrm, xjCand - xiCand) - targetSeparation;
                Cn = min(Cn, -pen);
                if (Cn >= 0.0)
                    continue;

                float lambdaPrev = _CollisionLambda[slot];
                float dLambdaCol = -(Cn + alphaCollision * lambdaPrev) / max(wTerm + alphaCollision, EPS);
                float lambdaNew = max(0.0, lambdaPrev + dLambdaCol);
                float appliedDLambda = lambdaNew - lambdaPrev;
                if (appliedDLambda <= 0.0)
                    continue;

                float2 dVi = -invMassICol * dt * appliedDLambda * nrm;
                float2 dVj = invMassJCol * dt * appliedDLambda * nrm;
                if (!all(isfinite(dVi)) || !all(isfinite(dVj)))
                    continue;

                if (!fixedI)
                    XPBI_SET_VEL(li, gi, XPBI_VEL(li, gi) + dVi);
                if (!fixedJ)
                    XPBI_SET_VEL(gjLi, gj, XPBI_VEL(gjLi, gj) + dVj);

                _CollisionLambda[slot] = lambdaNew;
            }
            continue;
        }

        float2 xiCandFine = XPBI_POS(li, gi) + XPBI_VEL(li, gi) * dt;
        float2 xjCandFine = XPBI_POS(gjLi, gj) + XPBI_VEL(gjLi, gj) * dt;
        float2 dx = xjCandFine - xiCandFine;
        float distSq = dot(dx, dx);
        if (distSq <= EPS * EPS || distSq > targetSeparationSq) continue;

        float dist = sqrt(distSq);
        float Cn = dist - targetSeparation;
        if (Cn >= 0.0) continue;

        float2 nrm = dx / max(dist, EPS);
        uint lambdaIdx = pairBase;
        float lambdaPrev = _CollisionLambda[lambdaIdx];
        float dLambdaCol = -(Cn + alphaCollision * lambdaPrev) / max(wTerm + alphaCollision, EPS);
        float lambdaNew = max(0.0, lambdaPrev + dLambdaCol);
        float appliedDLambda = lambdaNew - lambdaPrev;
        if (appliedDLambda <= 0.0) continue;

        float2 dVi = -invMassICol * dt * appliedDLambda * nrm;
        float2 dVj = invMassJCol * dt * appliedDLambda * nrm;
        if (!all(isfinite(dVi)) || !all(isfinite(dVj))) continue;

        if (!fixedI)
            XPBI_SET_VEL(li, gi, XPBI_VEL(li, gi) + dVi);
        if (!fixedJ)
            XPBI_SET_VEL(gjLi, gj, XPBI_VEL(gjLi, gj) + dVj);

        _CollisionLambda[lambdaIdx] = lambdaNew;
    }
}