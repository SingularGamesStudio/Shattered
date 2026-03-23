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
            bool anySlotUsed = false;

            [unroll]
            for (uint slotOffset = 0u; slotOffset < xferColManifoldSlots; slotOffset++)
            {
                uint slot = pairBase + slotOffset;
                if (_XferColCount[slot] == 0u)
                    continue;

                anySlotUsed = true;

                float pen = asfloat(_XferColPenBits[slot]);
                if (!(pen > EPS))
                    continue;

                float2 nrm = float2(asfloat(_XferColNXBits[slot]), asfloat(_XferColNYBits[slot]));
                float n2 = dot(nrm, nrm);
                if (!(n2 > 1e-12))
                    continue;
                nrm *= rsqrt(n2);

                uint qaGi = _XferColQAGi[slot];
                uint qbGi = _XferColQBGi[slot];
                uint oaGi = _XferColOAGi[slot];
                uint obGi = _XferColOBGi[slot];
                if (qaGi == ~0u || qbGi == ~0u || oaGi == ~0u || obGi == ~0u)
                    continue;

                uint qaLi = LocalIndexFromGlobal(qaGi);
                uint qbLi = LocalIndexFromGlobal(qbGi);
                uint oaLi = LocalIndexFromGlobal(oaGi);
                uint obLi = LocalIndexFromGlobal(obGi);
                if (qaLi == ~0u || qbLi == ~0u || oaLi == ~0u || obLi == ~0u)
                    continue;
                if (qaLi >= active || qbLi >= active || oaLi >= active || obLi >= active)
                    continue;

                float sSeg = saturate(asfloat(_XferColSBits[slot]));
                float tSeg = saturate(asfloat(_XferColTBits[slot]));

                bool fixedA0 = XPBI_NEIGHBOR_FIXED(qaLi, qaGi);
                bool fixedA1 = XPBI_NEIGHBOR_FIXED(qbLi, qbGi);
                bool fixedB0 = XPBI_NEIGHBOR_FIXED(oaLi, oaGi);
                bool fixedB1 = XPBI_NEIGHBOR_FIXED(obLi, obGi);

                float invMassA0 = fixedA0 ? 0.0 : XPBI_INV_MASS(qaLi, qaGi);
                float invMassA1 = fixedA1 ? 0.0 : XPBI_INV_MASS(qbLi, qbGi);
                float invMassB0 = fixedB0 ? 0.0 : XPBI_INV_MASS(oaLi, oaGi);
                float invMassB1 = fixedB1 ? 0.0 : XPBI_INV_MASS(obLi, obGi);

                float wA0 = (1.0 - sSeg) * invMassA0;
                float wA1 = sSeg * invMassA1;
                float wB0 = (1.0 - tSeg) * invMassB0;
                float wB1 = tSeg * invMassB1;
                float wTermWeighted = dt2 * (wA0 + wA1 + wB0 + wB1);
                if (wTermWeighted <= EPS && alphaCollision <= EPS)
                    continue;

                float2 qaCand = XPBI_POS(qaLi, qaGi) + XPBI_VEL(qaLi, qaGi) * dt;
                float2 qbCand = XPBI_POS(qbLi, qbGi) + XPBI_VEL(qbLi, qbGi) * dt;
                float2 oaCand = XPBI_POS(oaLi, oaGi) + XPBI_VEL(oaLi, oaGi) * dt;
                float2 obCand = XPBI_POS(obLi, obGi) + XPBI_VEL(obLi, obGi) * dt;

                float2 cA = lerp(qaCand, qbCand, sSeg);
                float2 cB = lerp(oaCand, obCand, tSeg);
                float Cn = dot(nrm, cB - cA) - targetSeparation;
                if (Cn >= 0.0)
                    continue;
                Cn = min(Cn, -pen);

                float lambdaPrev = _CollisionLambda[slot];
                float dLambdaCol = -(Cn + alphaCollision * lambdaPrev) / max(wTermWeighted + alphaCollision, EPS);
                float lambdaNew = max(0.0, lambdaPrev + dLambdaCol);
                float appliedDLambda = lambdaNew - lambdaPrev;
                if (appliedDLambda <= 0.0)
                    continue;

                float2 dVA0 = -wA0 * dt * appliedDLambda * nrm;
                float2 dVA1 = -wA1 * dt * appliedDLambda * nrm;
                float2 dVB0 = wB0 * dt * appliedDLambda * nrm;
                float2 dVB1 = wB1 * dt * appliedDLambda * nrm;
                if (!all(isfinite(dVA0)) || !all(isfinite(dVA1)) || !all(isfinite(dVB0)) || !all(isfinite(dVB1)))
                    continue;

                if (!fixedA0)
                    XPBI_SET_VEL(qaLi, qaGi, XPBI_VEL(qaLi, qaGi) + dVA0);
                if (!fixedA1)
                    XPBI_SET_VEL(qbLi, qbGi, XPBI_VEL(qbLi, qbGi) + dVA1);
                if (!fixedB0)
                    XPBI_SET_VEL(oaLi, oaGi, XPBI_VEL(oaLi, oaGi) + dVB0);
                if (!fixedB1)
                    XPBI_SET_VEL(obLi, obGi, XPBI_VEL(obLi, obGi) + dVB1);

                _CollisionLambda[slot] = lambdaNew;
            }

            if (anySlotUsed)
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