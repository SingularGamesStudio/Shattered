// Collision debug stat slots.
// 0..15 keep the original panel-compatible indices; 16+ are detailed rejects.
static const uint STAT_BOUNDARY_FLAGS = 0u;
static const uint STAT_BOUNDARY_EDGES = 1u;
static const uint STAT_OWNER_EDGE_OVERFLOW = 2u;
static const uint STAT_MAX_EDGE_BIN_LOAD = 3u;
static const uint STAT_MAX_VERT_BIN_LOAD = 4u;
static const uint STAT_EDGE_BIN_OVERFLOWS = 5u;
static const uint STAT_VERT_BIN_OVERFLOWS = 6u;
static const uint STAT_VERTEX_CANDIDATES = 7u;
static const uint STAT_VERTEX_FEATURE_HITS = 8u;
static const uint STAT_VERTEX_WITHIN_SUPPORT = 9u;
static const uint STAT_VERTEX_CONTACTS_WRITTEN = 10u;
static const uint STAT_EDGE_PAIR_CANDIDATES = 11u;
static const uint STAT_EDGE_WITHIN_SUPPORT = 12u;
static const uint STAT_EDGE_CONTACTS_EMITTED = 13u;
static const uint STAT_OWNER_AABB_REJECTS = 14u;
static const uint STAT_VERTEX_STAGE_OVERFLOW = 15u;

static const uint STAT_BUILD_HALFEDGE_THREADS = 16u;
static const uint STAT_BUILD_REJECT_NON_BOUNDARY_FLAG = 17u;
static const uint STAT_BUILD_REJECT_INVALID_FACE = 18u;
static const uint STAT_BUILD_REJECT_NOT_INTERNAL_BOUNDARY = 19u;
static const uint STAT_BUILD_REJECT_INVALID_GLOBAL_VERTEX = 20u;
static const uint STAT_BUILD_REJECT_INVALID_OWNER = 21u;
static const uint STAT_BUILD_OWNER_EDGE_REF_WRITES = 22u;
static const uint STAT_BUILD_OWNER_EDGE_REF_OVERFLOW = 23u;

static const uint STAT_VERTEX_WORK_ITEMS = 24u;
static const uint STAT_VERTEX_REJECT_PAIR_OOB = 25u;
static const uint STAT_VERTEX_REJECT_SAME_OWNER = 26u;
static const uint STAT_VERTEX_REJECT_OWNER_OVERFLOW = 27u;
static const uint STAT_VERTEX_REJECT_NO_OWNER_EDGE_REF = 28u;
static const uint STAT_VERTEX_REJECT_INVALID_VGI = 29u;
static const uint STAT_VERTEX_REJECT_NO_BOUNDARY_HIT = 30u;
static const uint STAT_VERTEX_REJECT_SUPPORT = 31u;
static const uint STAT_VERTEX_REJECT_DEGENERATE_NORMAL = 32u;
static const uint STAT_VERTEX_COMPACT_REJECT_INVALID = 33u;
static const uint STAT_VERTEX_COMPACT_VALID = 34u;
static const uint STAT_VERTEX_REJECT_AABB = 35u;

static const uint STAT_EDGE_WORK_ITEMS = 36u;
static const uint STAT_EDGE_REJECT_PAIR_OOB = 37u;
static const uint STAT_EDGE_REJECT_SWAP = 38u;
static const uint STAT_EDGE_REJECT_SAME_OWNER = 39u;
static const uint STAT_EDGE_REJECT_OWNER_OVERFLOW = 40u;
static const uint STAT_EDGE_REJECT_NO_OWNER_EDGE_REF = 41u;
static const uint STAT_EDGE_REJECT_INVALID_EDGE_A = 42u;
static const uint STAT_EDGE_BIN_CELLS_VISITED = 43u;
static const uint STAT_EDGE_BIN_EDGE_REFS_SCANNED = 44u;
static const uint STAT_EDGE_REJECT_OWNER_MISMATCH_B = 45u;
static const uint STAT_EDGE_REJECT_INVALID_EDGE_B = 46u;
static const uint STAT_EDGE_REJECT_EDGE_BBOX = 47u;
static const uint STAT_EDGE_REJECT_NO_INTERSECTION = 48u;
static const uint STAT_EDGE_POINT_INTERSECTIONS = 49u;
static const uint STAT_EDGE_REJECT_NON_POINT_HIT = 50u;
static const uint STAT_EDGE_REJECT_ENDPOINT_INTERSECTION = 51u;
static const uint STAT_EDGE_REJECT_NON_CANONICAL_BIN = 52u;
static const uint STAT_EDGE_REJECT_DEGENERATE_NORMAL = 53u;
static const uint STAT_EDGE_REJECT_NO_PENETRATION = 54u;
static const uint STAT_EDGE_REJECT_AABB = 55u;
static const uint STAT_BUILD_REJECT_INVALID_LOCAL_VERTEX = 56u;
static const uint STAT_BUILD_REJECT_DEGENERATE_ENDPOINTS = 57u;

[numthreads(64, 1, 1)]
void ClearState(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;

    if (i < COLLISION_DEBUG_STAT_COUNT)
    {
        _CollisionDebugStats[i] = 0u;
    }

    if (i < _OwnerCount)
    {
        _OwnerBoundaryEdgeCounts[i] = 0;
        _OwnerBoundaryOverflow[i] = 0;
    }

    if (_OwnerCount > 0u)
    {
        uint totalBins = _OwnerBinBase[_OwnerCount - 1] + _OwnerBinDim[_OwnerCount - 1].x * _OwnerBinDim[_OwnerCount - 1].y;

        if (i < totalBins)
        {
            _EdgeBinCounts[i] = 0;
            _EdgeBinOverflow[i] = 0;
            _VertBinCounts[i] = 0;
            _VertBinOverflow[i] = 0;
        }
    }

    if (i < _DtHalfEdgeCount)
    {
        _BoundaryEdgeOwner[i] = INVALID_U32;
        _BoundaryEdgeV0Gi[i] = INVALID_U32;
        _BoundaryEdgeV1Gi[i] = INVALID_U32;
        _BoundaryOutwardIsRight[i] = 0u;
        _BoundaryEdgeP0[i] = 0.0;
        _BoundaryEdgeP1[i] = 0.0;
        _BoundaryEdgeNOut[i] = 0.0;
        _BoundaryEdgeNIn[i] = 0.0;
        _BoundaryEdgePseudoN0[i] = 0.0;
        _BoundaryEdgePseudoN1[i] = 0.0;

        _BoundaryVertexOwner[i] = INVALID_U32;
        _BoundaryVertexGi[i] = INVALID_U32;
        _BoundaryVertexP[i] = 0.0;
        _BoundaryVertexPseudoN[i] = 0.0;
    }

    if (i == 0u)
    {
        _ContactCount[0] = 0;
    }
}

[numthreads(64, 1, 1)]
void BuildBoundaryFeatures(uint3 tid : SV_DispatchThreadID)
{
    uint heId = tid.x;
    if (heId >= _DtHalfEdgeCount) return;
    InterlockedAdd(_CollisionDebugStats[STAT_BUILD_HALFEDGE_THREADS], 1u);
    if (_DtBoundaryEdgeFlags[heId] == 0u)
    {
        InterlockedAdd(_CollisionDebugStats[STAT_BUILD_REJECT_NON_BOUNDARY_FLAG], 1u);
        return;
    }
    InterlockedAdd(_CollisionDebugStats[STAT_BOUNDARY_FLAGS], 1u);

    DT_HalfEdge he = _DtHalfEdges[heId];
    if (he.t < 0 || he.next < 0 || (uint)he.next >= _DtHalfEdgeCount)
    {
        InterlockedAdd(_CollisionDebugStats[STAT_BUILD_REJECT_INVALID_FACE], 1u);
        return;
    }

    int tw = he.twin;
    bool twValid = (tw >= 0) && ((uint)tw < _DtHalfEdgeCount);
    int twFace = twValid ? _DtHalfEdges[(uint)tw].t : -1;

    bool heInternal = ((uint)he.t < _DtTriCount) ? (_DtTriInternal[he.t] != 0u) : false;
    bool twInternal = (twFace >= 0 && (uint)twFace < _DtTriCount) ? (_DtTriInternal[twFace] != 0u) : false;

    if (!(heInternal && !twInternal))
    {
        InterlockedAdd(_CollisionDebugStats[STAT_BUILD_REJECT_NOT_INTERNAL_BOUNDARY], 1u);
        return;
    }
    InterlockedAdd(_CollisionDebugStats[STAT_BOUNDARY_EDGES], 1u);

    DT_HalfEdge heNext = _DtHalfEdges[(uint)he.next];
    if (heNext.next < 0 || (uint)heNext.next >= _DtHalfEdgeCount)
    {
        InterlockedAdd(_CollisionDebugStats[STAT_BUILD_REJECT_INVALID_FACE], 1u);
        return;
    }
    DT_HalfEdge heNextNext = _DtHalfEdges[(uint)heNext.next];

    if (he.v < 0 || heNext.v < 0 || heNextNext.v < 0)
    {
        InterlockedAdd(_CollisionDebugStats[STAT_BUILD_REJECT_INVALID_GLOBAL_VERTEX], 1u);
        return;
    }

    if ((uint)he.v >= _DtLocalVertexCount || (uint)heNext.v >= _DtLocalVertexCount || (uint)heNextNext.v >= _DtLocalVertexCount)
    {
        InterlockedAdd(_CollisionDebugStats[STAT_BUILD_REJECT_INVALID_LOCAL_VERTEX], 1u);
        return;
    }

    uint v0Local = (uint)he.v;
    uint v1Local = (uint)heNext.v;
    uint vOppLocal = (uint)heNextNext.v;

    uint v0Gi = _DtGlobalVertexByLocal[v0Local];
    uint v1Gi = _DtGlobalVertexByLocal[v1Local];
    uint vOppGi = _DtGlobalVertexByLocal[vOppLocal];
    if (v0Gi == INVALID_U32 || v1Gi == INVALID_U32 || vOppGi == INVALID_U32)
    {
        InterlockedAdd(_CollisionDebugStats[STAT_BUILD_REJECT_INVALID_GLOBAL_VERTEX], 1u);
        return;
    }

    if (v0Gi == v1Gi)
    {
        InterlockedAdd(_CollisionDebugStats[STAT_BUILD_REJECT_DEGENERATE_ENDPOINTS], 1u);
        return;
    }

    int ownerI = _DtCollisionOwnerByLocal[v0Local];
    if (ownerI < 0)
    {
        InterlockedAdd(_CollisionDebugStats[STAT_BUILD_REJECT_INVALID_OWNER], 1u);
        return;
    }
    uint owner = (uint)ownerI;

    float2 p0 = PredPos(v0Gi);
    float2 p1 = PredPos(v1Gi);
    float2 pOpp = PredPos(vOppGi);

    float side = Cross2(p1 - p0, pOpp - p0);
    uint outwardIsRight = (side > 0.0) ? 1u : 0u;

    float2 nOut = BoundaryOutwardNormal(outwardIsRight, p0, p1);
    float2 nIn = -nOut;
    float2 pseudoN0 = SafeNormalize(_DtBoundaryNormals[v0Local]);
    float2 pseudoN1 = SafeNormalize(_DtBoundaryNormals[v1Local]);

    _BoundaryEdgeOwner[heId] = owner;
    _BoundaryEdgeV0Gi[heId] = v0Gi;
    _BoundaryEdgeV1Gi[heId] = v1Gi;
    _BoundaryOutwardIsRight[heId] = outwardIsRight;
    _BoundaryEdgeP0[heId] = p0;
    _BoundaryEdgeP1[heId] = p1;
    _BoundaryEdgeNOut[heId] = nOut;
    _BoundaryEdgeNIn[heId] = nIn;
    _BoundaryEdgePseudoN0[heId] = pseudoN0;
    _BoundaryEdgePseudoN1[heId] = pseudoN1;

    _BoundaryVertexOwner[heId] = owner;
    _BoundaryVertexGi[heId] = v0Gi;
    _BoundaryVertexP[heId] = p0;
    _BoundaryVertexPseudoN[heId] = pseudoN0;

    uint slot;
    InterlockedAdd(_OwnerBoundaryEdgeCounts[owner], 1, slot);
    if (slot < _MaxBoundaryEdgesPerOwner)
    {
        _OwnerBoundaryEdgeRefs[OwnerEdgeRefIndex(owner, slot)] = heId;
        InterlockedAdd(_CollisionDebugStats[STAT_BUILD_OWNER_EDGE_REF_WRITES], 1u);
    }
    else
    {
        _OwnerBoundaryOverflow[owner] = 1;
        InterlockedAdd(_CollisionDebugStats[STAT_OWNER_EDGE_OVERFLOW], 1u);
        InterlockedAdd(_CollisionDebugStats[STAT_BUILD_OWNER_EDGE_REF_OVERFLOW], 1u);
    }
}

[numthreads(64, 1, 1)]
void BinBoundaryEdges(uint3 tid : SV_DispatchThreadID)
{
    uint heId = tid.x;
    if (heId >= _DtHalfEdgeCount) return;

    uint owner = _BoundaryEdgeOwner[heId];
    if (owner == INVALID_U32) return;

    uint v0Gi = _BoundaryEdgeV0Gi[heId];
    uint v1Gi = _BoundaryEdgeV1Gi[heId];
    if (v0Gi == INVALID_U32 || v1Gi == INVALID_U32) return;

    float2 p0 = PredPos(v0Gi);
    float2 p1 = PredPos(v1Gi);

    float2 bmin = min(p0, p1) - _SdfBandWorld;
    float2 bmax = max(p0, p1) + _SdfBandWorld;

    float2 origin = _OwnerBinOrigin[owner];
    uint2 dim = _OwnerBinDim[owner];
    uint base = _OwnerBinBase[owner];
    float binSize = _OwnerGridTexel[owner] * _OwnerBinSizeScale;

    int2 c0 = clamp((int2)floor((bmin - origin) / binSize), 0, (int2)dim - 1);
    int2 c1 = clamp((int2)floor((bmax - origin) / binSize), 0, (int2)dim - 1);

    for (int y = c0.y; y <= c1.y; ++y)
    {
        for (int x = c0.x; x <= c1.x; ++x)
        {
            uint binIndex = base + (uint)y * dim.x + (uint)x;

            uint slot;
            InterlockedAdd(_EdgeBinCounts[binIndex], 1, slot);
            InterlockedMax(_CollisionDebugStats[STAT_MAX_EDGE_BIN_LOAD], slot + 1u);

            if (slot < _MaxEdgesPerBin)
                _EdgeBinRefs[EdgeBinRefIndex(binIndex, slot)] = heId;
            else
            {
                _EdgeBinOverflow[binIndex] = 1;
                InterlockedAdd(_CollisionDebugStats[STAT_EDGE_BIN_OVERFLOWS], 1u);
            }
        }
    }
}

[numthreads(64, 1, 1)]
void BinBoundaryVertices(uint3 tid : SV_DispatchThreadID)
{
    uint heId = tid.x;
    if (heId >= _DtHalfEdgeCount) return;

    uint owner = _BoundaryVertexOwner[heId];
    if (owner == INVALID_U32) return;

    uint vGi = _BoundaryVertexGi[heId];
    if (vGi == INVALID_U32) return;

    float2 p = PredPos(vGi);

    float2 bmin = p - _SdfBandWorld;
    float2 bmax = p + _SdfBandWorld;

    float2 origin = _OwnerBinOrigin[owner];
    uint2 dim = _OwnerBinDim[owner];
    uint base = _OwnerBinBase[owner];
    float binSize = _OwnerGridTexel[owner] * _OwnerBinSizeScale;

    int2 c0 = clamp((int2)floor((bmin - origin) / binSize), 0, (int2)dim - 1);
    int2 c1 = clamp((int2)floor((bmax - origin) / binSize), 0, (int2)dim - 1);

    for (int y = c0.y; y <= c1.y; ++y)
    {
        for (int x = c0.x; x <= c1.x; ++x)
        {
            uint binIndex = base + (uint)y * dim.x + (uint)x;

            uint slot;
            InterlockedAdd(_VertBinCounts[binIndex], 1, slot);
            InterlockedMax(_CollisionDebugStats[STAT_MAX_VERT_BIN_LOAD], slot + 1u);

            if (slot < _MaxVertsPerBin)
                _VertBinRefs[VertBinRefIndex(binIndex, slot)] = heId;
            else
            {
                _VertBinOverflow[binIndex] = 1;
                InterlockedAdd(_CollisionDebugStats[STAT_VERT_BIN_OVERFLOWS], 1u);
            }
        }
    }
}

[numthreads(8, 8, 1)]
void BuildOwnerFeatureField(uint3 tid : SV_DispatchThreadID)
{
    uint owner = tid.z;
    if (owner >= _OwnerCount) return;

    uint2 ij = tid.xy;
    uint2 dim = _OwnerGridDim[owner];
    if (ij.x >= dim.x || ij.y >= dim.y) return;

    uint gi = GridIndex(owner, ij);

    if (_OwnerBoundaryOverflow[owner] != 0u)
    {
        _SdfPhi[gi] = _SdfFar;
        _SdfGrad[gi] = 0.0;
        _SdfFeatType[gi] = INVALID_U32;
        _SdfFeatId[gi] = INVALID_U32;
        return;
    }

    float texel = _OwnerGridTexel[owner];
    float2 origin = _OwnerGridOrigin[owner];
    float2 x = origin + (float2(ij) + 0.5) * texel;

    FeatureHit hit;
    if (!QueryOwnerBoundaryExact(owner, x, hit))
    {
        _SdfPhi[gi] = _SdfFar;
        _SdfGrad[gi] = 0.0;
        _SdfFeatType[gi] = INVALID_U32;
        _SdfFeatId[gi] = INVALID_U32;
        return;
    }

    _SdfPhi[gi] = hit.phi;
    _SdfGrad[gi] = SafeNormalize(hit.grad);
    _SdfFeatType[gi] = hit.type;
    _SdfFeatId[gi] = hit.id;
}

[numthreads(64, 1, 1)]
void QueryVertexContacts(uint3 tid : SV_DispatchThreadID, uint groupIndex : SV_GroupIndex)
{
    if (groupIndex == 0u)
        gQueryVertexContactCount = 0u;
    GroupMemoryBarrierWithGroupSync();

    uint workPerPair = _MaxBoundaryEdgesPerOwner;
    if (workPerPair == 0u) return;

    uint workItem = tid.x;
    uint pairIndex = workItem / workPerPair;
    uint k = workItem - pairIndex * workPerPair;
    bool laneActive = (pairIndex < _QueryPairCount);
    InterlockedAdd(_CollisionDebugStats[STAT_VERTEX_WORK_ITEMS], 1u);
    if (!laneActive)
    {
        InterlockedAdd(_CollisionDebugStats[STAT_VERTEX_REJECT_PAIR_OOB], 1u);
    }

    uint ownerA = 0u;
    uint ownerB = 0u;
    if (laneActive)
    {
        uint2 pair = _OwnerPairs[pairIndex];
        ownerA = (_QuerySwap != 0u) ? pair.y : pair.x;
        ownerB = (_QuerySwap != 0u) ? pair.x : pair.y;
        laneActive = (ownerA != ownerB);
        if (!laneActive)
            InterlockedAdd(_CollisionDebugStats[STAT_VERTEX_REJECT_SAME_OWNER], 1u);
    }

    float support = _LayerKernelH * _CollisionSupportScale;
    if (laneActive && !OwnerPairAabbOverlap(ownerA, ownerB, support))
    {
        InterlockedAdd(_CollisionDebugStats[STAT_OWNER_AABB_REJECTS], 1u);
        InterlockedAdd(_CollisionDebugStats[STAT_VERTEX_REJECT_AABB], 1u);
        laneActive = false;
    }

    if (laneActive && (_OwnerBoundaryOverflow[ownerA] != 0u || _OwnerBoundaryOverflow[ownerB] != 0u))
    {
        InterlockedAdd(_CollisionDebugStats[STAT_VERTEX_REJECT_OWNER_OVERFLOW], 1u);
        laneActive = false;
    }

    uint heA = INVALID_U32;
    uint vGi = INVALID_U32;
    if (laneActive)
    {
        uint countA = min(_OwnerBoundaryEdgeCounts[ownerA], _MaxBoundaryEdgesPerOwner);
        if (k < countA)
        {
            heA = _OwnerBoundaryEdgeRefs[OwnerEdgeRefIndex(ownerA, k)];
            vGi = _BoundaryEdgeV0Gi[heA];
            laneActive = (vGi != INVALID_U32);
            if (!laneActive)
                InterlockedAdd(_CollisionDebugStats[STAT_VERTEX_REJECT_INVALID_VGI], 1u);
        }
        else
        {
            InterlockedAdd(_CollisionDebugStats[STAT_VERTEX_REJECT_NO_OWNER_EDGE_REF], 1u);
            laneActive = false;
        }
    }
    if (laneActive)
        InterlockedAdd(_CollisionDebugStats[STAT_VERTEX_CANDIDATES], 1u);

    float2 x = 0.0;
    if (laneActive)
        x = PredPos(vGi);

    FeatureHit hitB;
    if (laneActive)
    {
        laneActive = QueryOwnerBoundaryExactWide(ownerB, x, 2, hitB);
        if (!laneActive)
            InterlockedAdd(_CollisionDebugStats[STAT_VERTEX_REJECT_NO_BOUNDARY_HIT], 1u);
    }
    if (laneActive)
        InterlockedAdd(_CollisionDebugStats[STAT_VERTEX_FEATURE_HITS], 1u);

    float distToB = 0.0;
    if (laneActive)
        distToB = abs(hitB.phi);

    if (laneActive && distToB >= support)
    {
        InterlockedAdd(_CollisionDebugStats[STAT_VERTEX_REJECT_SUPPORT], 1u);
        laneActive = false;
    }
    if (laneActive)
        InterlockedAdd(_CollisionDebugStats[STAT_VERTEX_WITHIN_SUPPORT], 1u);

    float2 n = 0.0;
    float2 pB = 0.0;
    if (laneActive)
    {
        pB = hitB.cp;

        // Use geometric contact direction first; fall back to feature gradient
        // only if query point lands exactly on closest point.
        float2 nGeom = x - pB;
        float nGeomL2 = dot(nGeom, nGeom);
        if (nGeomL2 > 1e-20)
            n = nGeom * rsqrt(nGeomL2);
        else
        {
            float2 nFeat = FeatureNormalFromOwnerB(hitB.id, hitB.u);
            if (dot(nFeat, nFeat) > 1e-20)
                n = SafeNormalize(nFeat);
            else
                n = SafeNormalize(hitB.grad);
        }

        if (dot(n, x - pB) < 0.0)
            n = -n;

        if (dot(n, n) <= 1e-20)
        {
            InterlockedAdd(_CollisionDebugStats[STAT_VERTEX_REJECT_DEGENERATE_NORMAL], 1u);
            laneActive = false;
        }
    }

    float pen = 0.0;
    uint heB = INVALID_U32;
    if (laneActive)
    {
        pen = support - distToB;
        heB = hitB.id;
    }

    if (laneActive)
    {
        Contact c;
        c.ownerA = ownerA;
        c.ownerB = ownerB;
        c.vGi = vGi;
        c.heA = heA;
        c.heB = heB;
        c.n = n;
        c.pen = pen;
        c.pA = x;
        c.pB = pB;

        uint localIdx;
        InterlockedAdd(gQueryVertexContactCount, 1u, localIdx);
        if (localIdx < QUERY_VERTEX_CONTACT_CAP)
            gQueryVertexContacts[localIdx] = c;
        else
            InterlockedAdd(_CollisionDebugStats[STAT_VERTEX_STAGE_OVERFLOW], 1u);
    }

    GroupMemoryBarrierWithGroupSync();

    if (groupIndex == 0u)
    {
        uint flushCount = min(gQueryVertexContactCount, QUERY_VERTEX_CONTACT_CAP);

        // Compact valid contacts in-place before touching the global counter
        uint validCount = 0;
        for (uint i = 0u; i < flushCount; ++i)
        {
            Contact wc = gQueryVertexContacts[i];
            float nl2 = dot(wc.n, wc.n);
            if (nl2 <= 1e-20 || wc.pen <= 0.0)
            {
                InterlockedAdd(_CollisionDebugStats[STAT_VERTEX_COMPACT_REJECT_INVALID], 1u);
                continue;
            }
            wc.n *= rsqrt(nl2);
            gQueryVertexContacts[validCount++] = wc; // compact in-place
        }
        InterlockedAdd(_CollisionDebugStats[STAT_VERTEX_COMPACT_VALID], validCount);

        if (validCount > 0u)
        {
            uint base;
            InterlockedAdd(_ContactCount[0], validCount, base); // exact count now

            uint writable = (base < _MaxContacts)
                ? min(validCount, _MaxContacts - base) : 0u;

            for (uint i = 0u; i < writable; ++i)
            {
                _Contacts[base + i] = gQueryVertexContacts[i];
                InterlockedAdd(_CollisionDebugStats[STAT_VERTEX_CONTACTS_WRITTEN], 1u);
            }
        }
    }
}

[numthreads(64, 1, 1)]
void QueryEdgeEdgeContacts(uint3 tid : SV_DispatchThreadID)
{
    uint workPerPair = _MaxBoundaryEdgesPerOwner;
    if (workPerPair == 0u) return;

    uint workItem = tid.x;
    uint pairIndex = workItem / workPerPair;
    uint k = workItem - pairIndex * workPerPair;
    InterlockedAdd(_CollisionDebugStats[STAT_EDGE_WORK_ITEMS], 1u);
    if (pairIndex >= _QueryPairCount)
    {
        InterlockedAdd(_CollisionDebugStats[STAT_EDGE_REJECT_PAIR_OOB], 1u);
        return;
    }

    if (_QuerySwap != 0u)
    {
        InterlockedAdd(_CollisionDebugStats[STAT_EDGE_REJECT_SWAP], 1u);
        return;
    }

    uint2 pair = _OwnerPairs[pairIndex];
    uint ownerA = pair.x;
    uint ownerB = pair.y;
    if (ownerA == ownerB)
    {
        InterlockedAdd(_CollisionDebugStats[STAT_EDGE_REJECT_SAME_OWNER], 1u);
        return;
    }

    float support = _LayerKernelH * _CollisionSupportScale;
    if (!OwnerPairAabbOverlap(ownerA, ownerB, support))
    {
        InterlockedAdd(_CollisionDebugStats[STAT_OWNER_AABB_REJECTS], 1u);
        InterlockedAdd(_CollisionDebugStats[STAT_EDGE_REJECT_AABB], 1u);
        return;
    }

    if (_OwnerBoundaryOverflow[ownerA] != 0u || _OwnerBoundaryOverflow[ownerB] != 0u)
    {
        InterlockedAdd(_CollisionDebugStats[STAT_EDGE_REJECT_OWNER_OVERFLOW], 1u);
        return;
    }

    uint countA = min(_OwnerBoundaryEdgeCounts[ownerA], _MaxBoundaryEdgesPerOwner);
    if (k >= countA)
    {
        InterlockedAdd(_CollisionDebugStats[STAT_EDGE_REJECT_NO_OWNER_EDGE_REF], 1u);
        return;
    }

    uint heA = _OwnerBoundaryEdgeRefs[OwnerEdgeRefIndex(ownerA, k)];
    uint aV0Gi = _BoundaryEdgeV0Gi[heA];
    uint aV1Gi = _BoundaryEdgeV1Gi[heA];
    if (aV0Gi == INVALID_U32 || aV1Gi == INVALID_U32)
    {
        InterlockedAdd(_CollisionDebugStats[STAT_EDGE_REJECT_INVALID_EDGE_A], 1u);
        return;
    }

    float2 a0 = PredPos(aV0Gi);
    float2 a1 = PredPos(aV1Gi);

    float2 amin = min(a0, a1) - support;
    float2 amax = max(a0, a1) + support;

    float2 originB = _OwnerBinOrigin[ownerB];
    uint2 dimB = _OwnerBinDim[ownerB];
    uint baseB = _OwnerBinBase[ownerB];
    float binSizeB = _OwnerGridTexel[ownerB] * _OwnerBinSizeScale;

    int2 c0 = clamp((int2)floor((amin - originB) / binSizeB), 0, (int2)dimB - 1);
    int2 c1 = clamp((int2)floor((amax - originB) / binSizeB), 0, (int2)dimB - 1);

    for (int y = c0.y; y <= c1.y; ++y)
    {
        for (int x = c0.x; x <= c1.x; ++x)
        {
            uint binIndex = baseB + (uint)y * dimB.x + (uint)x;
            InterlockedAdd(_CollisionDebugStats[STAT_EDGE_BIN_CELLS_VISITED], 1u);
            uint ec = min(_EdgeBinCounts[binIndex], _MaxEdgesPerBin);
            InterlockedAdd(_CollisionDebugStats[STAT_EDGE_BIN_EDGE_REFS_SCANNED], ec);

            for (uint i = 0u; i < ec; ++i)
            {
                uint heB = _EdgeBinRefs[EdgeBinRefIndex(binIndex, i)];
                if (_BoundaryEdgeOwner[heB] != ownerB)
                {
                    InterlockedAdd(_CollisionDebugStats[STAT_EDGE_REJECT_OWNER_MISMATCH_B], 1u);
                    continue;
                }
                InterlockedAdd(_CollisionDebugStats[STAT_EDGE_PAIR_CANDIDATES], 1u);

                uint bV0Gi = _BoundaryEdgeV0Gi[heB];
                uint bV1Gi = _BoundaryEdgeV1Gi[heB];
                if (bV0Gi == INVALID_U32 || bV1Gi == INVALID_U32)
                {
                    InterlockedAdd(_CollisionDebugStats[STAT_EDGE_REJECT_INVALID_EDGE_B], 1u);
                    continue;
                }

                float2 b0 = PredPos(bV0Gi);
                float2 b1 = PredPos(bV1Gi);

                float2 bmin = min(b0, b1) - support;
                float2 bmax = max(b0, b1) + support;
                if (!BBoxOverlap(amin, amax, bmin, bmax))
                {
                    InterlockedAdd(_CollisionDebugStats[STAT_EDGE_REJECT_EDGE_BBOX], 1u);
                    continue;
                }

                float sI, tI;
                float2 xI;
                uint hitKind;
                bool intersects = SegmentIntersection2D(a0, a1, b0, b1, sI, tI, xI, hitKind);
                if (!intersects)
                {
                    InterlockedAdd(_CollisionDebugStats[STAT_EDGE_REJECT_NO_INTERSECTION], 1u);
                    continue;
                }
                if (hitKind != SEG_HIT_POINT)
                {
                    InterlockedAdd(_CollisionDebugStats[STAT_EDGE_REJECT_NON_POINT_HIT], 1u);
                    continue;
                }
                InterlockedAdd(_CollisionDebugStats[STAT_EDGE_POINT_INTERSECTIONS], 1u);
                if (!IsInteriorParam(sI) || !IsInteriorParam(tI))
                {
                    InterlockedAdd(_CollisionDebugStats[STAT_EDGE_REJECT_ENDPOINT_INTERSECTION], 1u);
                    continue;
                }
                if (!IsCanonicalBinForPoint(ownerB, binIndex, xI))
                {
                    InterlockedAdd(_CollisionDebugStats[STAT_EDGE_REJECT_NON_CANONICAL_BIN], 1u);
                    continue;
                }

                InterlockedAdd(_CollisionDebugStats[STAT_EDGE_WITHIN_SUPPORT], 1u);

                float2 n = FeatureNormalFromOwnerB(heB, tI);
                float nl2 = dot(n, n);
                if (nl2 <= 1e-20)
                {
                    InterlockedAdd(_CollisionDebugStats[STAT_EDGE_REJECT_DEGENERATE_NORMAL], 1u);
                    continue;
                }
                n *= rsqrt(nl2);

                float2 midA = 0.5 * (a0 + a1);
                float2 midB = 0.5 * (b0 + b1);
                if (dot(n, midA - midB) < 0.0)
                    n = -n;

                float sep = abs(dot(midA - midB, n));
                float pen = support - sep;
                if (pen <= 0.0)
                {
                    InterlockedAdd(_CollisionDebugStats[STAT_EDGE_REJECT_NO_PENETRATION], 1u);
                    continue;
                }

                EmitContact(ownerA, ownerB, INVALID_U32, heA, heB, n, pen, xI, xI);
                InterlockedAdd(_CollisionDebugStats[STAT_EDGE_CONTACTS_EMITTED], 1u);
            }
        }
    }
}
