[numthreads(64, 1, 1)]
void ClearState(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;

    if (i < _OwnerCount)
    {
        _OwnerBoundaryEdgeCounts[i] = 0;
        _OwnerBoundaryOverflow[i] = 0;
    }

    if (_OwnerCount > 0u)
    {
        uint totalBins = _OwnerBinBase[_OwnerCount - 1] +
                         _OwnerBinDim[_OwnerCount - 1].x * _OwnerBinDim[_OwnerCount - 1].y;

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

    if (i < _MaxFineNodeContacts)
    {
        FineNodeContact c;
        FineNodeContactInit(c);
        _FineNodeContacts[i] = c;
    }

    if (i == 0u)
        _FineNodeContactCount[0] = 0u;
}

[numthreads(64, 1, 1)]
void BuildBoundaryFeatures(uint3 tid : SV_DispatchThreadID)
{
    uint heId = tid.x;
    if (heId >= _DtHalfEdgeCount) return;
    if (_DtBoundaryEdgeFlags[heId] == 0u)
    {
        return;
    }

    DT_HalfEdge he = _DtHalfEdges[heId];
    if (he.t < 0 || he.next < 0 || (uint)he.next >= _DtHalfEdgeCount)
    {
        return;
    }

    int tw = he.twin;
    bool twValid = (tw >= 0) && ((uint)tw < _DtHalfEdgeCount);
    int twFace = twValid ? _DtHalfEdges[(uint)tw].t : -1;

    bool heInternal = ((uint)he.t < _DtTriCount) ? (_DtTriInternal[he.t] != 0u) : false;
    bool twInternal = (twFace >= 0 && (uint)twFace < _DtTriCount) ? (_DtTriInternal[twFace] != 0u) : false;

    if (!(heInternal && !twInternal))
    {
        return;
    }

    DT_HalfEdge heNext = _DtHalfEdges[(uint)he.next];
    if (heNext.next < 0 || (uint)heNext.next >= _DtHalfEdgeCount)
    {
        return;
    }
    DT_HalfEdge heNextNext = _DtHalfEdges[(uint)heNext.next];

    if (he.v < 0 || heNext.v < 0 || heNextNext.v < 0)
    {
        return;
    }

    if ((uint)he.v >= _DtLocalVertexCount || (uint)heNext.v >= _DtLocalVertexCount || (uint)heNextNext.v >= _DtLocalVertexCount)
    {
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
        return;
    }

    if (v0Gi == v1Gi)
    {
        return;
    }

    int ownerI = _DtCollisionOwnerByLocal[v0Local];
    if (ownerI < 0)
    {
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
    }
    else
    {
        _OwnerBoundaryOverflow[owner] = 1;
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

            if (slot < _MaxEdgesPerBin)
                _EdgeBinRefs[EdgeBinRefIndex(binIndex, slot)] = heId;
            else
            {
                _EdgeBinOverflow[binIndex] = 1;
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

            if (slot < _MaxVertsPerBin)
                _VertBinRefs[VertBinRefIndex(binIndex, slot)] = heId;
            else
            {
                _VertBinOverflow[binIndex] = 1;
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

bool SampleOwnerFieldLinear(uint owner, float2 x, out float phi, out float2 grad)
{
    uint2 dim = _OwnerGridDim[owner];
    if (dim.x == 0u || dim.y == 0u)
    {
        phi = _SdfFar;
        grad = 0.0;
        return false;
    }

    float texel = _OwnerGridTexel[owner];
    float2 origin = _OwnerGridOrigin[owner];
    float2 g = (x - origin) / texel - 0.5;

    float2 maxG = float2((float)(dim.x - 1u), (float)(dim.y - 1u));
    if (g.x < 0.0 || g.y < 0.0 || g.x > maxG.x || g.y > maxG.y)
    {
        phi = _SdfFar;
        grad = 0.0;
        return false;
    }

    int2 i0 = (int2)floor(g);
    int2 i1 = min(i0 + 1, (int2)dim - 1);
    float2 f = frac(g);

    uint idx00 = GridIndex(owner, (uint2)i0);
    uint idx10 = GridIndex(owner, uint2((uint)i1.x, (uint)i0.y));
    uint idx01 = GridIndex(owner, uint2((uint)i0.x, (uint)i1.y));
    uint idx11 = GridIndex(owner, (uint2)i1);

    float phi00 = _SdfPhi[idx00];
    float phi10 = _SdfPhi[idx10];
    float phi01 = _SdfPhi[idx01];
    float phi11 = _SdfPhi[idx11];

    float2 g00 = _SdfGrad[idx00];
    float2 g10 = _SdfGrad[idx10];
    float2 g01 = _SdfGrad[idx01];
    float2 g11 = _SdfGrad[idx11];

    float phi0 = lerp(phi00, phi10, f.x);
    float phi1 = lerp(phi01, phi11, f.x);
    phi = lerp(phi0, phi1, f.y);

    float2 grad0 = lerp(g00, g10, f.x);
    float2 grad1 = lerp(g01, g11, f.x);
    grad = lerp(grad0, grad1, f.y);

    float gl2 = dot(grad, grad);
    if (gl2 > 1e-20)
        grad *= rsqrt(gl2);
    else
        grad = 0.0;

    return true;
}

[numthreads(64, 1, 1)]
void QueryNodeSurfaceContacts(uint3 tid : SV_DispatchThreadID)
{
    uint workPerPair = _MaxBoundaryEdgesPerOwner;
    if (workPerPair == 0u) return;

    uint workItem  = tid.x;
    uint pairIndex = workItem / workPerPair;
    uint k         = workItem - pairIndex * workPerPair;
    if (pairIndex >= _QueryPairCount) return;

    uint2 pair = _OwnerPairs[pairIndex];
    uint ownerA = (_QuerySwap != 0u) ? pair.y : pair.x;
    uint ownerB = (_QuerySwap != 0u) ? pair.x : pair.y;
    if (ownerA == ownerB) return;

    float support = _LayerKernelH * _CollisionSupportScale;
    if (!OwnerPairAabbOverlap(ownerA, ownerB, support)) return;
    if (_OwnerBoundaryOverflow[ownerA] != 0u) return;
    if (_OwnerBoundaryOverflow[ownerB] != 0u) return;

    uint countA = min(_OwnerBoundaryEdgeCounts[ownerA], _MaxBoundaryEdgesPerOwner);
    if (k >= countA) return;

    uint heA = _OwnerBoundaryEdgeRefs[OwnerEdgeRefIndex(ownerA, k)];
    uint vGi = _BoundaryEdgeV0Gi[heA];
    if (vGi == INVALID_U32) return;

    float2 x = PredPos(vGi);

    float phiField;
    float2 nField;
    if (!SampleOwnerFieldLinear(ownerB, x, phiField, nField)) return;
    if (phiField >= support) return;

    FeatureHit hit;
    if (!QueryOwnerBoundaryExact(ownerB, x, hit)) return;
    if (hit.valid == 0u) return;
    if (hit.phi >= support) return;

    FineNodeContact c;
    if (!BuildVertexSurfaceNodeContact(ownerA, ownerB, vGi, hit, support, c))
        return;

    AppendFineNodeContact(c);
}
[numthreads(64, 1, 1)]
void QueryEdgeEdgeNodeContacts(uint3 tid : SV_DispatchThreadID)
{
    uint workPerPair = _MaxBoundaryEdgesPerOwner;
    if (workPerPair == 0u) return;

    uint workItem  = tid.x;
    uint pairIndex = workItem / workPerPair;
    uint k         = workItem - pairIndex * workPerPair;
    if (pairIndex >= _QueryPairCount) return;

    uint2 pair = _OwnerPairs[pairIndex];
    uint ownerA = pair.x;
    uint ownerB = pair.y;
    if (ownerA == ownerB) return;

    float support = _LayerKernelH * _CollisionSupportScale;
    if (!OwnerPairAabbOverlap(ownerA, ownerB, support)) return;
    if (_OwnerBoundaryOverflow[ownerA] != 0u || _OwnerBoundaryOverflow[ownerB] != 0u) return;

    uint countA = min(_OwnerBoundaryEdgeCounts[ownerA], _MaxBoundaryEdgesPerOwner);
    if (k >= countA) return;

    uint heA = _OwnerBoundaryEdgeRefs[OwnerEdgeRefIndex(ownerA, k)];
    uint a0Gi = _BoundaryEdgeV0Gi[heA];
    uint a1Gi = _BoundaryEdgeV1Gi[heA];
    if (a0Gi == INVALID_U32 || a1Gi == INVALID_U32) return;

    float2 a0 = PredPos(a0Gi);
    float2 a1 = PredPos(a1Gi);

    float2 amin = min(a0, a1) - support;
    float2 amax = max(a0, a1) + support;

    float2 originB  = _OwnerBinOrigin[ownerB];
    uint2  dimB     = _OwnerBinDim[ownerB];
    uint   baseB    = _OwnerBinBase[ownerB];
    float  binSizeB = _OwnerGridTexel[ownerB] * _OwnerBinSizeScale;

    int2 c0 = clamp((int2)floor((amin - originB) / binSizeB), 0, (int2)dimB - 1);
    int2 c1 = clamp((int2)floor((amax - originB) / binSizeB), 0, (int2)dimB - 1);

    for (int y = c0.y; y <= c1.y; ++y)
    {
        for (int x = c0.x; x <= c1.x; ++x)
        {
            uint binIndex = baseB + (uint)y * dimB.x + (uint)x;
            uint ec = min(_EdgeBinCounts[binIndex], _MaxEdgesPerBin);

            for (uint i = 0u; i < ec; ++i)
            {
                uint heB = _EdgeBinRefs[EdgeBinRefIndex(binIndex, i)];
                if (_BoundaryEdgeOwner[heB] != ownerB) continue;

                uint b0Gi = _BoundaryEdgeV0Gi[heB];
                uint b1Gi = _BoundaryEdgeV1Gi[heB];
                if (b0Gi == INVALID_U32 || b1Gi == INVALID_U32) continue;

                float2 b0 = PredPos(b0Gi);
                float2 b1 = PredPos(b1Gi);

                float2 bmin = min(b0, b1) - support;
                float2 bmax = max(b0, b1) + support;
                if (!BBoxOverlap(amin, amax, bmin, bmax)) continue;

                float sI, tI;
                float2 xI;
                uint hitKind;
                bool intersects = SegmentIntersection2D(a0, a1, b0, b1, sI, tI, xI, hitKind);
                if (!intersects) continue;
                if (hitKind != SEG_HIT_POINT) continue;
                if (!IsInteriorParam(sI) || !IsInteriorParam(tI)) continue;
                if (!IsCanonicalBinForPoint(ownerB, binIndex, xI)) continue;

                float2 nAB = FeatureNormalFromOwnerB(heB, tI);
                float nl2 = dot(nAB, nAB);
                if (nl2 <= 1e-20) continue;
                nAB *= rsqrt(nl2);

                float2 midA = 0.5 * (a0 + a1);
                float2 midB = 0.5 * (b0 + b1);
                if (dot(nAB, midA - midB) < 0.0) nAB = -nAB;

                float sep = abs(dot(midA - midB, nAB));
                float pen = support - sep;
                if (pen <= 0.0) continue;

                EmitEdgeEdgeNodeContacts(
                    ownerA, ownerB,
                    a0Gi, a1Gi,
                    b0Gi, b1Gi,
                    saturate(sI), saturate(tI),
                    nAB, pen, xI);
            }
        }
    }
}


#define FINE_NODE_MANIFOLD_CAP 2

RWBuffer<uint> _FineNodeManifoldCount;      // size = _ActiveCount
RWBuffer<uint> _FineNodeManifoldOverflow;   // size = _ActiveCount
RWStructuredBuffer<FineNodeContact> _FineNodeManifold; // size = _ActiveCount * FINE_NODE_MANIFOLD_CAP

uint LocalIndexFromGlobal(uint gi)
{
    if (_UseDtGlobalNodeMap != 0u)
    {
        int li = _DtGlobalToLayerLocalMap[gi];
        return (li >= 0) ? (uint)li : INVALID_U32;
    }

    int liUnmapped = (int)gi - (int)_DtLocalBase;
    return (liUnmapped >= 0) ? (uint)liUnmapped : INVALID_U32;
}

[numthreads(64,1,1)]
void ClearFineNodeManifolds(uint3 tid : SV_DispatchThreadID)
{
    uint i = tid.x;
    if (i < _ActiveCount)
    {
        _FineNodeManifoldCount[i] = 0u;
        _FineNodeManifoldOverflow[i] = 0u;
    }

    uint total = _ActiveCount * FINE_NODE_MANIFOLD_CAP;
    if (i < total)
    {
        FineNodeContact c;
        FineNodeContactInit(c);
        _FineNodeManifold[i] = c;
    }
}

[numthreads(64,1,1)]
void ScatterFineNodeContactsToManifolds(uint3 tid : SV_DispatchThreadID)
{
    uint idx = tid.x;
    uint count = min(_FineNodeContactCount[0], _MaxFineNodeContacts);
    if (idx >= count) return;

    FineNodeContact c = _FineNodeContacts[idx];
    uint li = LocalIndexFromGlobal(c.slaveGi);
    if (li == INVALID_U32 || li >= _ActiveCount) return;

    uint slot;
    InterlockedAdd(_FineNodeManifoldCount[li], 1u, slot);
    if (slot >= FINE_NODE_MANIFOLD_CAP)
    {
        _FineNodeManifoldOverflow[li] = 1u;
        return;
    }

    _FineNodeManifold[li * FINE_NODE_MANIFOLD_CAP + slot] = c;
}