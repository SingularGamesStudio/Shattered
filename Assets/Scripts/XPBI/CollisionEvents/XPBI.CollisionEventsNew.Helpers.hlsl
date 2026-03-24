float2 PerpLeft(float2 v)
{
    return float2(-v.y, v.x);
}

float Cross2(float2 a, float2 b)
{
    return a.x * b.y - a.y * b.x;
}

float SafeRsqrt(float x)
{
    return rsqrt(max(x, 1e-20));
}

float2 SafeNormalize(float2 v)
{
    return v * SafeRsqrt(dot(v, v));
}

float2 PredPos(uint gi)
{
    return _Pos[gi] + _Vel[gi] * _Dt;
}

float2 BoundaryOutwardNormal(uint outwardIsRight, float2 p0, float2 p1)
{
    float2 d = p1 - p0;
    float2 nL = SafeNormalize(PerpLeft(d));
    return (outwardIsRight != 0u) ? -nL : nL;
}

uint OwnerEdgeRefIndex(uint owner, uint i)
{
    return owner * _MaxBoundaryEdgesPerOwner + i;
}

uint GridIndex(uint owner, uint2 ij)
{
    uint2 dim = _OwnerGridDim[owner];
    return _OwnerGridBase[owner] + ij.y * dim.x + ij.x;
}

uint BinIndex(uint owner, uint2 ij)
{
    uint2 dim = _OwnerBinDim[owner];
    return _OwnerBinBase[owner] + ij.y * dim.x + ij.x;
}

uint EdgeBinRefIndex(uint binIndex, uint slot)
{
    return binIndex * _MaxEdgesPerBin + slot;
}

uint VertBinRefIndex(uint binIndex, uint slot)
{
    return binIndex * _MaxVertsPerBin + slot;
}

void ClearFeatureHit(out FeatureHit h)
{
    h.phi = _SdfFar;
    h.grad = 0.0;
    h.cp = 0.0;
    h.type = INVALID_U32;
    h.id = INVALID_U32;
    h.u = 0.0;
    h.valid = 0u;
}

bool FeatureLess(FeatureHit a, FeatureHit b)
{
    if (a.valid == 0u) return false;
    if (b.valid == 0u) return true;

    float aa = abs(a.phi);
    float ba = abs(b.phi);

    if (aa + 1e-6 < ba) return true;
    if (ba + 1e-6 < aa) return false;

    if (a.type != b.type) return a.type > b.type;
    return a.id < b.id;
}

void ConsiderEdgeFeature(float2 x, uint heId, inout FeatureHit best)
{
    uint v0Gi = _BoundaryEdgeV0Gi[heId];
    uint v1Gi = _BoundaryEdgeV1Gi[heId];
    if (v0Gi == INVALID_U32 || v1Gi == INVALID_U32) return;

    float2 a = PredPos(v0Gi);
    float2 b = PredPos(v1Gi);
    float2 ab = b - a;

    float denom = dot(ab, ab);
    float t = (denom > 1e-20) ? saturate(dot(x - a, ab) / denom) : 0.0;
    if (t <= 1e-4 || t >= 1.0 - 1e-4) return;

    float2 cp = a + t * ab;
    float2 d = x - cp;
    float dsq = dot(d, d);
    float2 n = BoundaryOutwardNormal(_BoundaryOutwardIsRight[heId], a, b);

    FeatureHit h;
    h.cp = cp;
    h.type = 0u;
    h.id = heId;
    h.u = t;
    h.valid = 1u;

    if (dsq <= _SdfEps)
    {
        h.phi = 0.0;
        h.grad = n;
    }
    else
    {
        float dist = sqrt(dsq);
        float s = (dot(n, d) >= 0.0) ? 1.0 : -1.0;
        h.phi = s * dist;
        h.grad = s * d / dist;
    }

    if (FeatureLess(h, best)) best = h;
}

void ConsiderVertexFeature(float2 x, uint heId, inout FeatureHit best)
{
    uint vGi = _BoundaryVertexGi[heId];
    if (vGi == INVALID_U32) return;

    float2 v = PredPos(vGi);
    float2 np = _BoundaryVertexPseudoN[heId];
    float2 d = x - v;
    float dsq = dot(d, d);

    FeatureHit h;
    h.cp = v;
    h.type = 1u;
    h.id = heId;
    h.u = 0.0;
    h.valid = 1u;

    if (dsq <= _SdfEps)
    {
        h.phi = 0.0;
        h.grad = np;
    }
    else
    {
        float dist = sqrt(dsq);
        float s = (dot(np, d) >= 0.0) ? 1.0 : -1.0;
        h.phi = s * dist;
        h.grad = s * d / dist;
    }

    if (FeatureLess(h, best)) best = h;
}

bool QueryOwnerBoundaryExactWide(uint owner, float2 x, int radius, out FeatureHit best)
{
    ClearFeatureHit(best);
    if (_OwnerBoundaryOverflow[owner] != 0u) return false;

    float texel = _OwnerGridTexel[owner];
    float binSize = texel * _OwnerBinSizeScale;
    float2 binOrigin = _OwnerBinOrigin[owner];
    uint2 binDim = _OwnerBinDim[owner];

    int2 bc = clamp((int2)floor((x - binOrigin) / binSize), 0, (int2)binDim - 1);

    [loop]
    for (int oy = -radius; oy <= radius; ++oy)
    {
        [loop]
        for (int ox = -radius; ox <= radius; ++ox)
        {
            int2 cc = bc + int2(ox, oy);
            if (cc.x < 0 || cc.y < 0 || cc.x >= (int)binDim.x || cc.y >= (int)binDim.y) continue;

            uint binIndex = BinIndex(owner, uint2(cc));

            uint ec = min(_EdgeBinCounts[binIndex], _MaxEdgesPerBin);
            [loop]
            for (uint i = 0; i < ec; ++i)
                ConsiderEdgeFeature(x, _EdgeBinRefs[EdgeBinRefIndex(binIndex, i)], best);

            uint vc = min(_VertBinCounts[binIndex], _MaxVertsPerBin);
            [loop]
            for (uint i = 0; i < vc; ++i)
                ConsiderVertexFeature(x, _VertBinRefs[VertBinRefIndex(binIndex, i)], best);
        }
    }

    // If bins are stale relative to current predicted positions, recover by
    // scanning this owner's boundary-feature list directly.
    if (best.valid == 0u)
    {
        uint count = min(_OwnerBoundaryEdgeCounts[owner], _MaxBoundaryEdgesPerOwner);
        [loop]
        for (uint i = 0; i < count; ++i)
        {
            uint heId = _OwnerBoundaryEdgeRefs[OwnerEdgeRefIndex(owner, i)];
            ConsiderEdgeFeature(x, heId, best);
            ConsiderVertexFeature(x, heId, best);
        }
    }

    return best.valid != 0u;
}

bool QueryOwnerBoundaryExact(uint owner, float2 x, out FeatureHit best)
{
    if (!QueryOwnerBoundaryExactWide(owner, x, 1, best)) return false;
    return abs(best.phi) <= _SdfBandWorld;
}

void EmitContact(uint ownerA, uint ownerB, uint vGi, uint heA, uint heB, float2 n, float pen, float2 pA, float2 pB)
{
    if (pen <= 0.0) return;

    float nl2 = dot(n, n);
    if (nl2 <= 1e-20) return;

    uint dst;
    InterlockedAdd(_ContactCount[0], 1, dst);
    if (dst >= _MaxContacts) return;

    Contact c;
    c.ownerA = ownerA;
    c.ownerB = ownerB;
    c.vGi = vGi;
    c.heA = heA;
    c.heB = heB;
    c.n = n * rsqrt(nl2);
    c.pen = pen;
    c.pA = pA;
    c.pB = pB;
    _Contacts[dst] = c;
}

bool BBoxOverlap(float2 amin, float2 amax, float2 bmin, float2 bmax)
{
    return (amax.x >= bmin.x) && (amax.y >= bmin.y) && (bmax.x >= amin.x) && (bmax.y >= amin.y);
}

bool SegmentIntersection2D(
    float2 p, float2 p2,
    float2 q, float2 q2,
    out float sOut, out float tOut, out float2 xOut,
    out uint hitKind)
{
    hitKind = SEG_HIT_NONE;

    float2 r = p2 - p;
    float2 s = q2 - q;
    float rxs = Cross2(r, s);
    float2 qp = q - p;
    float qpxr = Cross2(qp, r);

    if (abs(rxs) <= 1e-20 && abs(qpxr) <= 1e-20)
    {
        float rr = dot(r, r);
        float ss = dot(s, s);

        if (rr <= 1e-20 && ss <= 1e-20)
        {
            sOut = 0.0;
            tOut = 0.0;
            xOut = p;
            if (dot(p - q, p - q) <= _SdfEps)
            {
                hitKind = SEG_HIT_COLLINEAR;
                return true;
            }
            return false;
        }

        if (rr <= 1e-20)
        {
            float tq = (ss > 1e-20) ? saturate(dot(p - q, s) / ss) : 0.0;
            float2 cq = q + tq * s;
            if (dot(cq - p, cq - p) <= _SdfEps)
            {
                sOut = 0.0;
                tOut = tq;
                xOut = p;
                hitKind = SEG_HIT_COLLINEAR;
                return true;
            }
            return false;
        }

        float t0 = dot(q - p, r) / rr;
        float t1 = dot(q2 - p, r) / rr;
        float lo = max(0.0, min(t0, t1));
        float hi = min(1.0, max(t0, t1));
        if (lo <= hi)
        {
            sOut = lo;
            xOut = p + lo * r;
            tOut = (ss > 1e-20) ? saturate(dot(xOut - q, s) / ss) : 0.0;
            hitKind = SEG_HIT_COLLINEAR;
            return true;
        }
        return false;
    }

    if (abs(rxs) <= 1e-20) return false;

    float t = Cross2(qp, s) / rxs;
    float u = Cross2(qp, r) / rxs;

    if (t >= 0.0 && t <= 1.0 && u >= 0.0 && u <= 1.0)
    {
        sOut = t;
        tOut = u;
        xOut = p + t * r;
        hitKind = SEG_HIT_POINT;
        return true;
    }

    return false;
}

bool IsInteriorParam(float t)
{
    return t > 1e-4 && t < 1.0 - 1e-4;
}

void ClosestPtSegmentSegment2D(float2 p1, float2 q1, float2 p2, float2 q2, out float s, out float t, out float2 c1, out float2 c2)
{
    float2 d1 = q1 - p1;
    float2 d2 = q2 - p2;
    float2 r = p1 - p2;
    float a = dot(d1, d1);
    float e = dot(d2, d2);
    float f = dot(d2, r);

    if (a <= 1e-20 && e <= 1e-20)
    {
        s = 0.0;
        t = 0.0;
        c1 = p1;
        c2 = p2;
        return;
    }

    if (a <= 1e-20)
    {
        s = 0.0;
        t = (e > 1e-20) ? saturate(f / e) : 0.0;
    }
    else
    {
        float c = dot(d1, r);

        if (e <= 1e-20)
        {
            t = 0.0;
            s = saturate(-c / a);
        }
        else
        {
            float b = dot(d1, d2);
            float denom = a * e - b * b;

            if (denom > 1e-20) s = saturate((b * f - c * e) / denom);
            else s = 0.0;

            float tnom = b * s + f;

            if (tnom < 0.0)
            {
                t = 0.0;
                s = saturate(-c / a);
            }
            else if (tnom > e)
            {
                t = 1.0;
                s = saturate((b - c) / a);
            }
            else
            {
                t = tnom / e;
            }
        }
    }

    c1 = p1 + d1 * s;
    c2 = p2 + d2 * t;
}

float2 FeatureNormalFromOwnerB(uint heId, float tB)
{
    if (tB > 1e-4 && tB < 1.0 - 1e-4)
    {
        uint v0Gi = _BoundaryEdgeV0Gi[heId];
        uint v1Gi = _BoundaryEdgeV1Gi[heId];
        if (v0Gi != INVALID_U32 && v1Gi != INVALID_U32)
            return BoundaryOutwardNormal(_BoundaryOutwardIsRight[heId], PredPos(v0Gi), PredPos(v1Gi));
    }

    if (tB <= 0.5)
        return _BoundaryEdgePseudoN0[heId];

    return _BoundaryEdgePseudoN1[heId];
}

bool IsCanonicalBinForPoint(uint ownerB, uint binIndex, float2 p)
{
    float binSize = _OwnerGridTexel[ownerB] * _OwnerBinSizeScale;
    float2 origin = _OwnerBinOrigin[ownerB];
    uint2 dim = _OwnerBinDim[ownerB];
    uint base = _OwnerBinBase[ownerB];

    int2 c = clamp((int2)floor((p - origin) / binSize), 0, (int2)dim - 1);
    uint pointBin = base + (uint)c.y * dim.x + (uint)c.x;

    return binIndex == pointBin;
}

bool OwnerPairAabbOverlap(uint ownerA, uint ownerB, float inflate)
{
    float2 oA = _OwnerGridOrigin[ownerA];
    float2 oB = _OwnerGridOrigin[ownerB];

    float2 eA = float2(_OwnerGridDim[ownerA]) * _OwnerGridTexel[ownerA];
    float2 eB = float2(_OwnerGridDim[ownerB]) * _OwnerGridTexel[ownerB];

    float2 aMin = oA - inflate;
    float2 aMax = oA + eA + inflate;
    float2 bMin = oB - inflate;
    float2 bMax = oB + eB + inflate;

    return BBoxOverlap(aMin, aMax, bMin, bMax);
}
