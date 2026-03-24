static const uint INVALID_U32 = 0xFFFFFFFFu;
static const uint SEG_HIT_NONE = 0u;
static const uint SEG_HIT_POINT = 1u;
static const uint SEG_HIT_COLLINEAR = 2u;

struct DT_HalfEdge
{
    int v;
    int next;
    int twin;
    int t;
};

struct Contact
{
    uint ownerA;
    uint ownerB;
    uint vGi;
    uint heA;
    uint heB;
    float2 n;
    float pen;
    float2 pA;
    float2 pB;
};

struct FeatureHit
{
    float phi;
    float2 grad;
    float2 cp;
    uint type; // 0=edge, 1=vertex
    uint id;   // heId for both edge and start-vertex
    float u;   // edge segment parameter
    uint valid;
};

cbuffer Params
{
    uint _OwnerCount;
    uint _DtHalfEdgeCount;
    uint _DtTriCount;
    uint _DtLocalVertexCount;
    uint _MaxBoundaryEdgesPerOwner;
    uint _MaxEdgesPerBin;
    uint _MaxVertsPerBin;
    uint _MaxContacts;
    uint _MaxGridDimX;
    uint _MaxGridDimY;
    uint _QueryPairCount;
    uint _QuerySwap;

    float _Dt;
    float _LayerKernelH;
    float _CollisionSupportScale;
    float _SdfBandWorld;
    float _SdfFar;
    float _SdfEps;
    float _OwnerBinSizeScale;
}

RWStructuredBuffer<float2> _Pos;
RWStructuredBuffer<float2> _Vel;

StructuredBuffer<int> _DtCollisionOwnerByLocal;
StructuredBuffer<DT_HalfEdge> _DtHalfEdges;
StructuredBuffer<uint> _DtBoundaryEdgeFlags;
StructuredBuffer<uint> _DtGlobalVertexByLocal;
StructuredBuffer<uint> _DtTriInternal;
StructuredBuffer<float2> _DtBoundaryNormals;

StructuredBuffer<float2> _OwnerGridOrigin;
StructuredBuffer<uint2> _OwnerGridDim;
StructuredBuffer<float> _OwnerGridTexel;
StructuredBuffer<uint> _OwnerGridBase;

StructuredBuffer<float2> _OwnerBinOrigin;
StructuredBuffer<uint2> _OwnerBinDim;
StructuredBuffer<uint> _OwnerBinBase;

StructuredBuffer<uint2> _OwnerPairs;

RWStructuredBuffer<uint> _OwnerBoundaryEdgeCounts;
RWStructuredBuffer<uint> _OwnerBoundaryOverflow;
RWStructuredBuffer<uint> _OwnerBoundaryEdgeRefs;

RWStructuredBuffer<uint> _BoundaryEdgeOwner;
RWStructuredBuffer<uint> _BoundaryEdgeV0Gi;
RWStructuredBuffer<uint> _BoundaryEdgeV1Gi;
RWStructuredBuffer<uint> _BoundaryOutwardIsRight;
RWStructuredBuffer<float2> _BoundaryEdgeP0;
RWStructuredBuffer<float2> _BoundaryEdgeP1;
RWStructuredBuffer<float2> _BoundaryEdgeNOut;
RWStructuredBuffer<float2> _BoundaryEdgeNIn;
RWStructuredBuffer<float2> _BoundaryEdgePseudoN0;
RWStructuredBuffer<float2> _BoundaryEdgePseudoN1;

RWStructuredBuffer<uint> _BoundaryVertexOwner;
RWStructuredBuffer<uint> _BoundaryVertexGi;
RWStructuredBuffer<float2> _BoundaryVertexP;
RWStructuredBuffer<float2> _BoundaryVertexPseudoN;

RWStructuredBuffer<uint> _EdgeBinCounts;
RWStructuredBuffer<uint> _EdgeBinOverflow;
RWStructuredBuffer<uint> _EdgeBinRefs;

RWStructuredBuffer<uint> _VertBinCounts;
RWStructuredBuffer<uint> _VertBinOverflow;
RWStructuredBuffer<uint> _VertBinRefs;

RWStructuredBuffer<float> _SdfPhi;
RWStructuredBuffer<float2> _SdfGrad;
RWStructuredBuffer<uint> _SdfFeatType;
RWStructuredBuffer<uint> _SdfFeatId;

RWStructuredBuffer<Contact> _Contacts;
RWStructuredBuffer<uint> _ContactCount;
