#ifndef XPBI_SOLVER_SHARED_INCLUDED
#define XPBI_SOLVER_SHARED_INCLUDED

#include "Utils.hlsl"
#include "Deformation.hlsl"

#define targetNeighborCount 6

RWStructuredBuffer<float2> _Pos;
RWStructuredBuffer<float2> _Vel;

StructuredBuffer<float> _InvMass;
StructuredBuffer<uint> _Flags;
StructuredBuffer<float> _RestVolume;
StructuredBuffer<int> _ParentIndex;

RWStructuredBuffer<float4> _F;
RWStructuredBuffer<float4> _Fp;

RWStructuredBuffer<uint> _CurrentVolumeBits;
RWStructuredBuffer<float> _KernelH;
RWStructuredBuffer<float4> _L;
RWStructuredBuffer<float4> _F0;
RWStructuredBuffer<float> _Lambda;

RWStructuredBuffer<float2> _SavedVelPrefix;
RWStructuredBuffer<uint> _VelDeltaBits;

StructuredBuffer<int> _DtNeighbors;
StructuredBuffer<int> _DtNeighborCounts;

StructuredBuffer<int> _ColorOrder;
StructuredBuffer<int> _ColorCounts;
StructuredBuffer<int> _ColorStarts;
int _ColorIndex;

int _DtNeighborCount;

int _Base;
int _ActiveCount;
int _TotalCount;
int _FineCount;
float _Dt;
float _Gravity;
float _Compliance;

static const float EPS = 1e-6;
static const float KERNEL_H_SCALE = 0.7;

static const float STRETCH_EPS = 1e-6;
static const float EIGEN_OFFDIAG_EPS = 1e-5;
static const float INV_DET_EPS = 1e-8;

static const float YOUNGS = 5e4;
static const float POISSON = 0.3;
static const float MU = (YOUNGS / (2.0 * (1.0 + POISSON)));
static const float LAMBDA = ((YOUNGS * POISSON) / ((1.0 + POISSON) * (1.0 - 2.0 * POISSON)));
static const float YIELD_HENCKY = 0.05;
static const float VOL_HENCKY_LIMIT = 0.3;

static bool XPBI_IsFixed(int gi)
{
    return (_Flags[gi] & 1u) != 0u || _InvMass[gi] <= 0.0;
}

static float XPBI_ReadCurrentVolume(int gi)
{
    return asfloat(_CurrentVolumeBits[gi]);
}

static uint XPBI_VelDeltaIndex(int gi, uint comp)
{
    return (uint)(gi * 2) + comp;
}

static void XPBI_AddVelDelta(int gi, float2 dv)
{
    XPBI_AtomicAddFloatBits(_VelDeltaBits, XPBI_VelDeltaIndex(gi, 0u), dv.x);
    XPBI_AtomicAddFloatBits(_VelDeltaBits, XPBI_VelDeltaIndex(gi, 1u), dv.y);
}

static float2 XPBI_ReadVelDelta(int gi)
{
    return float2(
        asfloat(_VelDeltaBits[XPBI_VelDeltaIndex(gi, 0u)]),
        asfloat(_VelDeltaBits[XPBI_VelDeltaIndex(gi, 1u)]));
}

static int XPBI_LocalIndexFromGlobal(int gi) { return gi - _Base; }
static int XPBI_GlobalIndexFromLocal(int li) { return _Base + li; }

static void XPBI_GetNeighbors(int gi, out int nCount, out int n0, out int n1, out int n2, out int n3, out int n4, out int n5)
{
    int li = XPBI_LocalIndexFromGlobal(gi);

    nCount = _DtNeighborCounts[li];
    if (nCount < 0)
        nCount = 0;
    if (nCount > _DtNeighborCount)
        nCount = _DtNeighborCount;
    if (nCount > targetNeighborCount)
        nCount = targetNeighborCount;

    n0 = -1;
    n1 = -1;
    n2 = -1;
    n3 = -1;
    n4 = -1;
    n5 = -1;

    int baseIdx = li * _DtNeighborCount;

    if (nCount > 0)
        n0 = XPBI_GlobalIndexFromLocal(_DtNeighbors[baseIdx + 0]);
    if (nCount > 1)
        n1 = XPBI_GlobalIndexFromLocal(_DtNeighbors[baseIdx + 1]);
    if (nCount > 2)
        n2 = XPBI_GlobalIndexFromLocal(_DtNeighbors[baseIdx + 2]);
    if (nCount > 3)
        n3 = XPBI_GlobalIndexFromLocal(_DtNeighbors[baseIdx + 3]);
    if (nCount > 4)
        n4 = XPBI_GlobalIndexFromLocal(_DtNeighbors[baseIdx + 4]);
    if (nCount > 5)
        n5 = XPBI_GlobalIndexFromLocal(_DtNeighbors[baseIdx + 5]);
}

#endif
