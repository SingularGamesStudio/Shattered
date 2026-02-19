#ifndef XPBI_SOLVER_SHARED_INCLUDED
    #define XPBI_SOLVER_SHARED_INCLUDED

    #include "Utils.hlsl"
    #include "Deformation.hlsl"

    #define targetNeighborCount 6

    // Core particle data
    RWStructuredBuffer<float2> _Pos;
    RWStructuredBuffer<float2> _Vel;

    // Material and state
    StructuredBuffer<float> _InvMass;
    StructuredBuffer<uint> _Flags;
    StructuredBuffer<float> _RestVolume;
    RWStructuredBuffer<int> _ParentIndex;

    // Deformation gradients
    RWStructuredBuffer<float4> _F;
    RWStructuredBuffer<float4> _Fp;

    // SPH and plasticity
    RWStructuredBuffer<uint> _CurrentVolumeBits;
    RWStructuredBuffer<float> _KernelH;
    RWStructuredBuffer<float4> _L;     // correction matrix
    RWStructuredBuffer<float4> _F0;    // initial deformation
    RWStructuredBuffer<float> _Lambda; // Lagrange multiplier

    // Velocity delta accumulation
    RWStructuredBuffer<float2> _SavedVelPrefix;
    RWStructuredBuffer<uint> _VelDeltaBits;

    // Neighbour lists
    StructuredBuffer<uint> _DtNeighbors;
    StructuredBuffer<uint> _DtNeighborCounts;

    // Colouring order
    StructuredBuffer<uint> _ColorOrder;
    StructuredBuffer<uint> _ColorCounts;
    StructuredBuffer<uint> _ColorStarts;
    uint _ColorIndex;

    uint _DtNeighborCount;

    // Simulation ranges
    uint _Base;
    uint _ActiveCount;
    uint _TotalCount;
    uint _FineCount;

    // Simulation parameters
    float _Dt;
    float _Gravity;
    float _Compliance;

    // Constants
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

    // ----------------------------------------------------------------------------
    // Utility functions
    // ----------------------------------------------------------------------------
    static bool IsFixedVertex(uint gi)
    {
        return (_Flags[gi] & 1u) != 0u || _InvMass[gi] <= 0.0;
    }

    static float ReadCurrentVolume(uint gi)
    {
        return asfloat(_CurrentVolumeBits[gi]);
    }

    static uint LocalIndexFromGlobal(uint gi) { return gi - _Base; }
    static uint GlobalIndexFromLocal(uint li) { return _Base + li; }

    // ----------------------------------------------------------------------------
    // Neighbour access – returns up to 6 neighbours (global indices) and count.
    // ----------------------------------------------------------------------------
    [forceinline] static void GetNeighbors(uint gi,
    out uint nCount,
    out uint n0, out uint n1, out uint n2,
    out uint n3, out uint n4, out uint n5)
    {
        uint li = LocalIndexFromGlobal(gi);

        nCount = _DtNeighborCounts[li];
        nCount = min(nCount, _DtNeighborCount);
        nCount = min(nCount, targetNeighborCount);

        uint baseIdx = li * _DtNeighborCount;

        n0 = (nCount > 0) ? GlobalIndexFromLocal(_DtNeighbors[baseIdx + 0u]) : ~0u;
        n1 = (nCount > 1) ? GlobalIndexFromLocal(_DtNeighbors[baseIdx + 1u]) : ~0u;
        n2 = (nCount > 2) ? GlobalIndexFromLocal(_DtNeighbors[baseIdx + 2u]) : ~0u;
        n3 = (nCount > 3) ? GlobalIndexFromLocal(_DtNeighbors[baseIdx + 3u]) : ~0u;
        n4 = (nCount > 4) ? GlobalIndexFromLocal(_DtNeighbors[baseIdx + 4u]) : ~0u;
        n5 = (nCount > 5) ? GlobalIndexFromLocal(_DtNeighbors[baseIdx + 5u]) : ~0u;
    }

    // ----------------------------------------------------------------------------
    // External force event structure
    // ----------------------------------------------------------------------------
    struct XPBI_ForceEvent
    {
        uint node;
        float2 force;
    };

    StructuredBuffer<XPBI_ForceEvent> _ForceEvents;
    uint _ForceEventCount;

    // ----------------------------------------------------------------------------
    // Multigrid prolongation ranges
    // ----------------------------------------------------------------------------
    RWStructuredBuffer<float2> _DtPositions;
    float2 _DtNormCenter;
    float _DtNormInvHalfExtent;

    uint _ParentRangeStart;
    uint _ParentRangeEnd;
    uint _ParentCoarseCount;

#endif // XPBI_SOLVER_SHARED_INCLUDED