#ifndef XPBI_SOLVER_SHARED_INCLUDED
    #define XPBI_SOLVER_SHARED_INCLUDED

    #include "Utils.hlsl"
    #include "Deformation.hlsl"

    #define targetNeighborCount 16

    // Core particle data
    RWStructuredBuffer<float2> _Pos;
    RWStructuredBuffer<float2> _Vel;

    // Material and state
    RWStructuredBuffer<uint> _CoarseFixed;
    StructuredBuffer<int> _MaterialIds;
    StructuredBuffer<float4> _PhysicalParams; // young, poisson, yieldHencky, volumetricHenckyLimit
    int _PhysicalParamCount;
    StructuredBuffer<float> _InvMass;
    StructuredBuffer<float> _RestVolume;
    RWStructuredBuffer<int> _ParentIndex;

    // Deformation gradients
    RWStructuredBuffer<float4> _F;
    RWStructuredBuffer<float4> _Fp;

    // SPH and plasticity
    RWStructuredBuffer<uint> _CurrentVolumeBits;
    RWStructuredBuffer<uint> _CurrentTotalMassBits;
    RWStructuredBuffer<uint> _FixedChildPosBits;
    RWStructuredBuffer<uint> _FixedChildCount;
    RWStructuredBuffer<float4> _L;     // correction matrix
    RWStructuredBuffer<float4> _F0;    // initial deformation
    RWStructuredBuffer<float> _Lambda; // Lagrange multiplier
    RWStructuredBuffer<float> _CollisionLambda;

    // Velocity delta accumulation
    RWStructuredBuffer<float2> _SavedVelPrefix;
    RWStructuredBuffer<uint> _VelDeltaBits;

    // Neighbour lists
    StructuredBuffer<uint> _DtNeighbors;
    StructuredBuffer<uint> _DtNeighborCounts;
    StructuredBuffer<int> _DtOwnerByLocal;
    StructuredBuffer<int> _DtGlobalNodeMap;
    StructuredBuffer<int> _DtGlobalToLayerLocalMap;
    uint _UseDtOwnerFilter;
    uint _UseDtGlobalNodeMap;
    uint _DtLocalBase;

    // Colouring order
    StructuredBuffer<uint> _ColorOrder;
    StructuredBuffer<uint> _ColorCounts;
    StructuredBuffer<uint> _ColorStarts;
    uint _ColorIndex;

    // Inherited forces for upper layers
    RWStructuredBuffer<uint>   _RestrictedDeltaVBits;
    RWStructuredBuffer<uint>   _RestrictedDeltaVCount;
    RWStructuredBuffer<float2> _RestrictedDeltaVAvg;

    float _RestrictedDeltaVScale;
    float _ProlongationScale;
    float _PostProlongSmoothing;
    float _LayerKernelH;
    float _WendlandSupport;
    float _CollisionSupportScale;
    float _CollisionCompliance;
    float _CollisionFriction;
    float _CollisionRestitution;
    float _CollisionRestitutionThreshold;
    uint _CollisionEnable;
    uint _UseAffineProlongation; 

    // Simulation ranges
    uint _DtNeighborCount;
    uint _Base;
    uint _ActiveCount;
    uint _TotalCount;
    uint _FineCount;

    // Simulation parameters
    float _Dt;
    float _Gravity;
    float _Compliance;
    float _MaxSpeed;
    float _MaxStep;

    // Constants
    static const float EPS = 1e-6;
    static const float KERNEL_H_SCALE = 0.7;

    static const float STRETCH_EPS = 1e-6;
    static const float EIGEN_OFFDIAG_EPS = 1e-5;
    static const float INV_DET_EPS = 1e-8;

    static const float DEFAULT_YOUNGS = 5e4;
    static const float DEFAULT_POISSON = 0.3;
    static const float DEFAULT_YIELD_HENCKY = 0.05;
    static const float DEFAULT_VOL_HENCKY_LIMIT = 0.3;
    static const float MIN_EFFECTIVE_MASS = 1e-4;
    static const float MAX_EFFECTIVE_INV_MASS = 1e4;

    static float WendlandSupportRadius(float h)
    {
        return max(_WendlandSupport, 0.0) * h;
    }

    static float WendlandKernelHFromSupport(float h)
    {
        return 0.5 * max(_WendlandSupport, 0.0) * h;
    }

    // ----------------------------------------------------------------------------
    // Utility functions
    // ----------------------------------------------------------------------------
    static bool IsFixedVertex(uint gi)
    {
        return _InvMass[gi] <= 0.0;
    }

    static float ReadCurrentVolume(uint gi)
    {
        return asfloat(_CurrentVolumeBits[gi]);
    }

    static float ReadCurrentTotalMass(uint gi)
    {
        return asfloat(_CurrentTotalMassBits[gi]);
    }

    static uint ReadFixedChildCount(uint gi)
    {
        return _FixedChildCount[gi];
    }

    static float2 ReadFixedChildPosSum(uint gi)
    {
        float2 s;
        s.x = asfloat(_FixedChildPosBits[gi * 2u + 0u]);
        s.y = asfloat(_FixedChildPosBits[gi * 2u + 1u]);
        return s;
    }

    static float2 ReadFixedChildAnchor(uint gi)
    {
        uint cnt = ReadFixedChildCount(gi);
        if (cnt == 0u)
        return 0.0;
        return ReadFixedChildPosSum(gi) / max((float)cnt, 1.0);
    }

    static float ReadEffectiveInvMass(uint gi)
    {
        float totalMass = ReadCurrentTotalMass(gi);
        if (totalMass > EPS)
        return min(1.0 / max(totalMass, MIN_EFFECTIVE_MASS), MAX_EFFECTIVE_INV_MASS);
        return 0.0;
    }

    static uint LocalIndexFromGlobal(uint gi)
    {
        if (_UseDtGlobalNodeMap != 0u)
        {
            int liSigned = _DtGlobalToLayerLocalMap[gi];
            if (liSigned < 0)
            return ~0u;
            return (uint)liSigned;
        }
        return gi - _Base;
    }

    static uint GlobalIndexFromLocal(uint li)
    {
        if (_UseDtGlobalNodeMap != 0u)
        {
            int giSigned = _DtGlobalNodeMap[li];
            if (giSigned < 0)
            return ~0u;
            return (uint)giSigned;
        }
        return _Base + li;
    }

    static float4 ReadMaterialPhysical(uint gi)
    {
        if (_PhysicalParamCount <= 0)
        return float4(DEFAULT_YOUNGS, DEFAULT_POISSON, DEFAULT_YIELD_HENCKY, DEFAULT_VOL_HENCKY_LIMIT);

        int materialId = _MaterialIds[gi];
        if (materialId < 0 || materialId >= _PhysicalParamCount)
        return float4(DEFAULT_YOUNGS, DEFAULT_POISSON, DEFAULT_YIELD_HENCKY, DEFAULT_VOL_HENCKY_LIMIT);

        return _PhysicalParams[materialId];
    }

    static void ComputeMaterialLame(uint gi, out float mu, out float lambda)
    {
        float4 mp = ReadMaterialPhysical(gi);
        float young = mp.x > EPS ? mp.x : DEFAULT_YOUNGS;
        float poisson = (mp.y > -0.5 && mp.y < 0.5) ? mp.y : DEFAULT_POISSON;
        poisson = clamp(poisson, -0.499, 0.499);
        mu = young / (2.0 * (1.0 + poisson));
        lambda = (young * poisson) / ((1.0 + poisson) * max(1.0 - 2.0 * poisson, EPS));
    }

    static float ReadMaterialYieldHencky(uint gi)
    {
        float yieldHencky = ReadMaterialPhysical(gi).z;
        return yieldHencky > EPS ? yieldHencky : DEFAULT_YIELD_HENCKY;
    }

    static float ReadMaterialVolHenckyLimit(uint gi)
    {
        float volHenckyLimit = ReadMaterialPhysical(gi).w;
        return volHenckyLimit > EPS ? volHenckyLimit : DEFAULT_VOL_HENCKY_LIMIT;
    }

    // ----------------------------------------------------------------------------
    // Neighbour access
    // ----------------------------------------------------------------------------
    [forceinline] static void GetNeighborsRaw(uint gi, out uint nCount, out uint ns[targetNeighborCount])
    {
        uint li = LocalIndexFromGlobal(gi);
        if (li == ~0u)
        {
            nCount = 0u;
            [unroll] for (uint i = 0; i < targetNeighborCount; i++) ns[i] = ~0u;
            return;
        }

        uint dtLi = (_UseDtGlobalNodeMap != 0u) ? li : (_DtLocalBase + li);

        nCount = _DtNeighborCounts[dtLi];
        nCount = min(nCount, _DtNeighborCount);
        nCount = min(nCount, targetNeighborCount);

        uint baseIdx = dtLi * _DtNeighborCount;

        [unroll] for (uint i = 0; i < targetNeighborCount; i++)
        {
            if (i < nCount)
            ns[i] = GlobalIndexFromLocal(_DtNeighbors[baseIdx + i]);
            else
            ns[i] = ~0u;
        }
    }

    [forceinline] static void GetNeighbors(uint gi, out uint nCount, out uint ns[targetNeighborCount])
    {
        uint rawCount;
        uint rawNs[targetNeighborCount];
        GetNeighborsRaw(gi, rawCount, rawNs);

        if (rawCount == 0u)
        {
            nCount = 0u;
            [unroll] for (uint i = 0u; i < targetNeighborCount; i++) ns[i] = ~0u;
            return;
        }

        float h = max(_LayerKernelH, 1e-4);
        float support = WendlandSupportRadius(h);
        if (support <= EPS)
        {
            nCount = 0u;
            [unroll] for (uint i = 0u; i < targetNeighborCount; i++) ns[i] = ~0u;
            return;
        }

        float supportSq = support * support;
        float2 xi = _Pos[gi];

        uint li = LocalIndexFromGlobal(gi);
        bool useOwnerFilter = (_UseDtOwnerFilter != 0u) && (li != ~0u) && (li < _ActiveCount);
        int owner = useOwnerFilter ? _DtOwnerByLocal[li] : -1;

        uint outCount = 0u;
        [unroll] for (uint i = 0u; i < targetNeighborCount; i++)
        {
            if (i >= rawCount) break;

            uint gj = rawNs[i];
            if (gj == ~0u) continue;

            if (useOwnerFilter)
            {
                uint gjLi = LocalIndexFromGlobal(gj);
                if (gjLi == ~0u || gjLi >= _ActiveCount) continue;
                if (_DtOwnerByLocal[gjLi] != owner) continue;
            }

            float2 d = _Pos[gj] - xi;
            if (dot(d, d) > supportSq) continue;

            ns[outCount++] = gj;
        }

        nCount = outCount;
        [unroll] for (uint i = outCount; i < targetNeighborCount; i++) ns[i] = ~0u;
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