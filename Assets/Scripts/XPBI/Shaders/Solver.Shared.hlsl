#ifndef XPBI_SOLVER_SHARED_INCLUDED
    #define XPBI_SOLVER_SHARED_INCLUDED

    #include "Utils.hlsl"

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
    RWStructuredBuffer<float> _DurabilityLambda;
    RWStructuredBuffer<float> _CollisionLambda;

    struct XPBI_CollisionEvent
    {
        uint aGi;
        uint bGi;
        float4 nPen;
    };

    RWStructuredBuffer<XPBI_CollisionEvent> _CollisionEvents;
    RWStructuredBuffer<uint> _CollisionEventCount;
    uint _CollisionEventCapacity;

    RWStructuredBuffer<uint> _XferColCount;
    RWStructuredBuffer<uint> _XferColNXBits;
    RWStructuredBuffer<uint> _XferColNYBits;
    RWStructuredBuffer<uint> _XferColPenBits;
    uint _UseTransferredCollisions;

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
    uint _PersistentIters;
    uint _PersistentBaseDebugIter;

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
    float _DurabilityCompliance;
    float _DurabilityMaxDistanceRatio;
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

    // JR stage buffers
    RWStructuredBuffer<float2> _VelPrev;     // snapshot of _Vel
    RWStructuredBuffer<float>  _LambdaPrev;  // snapshot of _Lambda

    RWStructuredBuffer<uint>   _JRVelDeltaBits;  // asuint(float2) packed as [2*gi+0, 2*gi+1]
    RWStructuredBuffer<float>  _JRLambdaDelta;   // per-vertex, no contention (1 thread -> 1 gi)

    // JR relaxation factors
    float _JROmegaV;
    float _JROmegaL;



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
        float2 s = 0.0;
        s.x = asfloat(_FixedChildPosBits[gi * 2u + 0u]);
        s.y = asfloat(_FixedChildPosBits[gi * 2u + 1u]);
        return s;
    }

    static float2 ReadFixedChildAnchor(uint gi)
    {
        float2 result = 0.0;
        uint cnt = 0u;
        cnt = ReadFixedChildCount(gi);
        if (cnt != 0u)
        {
            float2 posSum = 0.0;
            posSum = ReadFixedChildPosSum(gi);
            result = posSum / max((float)cnt, 1.0);
        }
        return result;
    }

    static float ReadEffectiveInvMass(uint gi)
    {
        float result = 0.0;
        float totalMass = 0.0;
        totalMass = ReadCurrentTotalMass(gi);
        if (totalMass > EPS)
        result = min(1.0 / max(totalMass, MIN_EFFECTIVE_MASS), MAX_EFFECTIVE_INV_MASS);
        return result;
    }

    static uint LocalIndexFromGlobal(uint gi)
    {
        uint result = ~0u;
        if (_UseDtGlobalNodeMap != 0u)
        {
            int liSigned = _DtGlobalToLayerLocalMap[gi];
            if (liSigned >= 0)
            result = (uint)liSigned;
        }
        else
        {
            result = gi - _Base;
        }
        return result;
    }

    static uint GlobalIndexFromLocal(uint li)
    {
        uint result = ~0u;
        if (_UseDtGlobalNodeMap != 0u)
        {
            int giSigned = _DtGlobalNodeMap[li];
            if (giSigned >= 0)
            result = (uint)giSigned;
        }
        else
        {
            result = _Base + li;
        }
        return result;
    }

    static float4 ReadMaterialPhysical(uint gi)
    {
        float4 result = float4(DEFAULT_YOUNGS, DEFAULT_POISSON, DEFAULT_YIELD_HENCKY, DEFAULT_VOL_HENCKY_LIMIT);
        if (_PhysicalParamCount > 0)
        {
            int materialId = _MaterialIds[gi];
            if (materialId >= 0 && materialId < _PhysicalParamCount)
            result = _PhysicalParams[materialId];
        }
        return result;
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
        float4 mp = 0.0;
        mp = ReadMaterialPhysical(gi);
        float yieldHencky = mp.z;
        float result = DEFAULT_YIELD_HENCKY;
        if (yieldHencky > EPS)
        result = yieldHencky;
        return result;
    }

    static float ReadMaterialVolHenckyLimit(uint gi)
    {
        float4 mp = 0.0;
        mp = ReadMaterialPhysical(gi);
        float volHenckyLimit = mp.w;
        float result = DEFAULT_VOL_HENCKY_LIMIT;
        if (volHenckyLimit > EPS)
        result = volHenckyLimit;
        return result;
    }

    static bool IsLayerFixed(uint gi)
    {
        bool result = false;
        if (IsFixedVertex(gi))
        {
            result = true;
        }
        else
        {
            uint li = ~0u;
            li = LocalIndexFromGlobal(gi);
            if (li != ~0u && li < _ActiveCount)
            result = _CoarseFixed[gi] != 0u;
        }
        return result;
    }

    static float EffectiveVolumeForCompliance(uint gi)
    {
        float result = max(_RestVolume[gi], EPS);
        float currentVol = 0.0;
        currentVol = ReadCurrentVolume(gi);
        if (currentVol > EPS)
        result = currentVol;
        return result;
    }

    static void ApplySingleAnchorRadialDampingOnVel(uint gi, float mu, float lambda, float2 pos, inout float2 vel)
    {
        uint fixedChildCount = 0u;
        fixedChildCount = ReadFixedChildCount(gi);
        if (fixedChildCount != 1u)
        return;

        float2 fixedAnchor = 0.0;
        fixedAnchor = ReadFixedChildAnchor(gi);
        float2 r = pos - fixedAnchor;
        float rLen = length(r);
        if (rLen <= EPS)
        return;

        float2 radial = r / rLen;
        float vr = dot(vel, radial);
        float radialKeep = saturate(_Compliance / (_Compliance + (_Dt * _Dt) * (mu + lambda) / EffectiveVolumeForCompliance(gi)));
        vel -= radial * vr * (1.0 - radialKeep);
    }

    // ----------------------------------------------------------------------------
    // Neighbour access
    // ----------------------------------------------------------------------------
    static void GetNeighborsRaw(uint gi, out uint nCount, out uint ns[targetNeighborCount])
    {
        uint li = ~0u;
        li = LocalIndexFromGlobal(gi);
        if (li == ~0u)
        {
            nCount = 0u;
            [unroll] for (uint clearIdx0 = 0u; clearIdx0 < targetNeighborCount; clearIdx0++) ns[clearIdx0] = ~0u;
            return;
        }

        uint dtLi = (_UseDtGlobalNodeMap != 0u) ? li : (_DtLocalBase + li);

        nCount = _DtNeighborCounts[dtLi];
        nCount = min(nCount, _DtNeighborCount);
        nCount = min(nCount, targetNeighborCount);

        uint baseIdx = dtLi * _DtNeighborCount;

        [unroll] for (uint copyIdx0 = 0u; copyIdx0 < targetNeighborCount; copyIdx0++)
        {
            if (copyIdx0 < nCount)
            {
                uint gjTmp = ~0u;
                gjTmp = GlobalIndexFromLocal(_DtNeighbors[baseIdx + copyIdx0]);
                ns[copyIdx0] = gjTmp;
            }
            else
            ns[copyIdx0] = ~0u;
        }
    }

    static void GetNeighbors(uint gi, out uint nCount, out uint ns[targetNeighborCount])
    {
        uint rawCount = 0u;
        uint rawNs[targetNeighborCount];
        GetNeighborsRaw(gi, rawCount, rawNs);

        if (rawCount == 0u)
        {
            nCount = 0u;
            [unroll] for (uint clearIdx1 = 0u; clearIdx1 < targetNeighborCount; clearIdx1++) ns[clearIdx1] = ~0u;
            return;
        }

        uint li = ~0u;
        li = LocalIndexFromGlobal(gi);
        bool useOwnerFilter = (_UseDtOwnerFilter != 0u) && (li != ~0u) && (li < _ActiveCount);
        int owner = useOwnerFilter ? _DtOwnerByLocal[li] : -1;

        uint outCount = 0u;
        [unroll] for (uint filterIdx = 0u; filterIdx < targetNeighborCount; filterIdx++)
        {
            if (filterIdx >= rawCount) break;

            uint gj = rawNs[filterIdx];
            if (gj == ~0u) continue;

            if (useOwnerFilter)
            {
                uint gjLi = ~0u;
                gjLi = LocalIndexFromGlobal(gj);
                if (gjLi == ~0u || gjLi >= _ActiveCount) continue;
                if (_DtOwnerByLocal[gjLi] != owner) continue;
            }

            ns[outCount++] = gj;
        }

        nCount = outCount;
        [unroll] for (uint clearIdx3 = outCount; clearIdx3 < targetNeighborCount; clearIdx3++) ns[clearIdx3] = ~0u;
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