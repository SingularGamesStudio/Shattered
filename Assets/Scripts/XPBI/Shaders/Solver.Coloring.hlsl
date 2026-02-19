#ifndef SOLVER_COLORING_INCLUDED
    #define SOLVER_COLORING_INCLUDED

    StructuredBuffer<uint> _ColoringDtNeighbors;
    StructuredBuffer<uint> _ColoringDtNeighborCounts;

    RWStructuredBuffer<int> _ColoringColor;
    RWStructuredBuffer<int> _ColoringProposed;
    RWStructuredBuffer<uint> _ColoringPrio;

    RWStructuredBuffer<uint> _ColoringCounts;
    RWStructuredBuffer<uint> _ColoringStarts;
    RWStructuredBuffer<uint> _ColoringWrite;
    RWStructuredBuffer<uint> _ColoringOrderOut;

    RWStructuredBuffer<uint> _RelaxArgs;

    uint _ColoringActiveCount;
    uint _ColoringDtNeighborCount;
    uint _ColoringMaxColors;
    uint _ColoringSeed;

    groupshared uint gColoringCountsScan[64];

    // -----------------------------------------------------------------------------
    // Hash function for priority generation.
    uint ColoringHash(uint x)
    {
        x ^= x >> 16;
        x *= 0x7feb352du;
        x ^= x >> 15;
        x *= 0x846ca68bu;
        x ^= x >> 16;
        return x;
    }

    uint GetPrio(uint i)
    {
        return ColoringHash(i ^ _ColoringSeed);
    }

    // Clamp neighbor count to valid range.
    uint ClampNeighborCount(uint n)
    {
        return min(n, _ColoringDtNeighborCount);
    }

    // -----------------------------------------------------------------------------
    // Bit mask helpers for colors 0..63 (colors beyond are not used in choose phase).
    void MarkUsed(inout uint2 used, int c)
    {
        if (c < 0 || c >= 64)
        return;
        if (c < 32)
        used.x |= (1u << c);
        else
        used.y |= (1u << (c - 32));
    }

    bool IsUsed(uint2 used, int c)
    {
        if (c < 0 || c >= 64)
        return false;
        if (c < 32)
        return (used.x & (1u << c)) != 0u;
        else
        return (used.y & (1u << (c - 32))) != 0u;
    }

    // -----------------------------------------------------------------------------
    // Check if there exists a vertex with the same color and higher priority
    // within 2-hop neighbourhood.
    bool HasHigherPrioSameColor2Hop(uint i, int myC, uint myP)
    {
        uint baseI = i * _ColoringDtNeighborCount;
        uint nI = ClampNeighborCount(_ColoringDtNeighborCounts[i]);

        // 1‑hop neighbours
        for (uint a = 0; a < nI; a++)
        {
            uint na = _ColoringDtNeighbors[baseI + a];
            if (na >= _ColoringActiveCount)
            continue;

            if (_ColoringColor[na] == myC)
            {
                uint p = _ColoringPrio[na];
                if (p > myP || (p == myP && na < i))
                return true; // early exit
            }
        }

        // 2‑hop neighbours (skip self)
        for (uint a = 0; a < nI; a++)
        {
            uint na = _ColoringDtNeighbors[baseI + a];
            if (na >= _ColoringActiveCount)
            continue;

            uint baseA = na * _ColoringDtNeighborCount;
            uint nA = ClampNeighborCount(_ColoringDtNeighborCounts[na]);
            for (uint b = 0; b < nA; b++)
            {
                uint nb = _ColoringDtNeighbors[baseA + b];
                if (nb >= _ColoringActiveCount || nb == i)
                continue;

                if (_ColoringColor[nb] == myC)
                {
                    uint p = _ColoringPrio[nb];
                    if (p > myP || (p == myP && nb < i))
                    return true;
                }
            }
        }

        return false;
    }

    // -----------------------------------------------------------------------------
    // Initialization: set priority and initial color (mod 64).
    [numthreads(256, 1, 1)] void ColoringInit(uint3 id : SV_DispatchThreadID)
    {
        uint i = id.x;
        if (i >= _ColoringActiveCount)
        return;

        _ColoringPrio[i] = GetPrio(i);
        _ColoringColor[i] = int(i & 63u);
        _ColoringProposed[i] = _ColoringColor[i];
    }

    // -----------------------------------------------------------------------------
    // Conflict detection: propose -1 if a higher‑priority same‑color vertex exists.
    [numthreads(256, 1, 1)] void ColoringDetectConflicts(uint3 id : SV_DispatchThreadID)
    {
        uint i = id.x;
        if (i >= _ColoringActiveCount)
        return;

        int myC = _ColoringColor[i];
        if (myC < 0)
        {
            _ColoringProposed[i] = -1;
            return;
        }

        uint myP = _ColoringPrio[i];
        _ColoringProposed[i] = HasHigherPrioSameColor2Hop(i, myC, myP) ? -1 : myC;
    }

    // -----------------------------------------------------------------------------
    // Choose a new color for uncolored vertices (color = -1) using 2‑hop avoidance.
    [numthreads(256, 1, 1)] void ColoringChoose(uint3 id : SV_DispatchThreadID)
    {
        uint i = id.x;
        if (i >= _ColoringActiveCount)
        return;

        // Already colored? Keep it.
        if (_ColoringColor[i] >= 0)
        {
            _ColoringProposed[i] = _ColoringColor[i];
            return;
        }

        // Collect used colors from 1‑hop and 2‑hop neighbours (colors 0..63).
        uint2 used = 0;

        uint baseI = i * _ColoringDtNeighborCount;
        uint nI = ClampNeighborCount(_ColoringDtNeighborCounts[i]);

        // 1‑hop
        for (uint a = 0; a < nI; a++)
        {
            uint na = _ColoringDtNeighbors[baseI + a];
            if (na >= _ColoringActiveCount)
            continue;
            MarkUsed(used, _ColoringColor[na]);
        }

        // 2‑hop
        for (uint a = 0; a < nI; a++)
        {
            uint na = _ColoringDtNeighbors[baseI + a];
            if (na >= _ColoringActiveCount)
            continue;

            uint baseA = na * _ColoringDtNeighborCount;
            uint nA = ClampNeighborCount(_ColoringDtNeighborCounts[na]);
            for (uint b = 0; b < nA; b++)
            {
                uint nb = _ColoringDtNeighbors[baseA + b];
                if (nb >= _ColoringActiveCount)
                continue;
                MarkUsed(used, _ColoringColor[nb]);
            }
        }

        // Find smallest free color (limit to 64).
        uint maxC = min(_ColoringMaxColors, 64u);
        int chosen = -1;
        for (uint c = 0; c < maxC; c++)
        {
            if (!IsUsed(used, int(c)))
            {
                chosen = int(c);
                break;
            }
        }

        _ColoringProposed[i] = chosen;
    }

    // -----------------------------------------------------------------------------
    // Apply the proposed colors.
    [numthreads(256, 1, 1)] void ColoringApply(uint3 id : SV_DispatchThreadID)
    {
        uint i = id.x;
        if (i >= _ColoringActiveCount)
        return;
        _ColoringColor[i] = _ColoringProposed[i];
    }

    // -----------------------------------------------------------------------------
    // Clear per‑color metadata (counts, starts, write pointers).
    [numthreads(256, 1, 1)] void ColoringClearMeta(uint3 id : SV_DispatchThreadID)
    {
        uint c = id.x;
        if (c >= _ColoringMaxColors)
        return;
        _ColoringCounts[c] = 0;
        _ColoringStarts[c] = 0;
        _ColoringWrite[c] = 0;
    }

    // -----------------------------------------------------------------------------
    // Count vertices per color.
    [numthreads(256, 1, 1)] void ColoringBuildCounts(uint3 id : SV_DispatchThreadID)
    {
        uint i = id.x;
        if (i >= _ColoringActiveCount)
        return;

        int color = _ColoringColor[i];
        if (color < 0 || uint(color) >= _ColoringMaxColors)
        return;

        InterlockedAdd(_ColoringCounts[color], 1u);
    }

    // -----------------------------------------------------------------------------
    // Compute exclusive prefix sums per color (start indices) using groupshared memory.
    [numthreads(64, 1, 1)] void ColoringBuildStarts(uint3 id : SV_DispatchThreadID)
    {
        uint t = id.x;
        uint cnt = (t < _ColoringMaxColors) ? _ColoringCounts[t] : 0u;
        gColoringCountsScan[t] = cnt;
        GroupMemoryBarrierWithGroupSync();

        // Up‑sweep (inclusive scan)
        [unroll] for (uint offset = 1; offset < 64; offset <<= 1)
        {
            uint v = (t >= offset) ? gColoringCountsScan[t - offset] : 0u;
            GroupMemoryBarrierWithGroupSync(); // ensure previous step is visible
            gColoringCountsScan[t] += v;
            GroupMemoryBarrierWithGroupSync();
        }

        if (t < _ColoringMaxColors)
        {
            uint inclusive = gColoringCountsScan[t];
            uint start = inclusive - cnt; // exclusive
            _ColoringStarts[t] = start;
            _ColoringWrite[t] = start;
        }
    }

    // -----------------------------------------------------------------------------
    // Scatter vertex indices into ordered buffer per color.
    [numthreads(256, 1, 1)] void ColoringScatterOrder(uint3 id : SV_DispatchThreadID)
    {
        uint i = id.x;
        if (i >= _ColoringActiveCount)
        return;

        int color = _ColoringColor[i];
        if (color < 0 || uint(color) >= _ColoringMaxColors)
        return;

        uint dst;
        InterlockedAdd(_ColoringWrite[color], 1u, dst);
        _ColoringOrderOut[dst] = i;
    }

    // -----------------------------------------------------------------------------
    // Build indirect dispatch arguments per color.
    [numthreads(64, 1, 1)] void ColoringBuildRelaxArgs(uint3 id : SV_DispatchThreadID)
    {
        uint c = id.x;
        if (c >= _ColoringMaxColors)
        return;

        uint count = _ColoringCounts[c];
        uint groupsX = (count + 255u) / 256u;

        _RelaxArgs[c * 3 + 0] = groupsX;
        _RelaxArgs[c * 3 + 1] = 1u;
        _RelaxArgs[c * 3 + 2] = 1u;
    }

#endif