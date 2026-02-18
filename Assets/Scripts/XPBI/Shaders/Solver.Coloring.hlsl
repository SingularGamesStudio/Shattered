#ifndef SOLVER_COLORING_INCLUDED
#define SOLVER_COLORING_INCLUDED

StructuredBuffer<int> _ColoringDtNeighbors;
StructuredBuffer<int> _ColoringDtNeighborCounts;

RWStructuredBuffer<int> _ColoringColor;
RWStructuredBuffer<int> _ColoringProposed;
RWStructuredBuffer<uint> _ColoringPrio;

RWStructuredBuffer<int> _ColoringCounts;
RWStructuredBuffer<int> _ColoringStarts;
RWStructuredBuffer<int> _ColoringWrite;
RWStructuredBuffer<int> _ColoringOrderOut;

RWStructuredBuffer<int> _ColoringStats; // [0]=uncoloredCount, [1]=changedFlag

int _ColoringActiveCount;
int _ColoringDtNeighborCount;
int _ColoringMaxColors;
uint _ColoringSeed;

groupshared int gColoringCountsScan[64];

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

int ClampNeighborCount(int n)
{
    if (n < 0)
        return 0;
    return n > _ColoringDtNeighborCount ? _ColoringDtNeighborCount : n;
}

void MarkUsed(inout uint2 used, int c)
{
    if (c < 0)
        return;
    if (c < 32)
        used.x |= (1u << c);
    else if (c < 64)
        used.y |= (1u << (c - 32));
}

bool IsUsed(uint2 used, int c)
{
    if (c < 0)
        return false;
    if (c < 32)
        return (used.x & (1u << c)) != 0u;
    if (c < 64)
        return (used.y & (1u << (c - 32))) != 0u;
    return true;
}

[numthreads(256, 1, 1)] void ColoringInit(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= (uint)_ColoringActiveCount)
        return;

    _ColoringColor[i] = -1;
    _ColoringProposed[i] = -1;
    _ColoringPrio[i] = GetPrio(i);
}

    [numthreads(1, 1, 1)] void ColoringClearChanged(uint3 id : SV_DispatchThreadID)
{
    _ColoringStats[1] = 0;
}

[numthreads(1, 1, 1)] void ColoringClearUncolored(uint3 id : SV_DispatchThreadID)
{
    _ColoringStats[0] = 0;
}

    [numthreads(256, 1, 1)] void ColoringPropose(uint3 id : SV_DispatchThreadID)
{
    int i = (int)id.x;
    if ((uint)i >= (uint)_ColoringActiveCount)
        return;
    if (_ColoringColor[i] >= 0)
        return;

    uint2 used = uint2(0u, 0u);

    int baseI = i * _ColoringDtNeighborCount;
    int nI = ClampNeighborCount(_ColoringDtNeighborCounts[i]);

    for (int a = 0; a < nI; a++)
    {
        int na = _ColoringDtNeighbors[baseI + a];
        if ((uint)na >= (uint)_ColoringActiveCount)
            continue;

        MarkUsed(used, _ColoringColor[na]);
    }

    for (int a = 0; a < nI; a++)
    {
        int na = _ColoringDtNeighbors[baseI + a];
        if ((uint)na >= (uint)_ColoringActiveCount)
            continue;

        int baseA = na * _ColoringDtNeighborCount;
        int nA = ClampNeighborCount(_ColoringDtNeighborCounts[na]);

        for (int b = 0; b < nA; b++)
        {
            int nb = _ColoringDtNeighbors[baseA + b];
            if ((uint)nb >= (uint)_ColoringActiveCount)
                continue;

            MarkUsed(used, _ColoringColor[nb]);
        }
    }

    int chosen = -1;
    int maxC = _ColoringMaxColors;
    if (maxC > 64)
        maxC = 64;

    for (int c = 0; c < maxC; c++)
    {
        if (!IsUsed(used, c))
        {
            chosen = c;
            break;
        }
    }

    _ColoringProposed[i] = chosen;
}

[numthreads(256, 1, 1)] void ColoringResolve(uint3 id : SV_DispatchThreadID)
{
    int i = (int)id.x;
    if ((uint)i >= (uint)_ColoringActiveCount)
        return;
    if (_ColoringColor[i] >= 0)
        return;

    int myC = _ColoringProposed[i];
    if (myC < 0)
        return;

    uint myP = _ColoringPrio[i];

    int baseI = i * _ColoringDtNeighborCount;
    int nI = ClampNeighborCount(_ColoringDtNeighborCounts[i]);

    for (int a = 0; a < nI; a++)
    {
        int na = _ColoringDtNeighbors[baseI + a];
        if ((uint)na >= (uint)_ColoringActiveCount)
            continue;

        if (_ColoringColor[na] < 0 && _ColoringProposed[na] == myC)
        {
            uint p = _ColoringPrio[na];
            if (p > myP || (p == myP && na < i))
                return;
        }
    }

    for (int a = 0; a < nI; a++)
    {
        int na = _ColoringDtNeighbors[baseI + a];
        if ((uint)na >= (uint)_ColoringActiveCount)
            continue;

        int baseA = na * _ColoringDtNeighborCount;
        int nA = ClampNeighborCount(_ColoringDtNeighborCounts[na]);

        for (int b = 0; b < nA; b++)
        {
            int nb = _ColoringDtNeighbors[baseA + b];
            if ((uint)nb >= (uint)_ColoringActiveCount)
                continue;

            if (_ColoringColor[nb] < 0 && _ColoringProposed[nb] == myC)
            {
                uint p = _ColoringPrio[nb];
                if (p > myP || (p == myP && nb < i))
                    return;
            }
        }
    }

    _ColoringColor[i] = myC;

    int original;
    InterlockedOr(_ColoringStats[1], 1, original);
}

    [numthreads(256, 1, 1)] void ColoringCountUncolored(uint3 id : SV_DispatchThreadID)
{
    int i = (int)id.x;
    if ((uint)i >= (uint)_ColoringActiveCount)
        return;

    if (_ColoringColor[i] < 0)
    {
        InterlockedAdd(_ColoringStats[0], 1);
    }
}

[numthreads(256, 1, 1)] void ColoringClearMeta(uint3 id : SV_DispatchThreadID)
{
    int i = (int)id.x;
    if (i >= _ColoringMaxColors)
        return;

    _ColoringCounts[i] = 0;
    _ColoringStarts[i] = 0;
    _ColoringWrite[i] = 0;
}

    [numthreads(256, 1, 1)] void ColoringBuildCounts(uint3 id : SV_DispatchThreadID)
{
    int i = (int)id.x;
    if ((uint)i >= (uint)_ColoringActiveCount)
        return;

    int c = _ColoringColor[i];
    if ((uint)c >= (uint)_ColoringMaxColors)
        return;

    InterlockedAdd(_ColoringCounts[c], 1);
}

[numthreads(64, 1, 1)] void ColoringBuildStarts(uint3 id : SV_DispatchThreadID)
{
    int t = (int)id.x;

    int cnt = (t < _ColoringMaxColors) ? _ColoringCounts[t] : 0;
    gColoringCountsScan[t] = cnt;
    GroupMemoryBarrierWithGroupSync();

    for (int offset = 1; offset < 64; offset <<= 1)
    {
        int v = (t >= offset) ? gColoringCountsScan[t - offset] : 0;
        GroupMemoryBarrierWithGroupSync();
        gColoringCountsScan[t] += v;
        GroupMemoryBarrierWithGroupSync();
    }

    if (t < _ColoringMaxColors)
    {
        int inclusive = gColoringCountsScan[t];
        int start = inclusive - cnt;
        _ColoringStarts[t] = start;
        _ColoringWrite[t] = start;
    }
}

    [numthreads(256, 1, 1)] void ColoringScatterOrder(uint3 id : SV_DispatchThreadID)
{
    int i = (int)id.x;
    if ((uint)i >= (uint)_ColoringActiveCount)
        return;

    int c = _ColoringColor[i];
    if ((uint)c >= (uint)_ColoringMaxColors)
        return;

    int dst;
    InterlockedAdd(_ColoringWrite[c], 1, dst);
    _ColoringOrderOut[dst] = i;
}

#endif
