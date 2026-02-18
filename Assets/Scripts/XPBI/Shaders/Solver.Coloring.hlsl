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

RWStructuredBuffer<uint> _RelaxArgs;

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

bool HasHigherPrioSameColor2Hop(int i, int myC, uint myP)
{
    int baseI = i * _ColoringDtNeighborCount;
    int nI = ClampNeighborCount(_ColoringDtNeighborCounts[i]);

    for (int a = 0; a < nI; a++)
    {
        int na = _ColoringDtNeighbors[baseI + a];
        if ((uint)na >= (uint)_ColoringActiveCount)
            continue;

        if (_ColoringColor[na] == myC)
        {
            uint p = _ColoringPrio[na];
            if (p > myP || (p == myP && na < i))
                return true;
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
            if (nb == i)
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

[numthreads(256, 1, 1)] void ColoringInit(uint3 id : SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= (uint)_ColoringActiveCount)
        return;

    _ColoringPrio[i] = GetPrio(i);
    _ColoringColor[i] = (int)(i & 63u);
    _ColoringProposed[i] = _ColoringColor[i];
}

    [numthreads(256, 1, 1)] void ColoringDetectConflicts(uint3 id : SV_DispatchThreadID)
{
    int i = (int)id.x;
    if ((uint)i >= (uint)_ColoringActiveCount)
        return;

    int myC = _ColoringColor[i];
    if (myC < 0)
    {
        _ColoringProposed[i] = -1;
        return;
    }

    uint myP = _ColoringPrio[i];

    if (HasHigherPrioSameColor2Hop(i, myC, myP))
        _ColoringProposed[i] = -1;
    else
        _ColoringProposed[i] = myC;
}

[numthreads(256, 1, 1)] void ColoringChoose(uint3 id : SV_DispatchThreadID)
{
    int i = (int)id.x;
    if ((uint)i >= (uint)_ColoringActiveCount)
        return;

    if (_ColoringColor[i] >= 0)
    {
        _ColoringProposed[i] = _ColoringColor[i];
        return;
    }

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

    [numthreads(256, 1, 1)] void ColoringApply(uint3 id : SV_DispatchThreadID)
{
    int i = (int)id.x;
    if ((uint)i >= (uint)_ColoringActiveCount)
        return;

    _ColoringColor[i] = _ColoringProposed[i];
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

[numthreads(64, 1, 1)] void ColoringBuildRelaxArgs(uint3 id : SV_DispatchThreadID)
{
    int c = (int)id.x;
    if (c >= _ColoringMaxColors)
        return;

    uint count = (uint)max(_ColoringCounts[c], 0);
    uint groupsX = (count + 255u) / 256u;

    _RelaxArgs[c * 3 + 0] = groupsX;
    _RelaxArgs[c * 3 + 1] = 1u;
    _RelaxArgs[c * 3 + 2] = 1u;
}

#endif
