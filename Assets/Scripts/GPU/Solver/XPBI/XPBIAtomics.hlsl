#ifndef XPBI_ATOMICS_INCLUDED
#define XPBI_ATOMICS_INCLUDED

static void XPBI_AtomicAddFloatBits(RWStructuredBuffer<uint> buf, uint idx, float add)
{
    uint expected;
    uint original;

    [loop] for (int it = 0; it < 64; it++)
    {
        expected = buf[idx];
        float cur = asfloat(expected);
        uint desired = asuint(cur + add);

        InterlockedCompareExchange(buf[idx], expected, desired, original);
        if (original == expected)
            return;
    }
}

#endif
