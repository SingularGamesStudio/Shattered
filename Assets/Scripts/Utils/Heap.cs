using System;
using System.Collections.Generic;

public struct Heap<TKey, TValue, TComparer> where TComparer : struct, IComparer<TKey>
{
    private TKey[] keys;
    private TValue[] values;
    private TComparer comparer;

    public int Count { get; private set; }

    public Heap(int capacity)
    {
        keys = new TKey[Math.Max(capacity, 1)];
        values = new TValue[keys.Length];
        comparer = default;
        Count = 0;
    }

    public void Clear()
    {
        Count = 0;
    }

    public TKey PeekKey()
    {
        return keys[0];
    }

    private bool Better(TKey a, TKey b)
    {
        return comparer.Compare(a, b) < 0;
    }

    public void Push(TKey key, TValue value)
    {
        if (Count == keys.Length)
        {
            Array.Resize(ref keys, Math.Max(Count * 2, 16));
            Array.Resize(ref values, keys.Length);
        }

        int i = Count++;
        while (i > 0)
        {
            int p = (i - 1) >> 1;
            if (!Better(key, keys[p]))
                break;

            keys[i] = keys[p];
            values[i] = values[p];
            i = p;
        }

        keys[i] = key;
        values[i] = value;
    }

    public void Pop(out TKey key, out TValue value)
    {
        key = keys[0];
        value = values[0];

        int last = --Count;
        if (Count <= 0)
        {
            Count = Math.Max(Count, 0);
            return;
        }

        TKey lk = keys[last];
        TValue lv = values[last];

        int i = 0;
        while (true)
        {
            int l = (i << 1) + 1;
            if (l >= Count)
                break;

            int r = l + 1;
            int c = (r < Count && Better(keys[r], keys[l])) ? r : l;

            if (!Better(keys[c], lk))
                break;

            keys[i] = keys[c];
            values[i] = values[c];
            i = c;
        }

        keys[i] = lk;
        values[i] = lv;
    }

    public void CopyValuesTo(List<TValue> dst)
    {
        for (int i = 0; i < Count; i++)
            dst.Add(values[i]);
    }
}

public readonly struct AscendingFloatComparer : IComparer<float>
{
    public int Compare(float x, float y)
    {
        if (x < y) return -1;
        if (x > y) return 1;
        return 0;
    }
}

public readonly struct DescendingFloatComparer : IComparer<float>
{
    public int Compare(float x, float y)
    {
        if (x > y) return -1;
        if (x < y) return 1;
        return 0;
    }
}
