using System.Collections.Generic;

public sealed class NeighborSet
{
    // Switch backend here.
    public const bool UseHashSets = true;

    private readonly HashSet<int> hash;
    private int[] data;
    private int count;

    public int Count => UseHashSets ? hash.Count : count;

    public NeighborSet(int capacity)
    {
        if (UseHashSets)
        {
            hash = new HashSet<int>(capacity);
        }
        else
        {
            data = new int[capacity > 0 ? capacity : 1];
            count = 0;
        }
    }

    public bool Contains(int value)
    {
        if (UseHashSets)
            return hash.Contains(value);

        for (int i = 0; i < count; i++)
        {
            if (data[i] == value)
                return true;
        }
        return false;
    }

    public bool Add(int value)
    {
        if (UseHashSets)
            return hash.Add(value);

        for (int i = 0; i < count; i++)
        {
            if (data[i] == value)
                return false;
        }

        if (count == data.Length)
        {
            int[] next = new int[data.Length * 2];
            for (int i = 0; i < data.Length; i++)
                next[i] = data[i];
            data = next;
        }

        data[count++] = value;
        return true;
    }

    public bool Remove(int value)
    {
        if (UseHashSets)
            return hash.Remove(value);

        for (int i = 0; i < count; i++)
        {
            if (data[i] != value)
                continue;

            count--;
            data[i] = data[count];
            return true;
        }
        return false;
    }

    public Enumerator GetEnumerator() => new Enumerator(this);

    public struct Enumerator
    {
        private HashSet<int>.Enumerator hashEnum;
        private readonly int[] data;
        private readonly int count;
        private int index;
        private int current;

        public int Current => current;

        public Enumerator(NeighborSet set)
        {
            if (UseHashSets)
            {
                hashEnum = set.hash.GetEnumerator();
                data = null;
                count = 0;
                index = 0;
                current = 0;
            }
            else
            {
                hashEnum = default;
                data = set.data;
                count = set.count;
                index = 0;
                current = 0;
            }
        }

        public bool MoveNext()
        {
            if (UseHashSets)
            {
                if (!hashEnum.MoveNext())
                    return false;

                current = hashEnum.Current;
                return true;
            }

            if (index >= count)
                return false;

            current = data[index++];
            return true;
        }
    }
}
