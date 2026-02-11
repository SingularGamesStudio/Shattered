public static class Const {
    public const float Eps = 1e-6f;
    public const int NeighborCount = 6;

    // sqrt(0.5)
    public const float KernelHScale = 0.7f;

    public const int _defaultIterations = 8;
    public static int Iterations = _defaultIterations;
    public const int HPBDIterations = 2;

    public const int HierarchyRebuildInterval = 60;
}
