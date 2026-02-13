using System;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Runtime.CompilerServices;
using System.Text;
using UnityEngine;

public static class LoopProfiler {
    public enum Section : int {
        TickTotal = 0,

        ExternalForces,
        SolveTotal,
        IntegrateAndUpdate,
        HierarchyRebuild,

        SolveSavedVelCopy,
        SolveBatchInit,
        SolveRelax,
        SolveProlongate,
        SolveFinalizeDebug,
        SolveCommitDeformation,

        // NodeBatch.Initialise internals
        BatchCacheVolumes,
        BatchCacheNeighbors,
        BatchCacheKernelRadii,
        BatchComputeCorrectionMatrices,
        BatchCorrectionMatrixSum,
        BatchCorrectionMatrixPseudoInverse,
        BatchResetDebugData,
        BatchCacheF0,

        // XPBIConstraint internals (Relax)
        RelaxEstimateGradV,
        RelaxFtrial,
        RelaxPlasticityReturn,
        RelaxConstraintEval,
        RelaxComputeGradient,
        RelaxGradCAccum,
        RelaxDenomAccum,
        RelaxLambdaUpdate,
        RelaxApplyVelocities,

        // XPBIConstraint internals (CommitDeformation)
        CommitEstimateGradV,
        CommitFtrial,
        CommitPlasticityReturn,
        CommitFpUpdate,

        Count
    }

    static bool enabledInController;
    static bool capturing;
    static int warmupRemaining;
    static int targetTicks;
    static int capturedTicks;

    static int configuredWarmupTicks;
    static int configuredTargetTicks;
    static string configuredOutputDirectory;

    static long[] ticks;
    static float[] dt;
    static int[] meshlessCount;
    static int[] nodesTotal;
    static int[] maxNodes;
    static int[] maxLevel;
    static bool[] hierarchical;

    static int currentIndex = -1;
    static string scenario = "default";
    static string lastSavedPath;

    public static bool IsActive => capturing || warmupRemaining > 0;
    public static string LastSavedPath => lastSavedPath;

    public static void SetEnabledInController(bool enabled) {
        enabledInController = enabled;
    }

    public static void StartCapture(int ticksToCapture, int warmupTicks, string scenarioName, string outputDirectory) {
        if (!enabledInController) return;
        if (ticksToCapture <= 0) return;

        scenario = SanitizeFileName(string.IsNullOrWhiteSpace(scenarioName) ? "default" : scenarioName);

        configuredTargetTicks = ticksToCapture;
        configuredWarmupTicks = Mathf.Max(0, warmupTicks);
        configuredOutputDirectory = outputDirectory;

        targetTicks = ticksToCapture;
        warmupRemaining = configuredWarmupTicks;
        capturedTicks = 0;
        currentIndex = -1;

        ticks = new long[ticksToCapture * (int)Section.Count];
        dt = new float[ticksToCapture];
        meshlessCount = new int[ticksToCapture];
        nodesTotal = new int[ticksToCapture];
        maxNodes = new int[ticksToCapture];
        maxLevel = new int[ticksToCapture];
        hierarchical = new bool[ticksToCapture];

        capturing = true;
        lastSavedPath = null;
    }

    public static void CancelCapture() {
        capturing = false;
        warmupRemaining = 0;
        targetTicks = 0;
        capturedTicks = 0;
        currentIndex = -1;

        configuredTargetTicks = 0;
        configuredWarmupTicks = 0;
        configuredOutputDirectory = null;

        ticks = null;
        dt = null;
        meshlessCount = null;
        nodesTotal = null;
        maxNodes = null;
        maxLevel = null;
        hierarchical = null;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static long Stamp() {
        return IsActive ? Stopwatch.GetTimestamp() : 0L;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void Add(Section section, long startStamp) {
        if (!IsActive || startStamp == 0L || currentIndex < 0) return;
        ticks[currentIndex * (int)Section.Count + (int)section] += Stopwatch.GetTimestamp() - startStamp;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void BeginTick(float tickDt, int tickMeshlessCount, int tickNodesTotal, int tickMaxNodes, int tickMaxLevel, bool useHierarchicalSolver) {
        if (!capturing) return;

        if (warmupRemaining > 0) {
            warmupRemaining--;
            currentIndex = -1;
            return;
        }

        if (capturedTicks >= targetTicks) {
            currentIndex = -1;
            return;
        }

        currentIndex = capturedTicks;

        dt[currentIndex] = tickDt;
        meshlessCount[currentIndex] = tickMeshlessCount;
        nodesTotal[currentIndex] = tickNodesTotal;
        maxNodes[currentIndex] = tickMaxNodes;
        maxLevel[currentIndex] = tickMaxLevel;
        hierarchical[currentIndex] = useHierarchicalSolver;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static void EndTick() {
        if (!capturing) return;
        if (currentIndex < 0) return;

        capturedTicks++;
        currentIndex = -1;

        if (capturedTicks >= targetTicks) {
            SaveMarkdownSummary();
            CancelCapture();
        }
    }

    static void SaveMarkdownSummary() {
        string dir = ResolveOutputDirectory(configuredOutputDirectory);
        string stamp = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss", CultureInfo.InvariantCulture);
        string fileName = $"loop_profile_{scenario}_{stamp}_summary_ticks{targetTicks}.md";

        try {
            Directory.CreateDirectory(dir);
            string path = Path.Combine(dir, fileName);

            WriteMarkdown(path);

            lastSavedPath = path;
            UnityEngine.Debug.Log($"LoopProfiler saved: {path}");
        }
        catch (Exception e) {
            UnityEngine.Debug.LogError($"LoopProfiler failed to save to '{dir}' (file '{fileName}'): {e.Message}");

            try {
                string fallbackDir = Application.persistentDataPath;
                Directory.CreateDirectory(fallbackDir);
                string fallbackPath = Path.Combine(fallbackDir, fileName);

                WriteMarkdown(fallbackPath);

                lastSavedPath = fallbackPath;
                UnityEngine.Debug.Log($"LoopProfiler saved (fallback): {fallbackPath}");
            }
            catch (Exception fallbackE) {
                UnityEngine.Debug.LogError($"LoopProfiler fallback save failed: {fallbackE.Message}");
                lastSavedPath = null;
            }
        }
    }

    static void WriteMarkdown(string path) {
        using var stream = new FileStream(path, FileMode.Create, FileAccess.Write, FileShare.Read);
        using var writer = new StreamWriter(stream, new UTF8Encoding(false), 1 << 20);

        int sectionCount = (int)Section.Count;

        var sum = new long[sectionCount];
        var min = new long[sectionCount];
        var max = new long[sectionCount];
        for (int s = 0; s < sectionCount; s++) min[s] = long.MaxValue;

        for (int i = 0; i < targetTicks; i++) {
            int baseIdx = i * sectionCount;
            for (int s = 0; s < sectionCount; s++) {
                long t = ticks[baseIdx + s];
                sum[s] += t;
                if (t < min[s]) min[s] = t;
                if (t > max[s]) max[s] = t;
            }
        }

        double freq = Stopwatch.Frequency;
        double tickTotalAvgTicks = sum[(int)Section.TickTotal] / (double)targetTicks;

        writer.WriteLine("# Loop profile summary");
        writer.WriteLine();
        writer.WriteLine($"- UTC: `{DateTime.UtcNow:O}`");
        writer.WriteLine($"- Scenario: `{scenario}`");
        writer.WriteLine($"- Captured ticks: `{configuredTargetTicks}`");
        writer.WriteLine($"- Warmup ticks: `{configuredWarmupTicks}`");
        writer.WriteLine($"- Output directory: `{Path.GetDirectoryName(path)}`");
        writer.WriteLine($"- Unity: `{Application.unityVersion}`");
        writer.WriteLine($"- App version: `{Application.version}`");
        writer.WriteLine($"- Platform: `{Application.platform}`");
        writer.WriteLine();
        writer.WriteLine("Notes: Timings are measured with `Stopwatch` and aggregated over captured ticks; nested sections can overlap if you time both outer and inner scopes.");
        writer.WriteLine();

        writer.WriteLine("| Section | Min (ms) | Avg (ms) | Max (ms) | Avg % of tick |");
        writer.WriteLine("|---|---:|---:|---:|---:|");

        for (int s = 0; s < sectionCount; s++) {
            long minTicks = min[s] == long.MaxValue ? 0L : min[s];
            double minMs = minTicks * 1000.0 / freq;
            double avgMs = (sum[s] / (double)targetTicks) * 1000.0 / freq;
            double maxMs = max[s] * 1000.0 / freq;
            double avgPct = tickTotalAvgTicks > 0.0 ? (sum[s] / (double)targetTicks) / tickTotalAvgTicks * 100.0 : 0.0;

            writer.Write("| ");
            writer.Write(((Section)s).ToString());
            writer.Write(" | ");
            writer.Write(FormatHuman(minMs));
            writer.Write(" | ");
            writer.Write(FormatHuman(avgMs));
            writer.Write(" | ");
            writer.Write(FormatHuman(maxMs));
            writer.Write(" | ");
            writer.Write(FormatHuman(avgPct));
            writer.WriteLine(" |");
        }

        writer.WriteLine();
        writer.WriteLine("## Meta (averages)");
        writer.WriteLine();
        writer.WriteLine("| Metric | Avg |");
        writer.WriteLine("|---|---:|");

        double sumDt = 0.0;
        double sumMeshless = 0.0;
        double sumNodesTotal = 0.0;
        double sumMaxNodes = 0.0;
        double sumMaxLevel = 0.0;
        double sumHier = 0.0;

        for (int i = 0; i < targetTicks; i++) {
            sumDt += dt[i];
            sumMeshless += meshlessCount[i];
            sumNodesTotal += nodesTotal[i];
            sumMaxNodes += maxNodes[i];
            sumMaxLevel += maxLevel[i];
            sumHier += hierarchical[i] ? 1.0 : 0.0;
        }

        writer.Write("| dt | ");
        writer.Write(FormatHuman(sumDt / targetTicks));
        writer.WriteLine(" |");

        writer.Write("| meshlessCount | ");
        writer.Write(FormatHuman(sumMeshless / targetTicks));
        writer.WriteLine(" |");

        writer.Write("| nodesTotal | ");
        writer.Write(FormatHuman(sumNodesTotal / targetTicks));
        writer.WriteLine(" |");

        writer.Write("| maxNodes | ");
        writer.Write(FormatHuman(sumMaxNodes / targetTicks));
        writer.WriteLine(" |");

        writer.Write("| maxLevel | ");
        writer.Write(FormatHuman(sumMaxLevel / targetTicks));
        writer.WriteLine(" |");

        writer.Write("| hierarchical | ");
        writer.Write(FormatHuman(sumHier / targetTicks));
        writer.WriteLine(" |");
    }

    static string ResolveOutputDirectory(string requested) {
        if (string.IsNullOrWhiteSpace(requested)) return Application.persistentDataPath;

        if (Path.IsPathRooted(requested)) return requested;

#if UNITY_EDITOR
        string projectRoot = Path.GetDirectoryName(Application.dataPath);
        return Path.GetFullPath(Path.Combine(projectRoot, requested));
#else
        return Path.GetFullPath(Path.Combine(Application.persistentDataPath, requested));
#endif
    }

    static string SanitizeFileName(string s) {
        var invalid = Path.GetInvalidFileNameChars();
        for (int i = 0; i < invalid.Length; i++) {
            s = s.Replace(invalid[i], '_');
        }
        return s;
    }

    static string FormatHuman(double v) {
        if (double.IsNaN(v) || double.IsInfinity(v)) return "NaN";

        double a = v < 0.0 ? -v : v;

        if (a >= 1000.0) return v.ToString("0.0", CultureInfo.InvariantCulture);
        if (a >= 100.0) return v.ToString("0.00", CultureInfo.InvariantCulture);
        if (a >= 10.0) return v.ToString("0.000", CultureInfo.InvariantCulture);
        return v.ToString("0.0000", CultureInfo.InvariantCulture);
    }
}
