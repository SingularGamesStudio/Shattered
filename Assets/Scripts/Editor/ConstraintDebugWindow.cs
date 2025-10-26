using UnityEngine;
using UnityEditor;
using Physics;
using System.Linq;
using System.Collections.Generic;

public class ConstraintDebugWindow : EditorWindow {
    private Vector2 scrollPos;
    private int selectedNode = -1;
    private bool showHealthyNodes = false;
    private float positionDeltaThreshold = 0.01f;
    private bool debugMode = false;

    private enum SortMode { NodeIndex, MaxDelta, AvgDelta, DegenerateCount, NaNCount, SkippedCount }
    private SortMode sortMode = SortMode.MaxDelta;
    private bool sortDescending = true;

    private string selectedConstraintType = "All";
    private List<string> availableConstraintTypes = new List<string>();

    [MenuItem("Window/Constraint Debugger")]
    public static void ShowWindow() {
        GetWindow<ConstraintDebugWindow>("Constraint Debug");
    }

    void OnEnable() {
        EditorApplication.update += UpdateWindow;
    }

    void OnDisable() {
        EditorApplication.update -= UpdateWindow;
    }

    void UpdateWindow() {
        Repaint();
    }


    void OnGUI() {
        debugMode = EditorGUILayout.Toggle("1 Iter/Frame", debugMode);

        if (debugMode) {
            Const.SolverIterations = 1;
        } else {
            Const.SolverIterations = Const._defaultSolverIterations;
        }

        Meshless targetMeshless = null;
        if (Selection.activeGameObject != null)
            targetMeshless = Selection.activeGameObject.GetComponent<Meshless>();

        EditorGUILayout.LabelField("Constraint Debugger", EditorStyles.boldLabel);

        if (targetMeshless == null) {
            EditorGUILayout.HelpBox("Select a GameObject with a Meshless component in the Hierarchy.", MessageType.Info);
            return;
        }

        var nodeBatch = targetMeshless.lastBatchDebug;
        if (nodeBatch == null) {
            EditorGUILayout.HelpBox("Node batch not available", MessageType.Warning);
            return;
        }

        UpdateAvailableConstraints(nodeBatch);

        // Control panel
        EditorGUILayout.BeginVertical(EditorStyles.helpBox);
        EditorGUILayout.LabelField("Filters & Settings", EditorStyles.boldLabel);

        EditorGUILayout.BeginHorizontal();
        EditorGUILayout.LabelField("Constraint Type:", GUILayout.Width(120));
        selectedConstraintType = EditorGUILayout.Popup(
            availableConstraintTypes.IndexOf(selectedConstraintType),
            availableConstraintTypes.ToArray()
        ) >= 0 ? availableConstraintTypes[EditorGUILayout.Popup(
            availableConstraintTypes.IndexOf(selectedConstraintType),
            availableConstraintTypes.ToArray()
        )] : "All";
        EditorGUILayout.EndHorizontal();

        showHealthyNodes = EditorGUILayout.Toggle("Show Healthy Nodes", showHealthyNodes);
        positionDeltaThreshold = EditorGUILayout.Slider("Position Δ Threshold", positionDeltaThreshold, 0.0001f, 1f);

        EditorGUILayout.BeginHorizontal();
        EditorGUILayout.LabelField("Sort By:", GUILayout.Width(60));
        sortMode = (SortMode)EditorGUILayout.EnumPopup(sortMode);
        sortDescending = GUILayout.Toggle(sortDescending, sortDescending ? "↓" : "↑", GUILayout.Width(30));
        EditorGUILayout.EndHorizontal();
        EditorGUILayout.EndVertical();

        EditorGUILayout.Space();

        // Global statistics
        DrawGlobalStats(nodeBatch);

        EditorGUILayout.Space();

        // Node list
        scrollPos = EditorGUILayout.BeginScrollView(scrollPos);
        DrawNodeList(nodeBatch);
        EditorGUILayout.EndScrollView();
    }

    void UpdateAvailableConstraints(NodeBatch batch) {
        HashSet<string> types = new HashSet<string> { "All" };

        foreach (var cache in batch.caches) {
            foreach (var key in cache.debugDataPerConstraint.Keys) {
                types.Add(key);
            }
        }

        availableConstraintTypes = types.OrderBy(t => t == "All" ? "" : t).ToList();

        if (!availableConstraintTypes.Contains(selectedConstraintType)) {
            selectedConstraintType = "All";
        }
    }

    void DrawGlobalStats(NodeBatch batch) {
        EditorGUILayout.BeginVertical(EditorStyles.helpBox);
        EditorGUILayout.LabelField("Global Statistics", EditorStyles.boldLabel);

        var constraintsToShow = selectedConstraintType == "All"
            ? availableConstraintTypes.Where(t => t != "All").ToList()
            : new List<string> { selectedConstraintType };

        foreach (var constraintType in constraintsToShow) {
            EditorGUILayout.LabelField($"--- {constraintType} ---", EditorStyles.boldLabel);

            int totalDegenerate = 0;
            int totalNaN = 0;
            int totalSkipped = 0;
            float maxDelta = 0f;
            float avgDelta = 0f;
            int nodesAffected = 0;

            foreach (var cache in batch.caches) {
                if (!cache.debugDataPerConstraint.ContainsKey(constraintType)) continue;

                var debug = cache.debugDataPerConstraint[constraintType];
                totalDegenerate += debug.degenerateCount;
                totalNaN += debug.nanInfCount;
                totalSkipped += debug.skippedIterations;
                maxDelta = Mathf.Max(maxDelta, debug.maxPositionDelta);
                avgDelta += debug.avgPositionDelta;
                if (debug.positionUpdateCount > 0) nodesAffected++;
            }

            if (nodesAffected > 0) avgDelta /= nodesAffected;

            EditorGUILayout.LabelField($"  Nodes Affected: {nodesAffected} / {batch.Count}");
            EditorGUILayout.LabelField($"  Max Position Δ: {maxDelta:F6}", maxDelta > positionDeltaThreshold ? GetErrorStyle() : EditorStyles.label);
            EditorGUILayout.LabelField($"  Avg Position Δ: {avgDelta:F6}");
            EditorGUILayout.LabelField($"  Degenerate Cases: {totalDegenerate}", totalDegenerate > 0 ? GetWarningStyle() : EditorStyles.label);
            EditorGUILayout.LabelField($"  NaN/Inf Cases: {totalNaN}", totalNaN > 0 ? GetErrorStyle() : EditorStyles.label);
            EditorGUILayout.LabelField($"  Skipped Iterations: {totalSkipped}", totalSkipped > 0 ? GetWarningStyle() : EditorStyles.label);

            EditorGUILayout.Space(5);
        }

        EditorGUILayout.EndVertical();
    }

    void DrawNodeList(NodeBatch batch) {
        var sortedIndices = Enumerable.Range(0, batch.Count).ToList();

        sortedIndices.Sort((a, b) => {
            float metricA = GetSortMetric(batch.caches[a], selectedConstraintType);
            float metricB = GetSortMetric(batch.caches[b], selectedConstraintType);
            int comparison = metricA.CompareTo(metricB);
            return sortDescending ? -comparison : comparison;
        });

        foreach (int i in sortedIndices) {
            var node = batch.nodes[i];
            var cache = batch.caches[i];

            bool hasIssues = CheckNodeHasIssues(cache, selectedConstraintType);

            if (!hasIssues && !showHealthyNodes) continue;

            DrawNodeEntry(i, node, cache, hasIssues);
        }
    }

    float GetSortMetric(ConstraintCache cache, string constraintType) {
        var debugData = GetAggregatedDebugData(cache, constraintType);

        switch (sortMode) {
            case SortMode.NodeIndex: return 0;
            case SortMode.MaxDelta: return debugData.maxPositionDelta;
            case SortMode.AvgDelta: return debugData.avgPositionDelta;
            case SortMode.DegenerateCount: return debugData.degenerateCount;
            case SortMode.NaNCount: return debugData.nanInfCount;
            case SortMode.SkippedCount: return debugData.skippedIterations;
            default: return 0;
        }
    }

    ConstraintDebugData GetAggregatedDebugData(ConstraintCache cache, string constraintType) {
        var aggregated = new ConstraintDebugData();

        var dataList = constraintType == "All"
            ? cache.debugDataPerConstraint.Values.ToList()
            : cache.debugDataPerConstraint.ContainsKey(constraintType)
                ? new List<ConstraintDebugData> { cache.debugDataPerConstraint[constraintType] }
                : new List<ConstraintDebugData>();

        foreach (var data in dataList) {
            aggregated.maxPositionDelta = Mathf.Max(aggregated.maxPositionDelta, data.maxPositionDelta);
            aggregated.avgPositionDelta += data.avgPositionDelta;
            aggregated.degenerateCount += data.degenerateCount;
            aggregated.nanInfCount += data.nanInfCount;
            aggregated.skippedIterations += data.skippedIterations;
            aggregated.positionUpdateCount += data.positionUpdateCount;
        }

        if (dataList.Count > 0) aggregated.avgPositionDelta /= dataList.Count;

        return aggregated;
    }

    bool CheckNodeHasIssues(ConstraintCache cache, string constraintType) {
        var data = GetAggregatedDebugData(cache, constraintType);
        return data.maxPositionDelta > positionDeltaThreshold ||
               data.degenerateCount > 0 ||
               data.nanInfCount > 0 ||
               data.skippedIterations > 0;
    }

    void DrawNodeEntry(int index, Node node, ConstraintCache cache, bool hasIssues) {
        var bgColor = hasIssues ? new Color(1f, 0.8f, 0.8f) : new Color(0.8f, 1f, 0.8f);
        var oldColor = GUI.backgroundColor;
        GUI.backgroundColor = bgColor;

        EditorGUILayout.BeginVertical(EditorStyles.helpBox);
        GUI.backgroundColor = oldColor;

        bool expanded = selectedNode == index;

        var aggregated = GetAggregatedDebugData(cache, selectedConstraintType);
        string header = $"Node {index} | Max Δ: {aggregated.maxPositionDelta:F4} | Avg Δ: {aggregated.avgPositionDelta:F4}";
        if (aggregated.degenerateCount > 0) header += $" | Degen: {aggregated.degenerateCount}";
        if (aggregated.nanInfCount > 0) header += $" | NaN: {aggregated.nanInfCount}";
        if (aggregated.skippedIterations > 0) header += $" | Skip: {aggregated.skippedIterations}";

        if (GUILayout.Button(header, expanded ? EditorStyles.boldLabel : EditorStyles.label)) {
            selectedNode = expanded ? -1 : index;
        }

        if (expanded) {
            EditorGUI.indentLevel++;

            // Basic node info
            EditorGUILayout.LabelField("Node Information", EditorStyles.boldLabel);
            EditorGUILayout.LabelField($"Position: ({node.pos.x:F3}, {node.pos.y:F3})");
            EditorGUILayout.LabelField($"Predicted: ({node.predPos.x:F3}, {node.predPos.y:F3})");
            EditorGUILayout.LabelField($"Inv Mass: {node.invMass:F4}");
            EditorGUILayout.LabelField($"Fixed: {node.isFixed}");

            EditorGUILayout.Space();

            // Per-constraint breakdown
            var constraintsToShow = selectedConstraintType == "All"
                ? cache.debugDataPerConstraint.Keys.ToList()
                : cache.debugDataPerConstraint.ContainsKey(selectedConstraintType)
                    ? new List<string> { selectedConstraintType }
                    : new List<string>();

            foreach (var constraintType in constraintsToShow) {
                var debug = cache.debugDataPerConstraint[constraintType];

                EditorGUILayout.LabelField($"--- {constraintType} ---", EditorStyles.boldLabel);

                EditorGUILayout.LabelField($"  Position Updates: {debug.positionUpdateCount}");
                EditorGUILayout.LabelField($"  Max Position Δ: {debug.maxPositionDelta:F6}",
                    debug.maxPositionDelta > positionDeltaThreshold ? GetErrorStyle() : EditorStyles.label);
                EditorGUILayout.LabelField($"  Avg Position Δ: {debug.avgPositionDelta:F6}");

                if (debug.degenerateCount > 0) {
                    EditorGUILayout.LabelField($"  Degenerate: {debug.degenerateCount}", GetWarningStyle());
                }
                if (debug.nanInfCount > 0) {
                    EditorGUILayout.LabelField($"  NaN/Inf: {debug.nanInfCount}", GetErrorStyle());
                }
                if (debug.skippedIterations > 0) {
                    EditorGUILayout.LabelField($"  Skipped: {debug.skippedIterations}", GetWarningStyle());
                }

                EditorGUILayout.Space(5);
            }

            EditorGUI.indentLevel--;
        }

        EditorGUILayout.EndVertical();
    }

    GUIStyle GetErrorStyle() {
        var style = new GUIStyle(EditorStyles.label);
        style.normal.textColor = Color.red;
        style.fontStyle = FontStyle.Bold;
        return style;
    }

    GUIStyle GetWarningStyle() {
        var style = new GUIStyle(EditorStyles.label);
        style.normal.textColor = new Color(1f, 0.5f, 0f);
        return style;
    }
}