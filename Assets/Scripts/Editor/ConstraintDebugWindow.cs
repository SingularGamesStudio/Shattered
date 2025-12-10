using UnityEngine;
using UnityEditor;
using Physics;
using System.Linq;
using System.Collections.Generic;
using Unity.Mathematics;

public class ConstraintDebugWindow : EditorWindow {
    private Vector2 scrollPos;
    private int selectedNode = -1;
    private bool showHealthyNodes = false;
    private float positionDeltaThreshold = 0.01f;
    private bool debugMode = false;

    // View options
    private bool showNodeInfo = true;
    private bool showDeformationInfo = true;
    private bool showNeighborInfo = false;
    private bool showLambdaInfo = false;

    private enum SortMode {
        NodeIndex,
        MaxDelta,
        AvgDelta,
        DegenerateCount,
        NaNCount,
        PlasticFlowCount
    }
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
        EditorGUILayout.LabelField("Constraint Debugger", EditorStyles.boldLabel);

        scrollPos = EditorGUILayout.BeginScrollView(scrollPos);
        // Simulation controls
        DrawSimulationControls();
        EditorGUILayout.Space();

        Meshless targetMeshless = null;
        if (Selection.activeGameObject != null)
            targetMeshless = Selection.activeGameObject.GetComponent<Meshless>();

        if (targetMeshless == null) {
            EditorGUILayout.EndScrollView();
            EditorGUILayout.HelpBox("Select a GameObject with a Meshless component in the Hierarchy.", MessageType.Info);
            return;
        }

        var nodeBatch = targetMeshless.lastBatchDebug;
        if (nodeBatch == null) {
            EditorGUILayout.EndScrollView();
            EditorGUILayout.HelpBox("Node batch not available", MessageType.Warning);
            return;
        }

        UpdateAvailableConstraints(nodeBatch);

        // Filter and view options
        DrawControlPanel();

        EditorGUILayout.Space();

        // Global statistics
        DrawGlobalStats(nodeBatch);

        EditorGUILayout.Space();

        // Node list

        DrawNodeList(nodeBatch);
        EditorGUILayout.EndScrollView();
    }


    void DrawSimulationControls() {
        EditorGUILayout.BeginVertical(EditorStyles.helpBox);
        EditorGUILayout.LabelField("Simulation Controls", EditorStyles.boldLabel);

        EditorGUI.BeginChangeCheck();
        debugMode = EditorGUILayout.Toggle("Debug Mode (1 Iter/Frame)", debugMode);
        if (EditorGUI.EndChangeCheck()) {
            Const.SolverIterations = debugMode ? 1 : Const._defaultSolverIterations;
        }

        EditorGUILayout.LabelField($"Current Iterations: {Const.SolverIterations}");
        EditorGUILayout.EndVertical();
    }

    void DrawControlPanel() {
        EditorGUILayout.BeginVertical(EditorStyles.helpBox);
        EditorGUILayout.LabelField("Filters & View Options", EditorStyles.boldLabel);

        // Constraint type filter
        EditorGUILayout.BeginHorizontal();
        EditorGUILayout.LabelField("Constraint Type:", GUILayout.Width(120));
        int selectedIndex = availableConstraintTypes.IndexOf(selectedConstraintType);
        selectedIndex = EditorGUILayout.Popup(selectedIndex, availableConstraintTypes.ToArray());
        if (selectedIndex >= 0 && selectedIndex < availableConstraintTypes.Count) {
            selectedConstraintType = availableConstraintTypes[selectedIndex];
        }
        EditorGUILayout.EndHorizontal();

        // Visibility toggles
        showHealthyNodes = EditorGUILayout.Toggle("Show Healthy Nodes", showHealthyNodes);
        positionDeltaThreshold = EditorGUILayout.Slider("Position Δ Threshold", positionDeltaThreshold, 0.0001f, 1f);

        EditorGUILayout.Space(5);
        EditorGUILayout.LabelField("Detail View Options", EditorStyles.boldLabel);
        showNodeInfo = EditorGUILayout.Toggle("Show Node Info", showNodeInfo);
        showDeformationInfo = EditorGUILayout.Toggle("Show Deformation Info", showDeformationInfo);
        showNeighborInfo = EditorGUILayout.Toggle("Show Neighbor Info", showNeighborInfo);
        showLambdaInfo = EditorGUILayout.Toggle("Show Lambda Info", showLambdaInfo);

        EditorGUILayout.Space(5);

        // Sort controls
        EditorGUILayout.BeginHorizontal();
        EditorGUILayout.LabelField("Sort By:", GUILayout.Width(60));
        sortMode = (SortMode)EditorGUILayout.EnumPopup(sortMode);
        sortDescending = GUILayout.Toggle(sortDescending, sortDescending ? "↓" : "↑", GUILayout.Width(30));
        EditorGUILayout.EndHorizontal();

        EditorGUILayout.EndVertical();
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
            int totalPlasticFlow = 0;
            float maxDelta = 0f;
            float avgDelta = 0f;
            float totalEnergy = 0f;
            int nodesAffected = 0;
            float avgIterations = 0f;
            int iterationCount = 0;

            foreach (var cache in batch.caches) {
                if (!cache.debugDataPerConstraint.ContainsKey(constraintType)) continue;

                var debug = cache.debugDataPerConstraint[constraintType];
                totalDegenerate += debug.degenerateCount;
                totalNaN += debug.nanInfCount;
                totalPlasticFlow += debug.plasticFlowCount;
                maxDelta = Mathf.Max(maxDelta, debug.maxPositionDelta);
                avgDelta += debug.avgPositionDelta;
                totalEnergy += debug.constraintEnergy;

                if (debug.positionUpdateCount > 0) {
                    nodesAffected++;
                    avgIterations += debug.iterationsToConverge;
                    iterationCount++;
                }
            }

            if (nodesAffected > 0) {
                avgDelta /= nodesAffected;
                avgIterations /= iterationCount;
            }

            EditorGUILayout.LabelField($"  Nodes Affected: {nodesAffected} / {batch.Count}");
            EditorGUILayout.LabelField($"  Max Position Δ: {maxDelta:F6}",
                maxDelta > positionDeltaThreshold ? GetErrorStyle() : EditorStyles.label);
            EditorGUILayout.LabelField($"  Avg Position Δ: {avgDelta:F6}");
            EditorGUILayout.LabelField($"  Avg Iterations to Converge: {avgIterations:F2}");
            EditorGUILayout.LabelField($"  Total Constraint Energy: {totalEnergy:F6}");
            EditorGUILayout.LabelField($"  Degenerate Cases: {totalDegenerate}",
                totalDegenerate > 0 ? GetWarningStyle() : EditorStyles.label);
            EditorGUILayout.LabelField($"  NaN/Inf Cases: {totalNaN}",
                totalNaN > 0 ? GetErrorStyle() : EditorStyles.label);
            EditorGUILayout.LabelField($"  Plastic Flow Events: {totalPlasticFlow}",
                totalPlasticFlow > 0 ? GetInfoStyle() : EditorStyles.label);

            EditorGUILayout.Space(5);
        }

        // System-wide stats
        EditorGUILayout.Space(5);
        EditorGUILayout.LabelField("System Stats", EditorStyles.boldLabel);
        float kineticEnergy = batch.CalculateKineticEnergy();
        EditorGUILayout.LabelField($"  Total Kinetic Energy: {kineticEnergy:F6}");
        EditorGUILayout.LabelField($"  Total Nodes: {batch.Count}");

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
            case SortMode.PlasticFlowCount: return debugData.plasticFlowCount;
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
            aggregated.plasticFlowCount += data.plasticFlowCount;
            aggregated.positionUpdateCount += data.positionUpdateCount;
            aggregated.constraintEnergy += data.constraintEnergy;
            aggregated.iterationsToConverge = Mathf.Max(aggregated.iterationsToConverge, data.iterationsToConverge);
        }

        if (dataList.Count > 0) aggregated.avgPositionDelta /= dataList.Count;

        return aggregated;
    }

    bool CheckNodeHasIssues(ConstraintCache cache, string constraintType) {
        var data = GetAggregatedDebugData(cache, constraintType);
        return data.maxPositionDelta > positionDeltaThreshold ||
               data.degenerateCount > 0 ||
               data.nanInfCount > 0;
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
        if (aggregated.plasticFlowCount > 0) header += $" | Plastic: {aggregated.plasticFlowCount}";

        if (GUILayout.Button(header, expanded ? EditorStyles.boldLabel : EditorStyles.label)) {
            selectedNode = expanded ? -1 : index;
        }

        if (expanded) {
            EditorGUI.indentLevel++;

            // Basic node info
            if (showNodeInfo) {
                DrawNodeInfo(node, index);
            }

            // Deformation gradient info
            if (showDeformationInfo) {
                DrawDeformationInfo(node, cache);
            }

            // Neighbor info
            if (showNeighborInfo && cache.neighbors != null) {
                DrawNeighborInfo(index, cache);
            }

            // Lambda info
            if (showLambdaInfo) {
                DrawLambdaInfo(cache);
            }

            EditorGUILayout.Space();

            // Per-constraint breakdown
            DrawConstraintBreakdown(cache);

            EditorGUI.indentLevel--;
        }

        EditorGUILayout.EndVertical();
    }

    void DrawNodeInfo(Node node, int index) {
        EditorGUILayout.LabelField("Node Information", EditorStyles.boldLabel);
        EditorGUILayout.LabelField($"  Index: {index}");
        EditorGUILayout.LabelField($"  Position: ({node.pos.x:F3}, {node.pos.y:F3})");
        EditorGUILayout.LabelField($"  Velocity: ({node.vel.x:F3}, {node.vel.y:F3}) | Mag: {math.length(node.vel):F3}");
        EditorGUILayout.LabelField($"  Original Pos: ({node.originalPos.x:F3}, {node.originalPos.y:F3})");
        EditorGUILayout.LabelField($"  Inv Mass: {node.invMass:F4} | Mass: {(node.invMass > 0 ? 1f / node.invMass : float.PositiveInfinity):F4}");
        EditorGUILayout.LabelField($"  Fixed: {node.isFixed}");
        EditorGUILayout.Space(5);
    }

    void DrawDeformationInfo(Node node, ConstraintCache cache) {
        EditorGUILayout.LabelField("Deformation Gradient (Fp)", EditorStyles.boldLabel);

        float det = node.Fp.c0.x * node.Fp.c1.y - node.Fp.c0.y * node.Fp.c1.x;
        var detStyle = Mathf.Abs(det - 1.0f) > 0.1f ? GetWarningStyle() : EditorStyles.label;

        EditorGUILayout.LabelField($"  Fp = [{node.Fp.c0.x:F4}, {node.Fp.c0.y:F4}]");
        EditorGUILayout.LabelField($"       [{node.Fp.c1.x:F4}, {node.Fp.c1.y:F4}]");
        EditorGUILayout.LabelField($"  det(Fp): {det:F6}", detStyle);

        if (cache.L.c0.x != 0 || cache.L.c0.y != 0 || cache.L.c1.x != 0 || cache.L.c1.y != 0) {
            EditorGUILayout.LabelField("Correction Matrix (L)", EditorStyles.boldLabel);
            float detL = cache.L.c0.x * cache.L.c1.y - cache.L.c0.y * cache.L.c1.x;
            EditorGUILayout.LabelField($"  L = [{cache.L.c0.x:F4}, {cache.L.c0.y:F4}]");
            EditorGUILayout.LabelField($"      [{cache.L.c1.x:F4}, {cache.L.c1.y:F4}]");
            EditorGUILayout.LabelField($"  det(L): {detL:F6}");
        }

        EditorGUILayout.Space(5);
    }

    void DrawNeighborInfo(int nodeIndex, ConstraintCache cache) {
        EditorGUILayout.LabelField($"Neighbors ({cache.neighbors.Count})", EditorStyles.boldLabel);

        for (int i = 0; i < cache.neighbors.Count; i++) {
            int neighborIdx = cache.neighbors[i];
            EditorGUILayout.LabelField($"  [{i}] Node {neighborIdx}");
        }

        EditorGUILayout.Space(5);
    }

    void DrawLambdaInfo(ConstraintCache cache) {
        EditorGUILayout.LabelField("Lambda Accumulators", EditorStyles.boldLabel);

        if (cache.lambdas.neighborDistance != null && cache.lambdas.neighborDistance.Count > 0) {
            EditorGUILayout.LabelField("  Distance Constraint Lambdas:");
            for (int i = 0; i < Mathf.Min(cache.lambdas.neighborDistance.Count, 10); i++) {
                EditorGUILayout.LabelField($"    [{i}]: {cache.lambdas.neighborDistance[i]:F6}");
            }
            if (cache.lambdas.neighborDistance.Count > 10) {
                EditorGUILayout.LabelField($"    ... and {cache.lambdas.neighborDistance.Count - 10} more");
            }
        }

        if (cache.lambdas.volume != null && cache.lambdas.volume.Count > 0) {
            EditorGUILayout.LabelField("  Volume Constraint Lambdas:");
            for (int i = 0; i < Mathf.Min(cache.lambdas.volume.Count, 10); i++) {
                EditorGUILayout.LabelField($"    [{i}]: {cache.lambdas.volume[i]:F6}");
            }
            if (cache.lambdas.volume.Count > 10) {
                EditorGUILayout.LabelField($"    ... and {cache.lambdas.volume.Count - 10} more");
            }
        }

        EditorGUILayout.Space(5);
    }

    void DrawConstraintBreakdown(ConstraintCache cache) {
        var constraintsToShow = selectedConstraintType == "All"
            ? cache.debugDataPerConstraint.Keys.ToList()
            : cache.debugDataPerConstraint.ContainsKey(selectedConstraintType)
                ? new List<string> { selectedConstraintType }
                : new List<string>();

        EditorGUILayout.LabelField("Constraint Breakdown", EditorStyles.boldLabel);

        foreach (var constraintType in constraintsToShow) {
            var debug = cache.debugDataPerConstraint[constraintType];

            EditorGUILayout.LabelField($"--- {constraintType} ---", EditorStyles.boldLabel);

            EditorGUILayout.LabelField($"  Position Updates: {debug.positionUpdateCount}");
            EditorGUILayout.LabelField($"  Max Position Δ: {debug.maxPositionDelta:F6}",
                debug.maxPositionDelta > positionDeltaThreshold ? GetErrorStyle() : EditorStyles.label);
            EditorGUILayout.LabelField($"  Avg Position Δ: {debug.avgPositionDelta:F6}");
            EditorGUILayout.LabelField($"  Iterations to Converge: {debug.iterationsToConverge}");
            EditorGUILayout.LabelField($"  Constraint Energy: {debug.constraintEnergy:F6}");

            if (debug.degenerateCount > 0) {
                EditorGUILayout.LabelField($"  Degenerate: {debug.degenerateCount}", GetWarningStyle());
            }
            if (debug.nanInfCount > 0) {
                EditorGUILayout.LabelField($"  NaN/Inf: {debug.nanInfCount}", GetErrorStyle());
            }
            if (debug.plasticFlowCount > 0) {
                EditorGUILayout.LabelField($"  Plastic Flow Events: {debug.plasticFlowCount}", GetInfoStyle());
            }

            EditorGUILayout.Space(5);
        }
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

    GUIStyle GetInfoStyle() {
        var style = new GUIStyle(EditorStyles.label);
        style.normal.textColor = new Color(0.2f, 0.6f, 1f);
        return style;
    }
}
