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

    // Thresholds
    private float positionDeltaThreshold = 0.01f;
    private float absCThreshold = 0.01f;
    private float absDeltaLambdaThreshold = 0.001f;

    private bool debugMode = false;

    // View options
    private bool showNodeInfo = true;
    private bool showDeformationInfo = true;
    private bool showNeighborInfo = false;
    private bool showLambdaInfo = true;

    private enum SortMode {
        NodeIndex,
        MaxDeltaX,
        AvgDeltaX,
        MaxAbsC,
        AvgAbsC,
        MaxAbsDeltaLambda,
        MaxAbsLambda,
        MinDenominator,
        DegenerateCount,
        NaNCount,
        PlasticFlowCount
    }

    private SortMode sortMode = SortMode.MaxAbsC;
    private bool sortDescending = true;

    [MenuItem("Window/Constraint Debugger (XPBI)")]
    public static void ShowWindow() {
        GetWindow<ConstraintDebugWindow>("XPBI Debug");
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
        EditorGUILayout.LabelField("XPBI Constraint Debugger", EditorStyles.boldLabel);

        scrollPos = EditorGUILayout.BeginScrollView(scrollPos);

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

        var batch = targetMeshless.lastBatchDebug;
        if (batch == null) {
            EditorGUILayout.EndScrollView();
            EditorGUILayout.HelpBox("Node batch not available", MessageType.Warning);
            return;
        }

        DrawControlPanel();
        EditorGUILayout.Space();

        DrawGlobalStats(batch);
        EditorGUILayout.Space();

        DrawNodeList(batch);

        EditorGUILayout.EndScrollView();
    }

    void DrawSimulationControls() {
        EditorGUILayout.BeginVertical(EditorStyles.helpBox);
        EditorGUILayout.LabelField("Simulation Controls", EditorStyles.boldLabel);

        debugMode = EditorGUILayout.Toggle("Debug Mode (1 Iter/Frame)", debugMode);
        Const.Iterations = debugMode ? 1 : Const._defaultIterations;

        var controller = Object.FindFirstObjectByType<SimulationController>();
        if (controller != null) {
            controller.forceTPSToFPS =
                EditorGUILayout.Toggle("Debug: FPS=TPS", controller.forceTPSToFPS);
        }

        EditorGUILayout.LabelField($"Current Iterations: {Const.Iterations}");
        EditorGUILayout.EndVertical();
    }

    void DrawControlPanel() {
        EditorGUILayout.BeginVertical(EditorStyles.helpBox);
        EditorGUILayout.LabelField("Filters & View Options", EditorStyles.boldLabel);

        showHealthyNodes = EditorGUILayout.Toggle("Show Healthy Nodes", showHealthyNodes);

        positionDeltaThreshold = EditorGUILayout.Slider("Δx Threshold", positionDeltaThreshold, 0.0001f, 1f);
        absCThreshold = EditorGUILayout.Slider("|C| Threshold", absCThreshold, 0.0001f, 10f);
        absDeltaLambdaThreshold = EditorGUILayout.Slider("|Δλ| Threshold", absDeltaLambdaThreshold, 0.000001f, 1f);

        EditorGUILayout.Space(5);
        EditorGUILayout.LabelField("Detail View Options", EditorStyles.boldLabel);
        showNodeInfo = EditorGUILayout.Toggle("Show Node Info", showNodeInfo);
        showDeformationInfo = EditorGUILayout.Toggle("Show Deformation Info", showDeformationInfo);
        showNeighborInfo = EditorGUILayout.Toggle("Show Neighbor Info", showNeighborInfo);
        showLambdaInfo = EditorGUILayout.Toggle("Show XPBI (C/λ) Info", showLambdaInfo);

        EditorGUILayout.Space(5);

        EditorGUILayout.BeginHorizontal();
        EditorGUILayout.LabelField("Sort By:", GUILayout.Width(60));
        sortMode = (SortMode)EditorGUILayout.EnumPopup(sortMode);
        sortDescending = GUILayout.Toggle(sortDescending, sortDescending ? "↓" : "↑", GUILayout.Width(30));
        EditorGUILayout.EndHorizontal();

        EditorGUILayout.EndVertical();
    }

    void DrawGlobalStats(NodeBatch batch) {
        EditorGUILayout.BeginVertical(EditorStyles.helpBox);
        EditorGUILayout.LabelField("Global Statistics (XPBI)", EditorStyles.boldLabel);

        int nodesAffected = 0;
        int totalDegenerate = 0;
        int totalNaN = 0;
        int totalPlastic = 0;

        float maxDeltaX = 0f;
        float avgDeltaX = 0f;

        float maxAbsC = 0f;
        float avgAbsC = 0f;

        float maxAbsDeltaLambda = 0f;
        float maxAbsLambda = 0f;

        float totalEnergy = 0f;
        float avgIterations = 0f;
        int iterCount = 0;

        for (int i = 0; i < batch.Count; i++) {
            var d = batch.debug[i];
            totalDegenerate += d.degenerateCount;
            totalNaN += d.nanInfCount;
            totalPlastic += d.plasticFlowCount;

            maxDeltaX = Mathf.Max(maxDeltaX, d.maxPositionDelta);
            avgDeltaX += d.avgPositionDelta;

            maxAbsC = Mathf.Max(maxAbsC, d.maxAbsC);
            avgAbsC += d.avgAbsC;

            maxAbsDeltaLambda = Mathf.Max(maxAbsDeltaLambda, d.maxAbsDeltaLambda);
            maxAbsLambda = Mathf.Max(maxAbsLambda, d.maxAbsLambda);

            totalEnergy += d.constraintEnergy;

            if (d.positionUpdateCount > 0 || d.constraintEvalCount > 0) {
                nodesAffected++;
                avgIterations += d.iterationsToConverge;
                iterCount++;
            }
        }

        if (nodesAffected > 0) {
            avgDeltaX /= nodesAffected;
            avgAbsC /= nodesAffected;
            if (iterCount > 0) avgIterations /= iterCount;
        }

        EditorGUILayout.LabelField($"Nodes Affected: {nodesAffected} / {batch.Count}");

        EditorGUILayout.LabelField($"Max Δx: {maxDeltaX:F6}",
            maxDeltaX > positionDeltaThreshold ? GetErrorStyle() : EditorStyles.label);
        EditorGUILayout.LabelField($"Avg Δx: {avgDeltaX:F6}");

        EditorGUILayout.LabelField($"Max |C|: {maxAbsC:F6}",
            maxAbsC > absCThreshold ? GetErrorStyle() : EditorStyles.label);
        EditorGUILayout.LabelField($"Avg |C|: {avgAbsC:F6}");

        EditorGUILayout.LabelField($"Max |Δλ|: {maxAbsDeltaLambda:F6}",
            maxAbsDeltaLambda > absDeltaLambdaThreshold ? GetWarningStyle() : EditorStyles.label);
        EditorGUILayout.LabelField($"Max |λ|: {maxAbsLambda:F6}");

        EditorGUILayout.LabelField($"Avg Iterations (last update): {avgIterations:F2}");
        EditorGUILayout.LabelField($"Total |C| Energy (accum): {totalEnergy:F6}");

        EditorGUILayout.LabelField($"Degenerate Cases: {totalDegenerate}",
            totalDegenerate > 0 ? GetWarningStyle() : EditorStyles.label);
        EditorGUILayout.LabelField($"NaN/Inf Cases: {totalNaN}",
            totalNaN > 0 ? GetErrorStyle() : EditorStyles.label);
        EditorGUILayout.LabelField($"Plastic Flow Events: {totalPlastic}",
            totalPlastic > 0 ? GetInfoStyle() : EditorStyles.label);

        EditorGUILayout.Space(5);
        EditorGUILayout.LabelField("System Stats", EditorStyles.boldLabel);
        EditorGUILayout.LabelField($"Total Kinetic Energy: {batch.CalculateKineticEnergy():F6}");
        EditorGUILayout.LabelField($"Total Nodes: {batch.Count}");

        EditorGUILayout.EndVertical();
    }

    void DrawNodeList(NodeBatch batch) {
        var indices = Enumerable.Range(0, batch.Count).ToList();

        indices.Sort((a, b) => {
            float metricA = GetSortMetric(batch.debug[a], a);
            float metricB = GetSortMetric(batch.debug[b], b);
            int cmp = metricA.CompareTo(metricB);
            return sortDescending ? -cmp : cmp;
        });

        for (int i = 0; i < indices.Count; i++) {
            int idx = indices[i];
            var node = batch.nodes[idx];
            var cache = batch.caches[idx];
            var debug = batch.debug[idx];

            bool hasIssues = CheckNodeHasIssues(debug);
            if (!hasIssues && !showHealthyNodes) continue;

            DrawNodeEntry(idx, node, cache, debug, hasIssues);
        }
    }

    float GetSortMetric(DebugData d, int index) {
        switch (sortMode) {
            case SortMode.NodeIndex: return index;
            case SortMode.MaxDeltaX: return d.maxPositionDelta;
            case SortMode.AvgDeltaX: return d.avgPositionDelta;
            case SortMode.MaxAbsC: return d.maxAbsC;
            case SortMode.AvgAbsC: return d.avgAbsC;
            case SortMode.MaxAbsDeltaLambda: return d.maxAbsDeltaLambda;
            case SortMode.MaxAbsLambda: return d.maxAbsLambda;
            case SortMode.MinDenominator:
                return float.IsInfinity(d.minDenominator) ? 0f : d.minDenominator;
            case SortMode.DegenerateCount: return d.degenerateCount;
            case SortMode.NaNCount: return d.nanInfCount;
            case SortMode.PlasticFlowCount: return d.plasticFlowCount;
            default: return 0f;
        }
    }

    bool CheckNodeHasIssues(DebugData d) {
        return d.maxPositionDelta > positionDeltaThreshold ||
               d.maxAbsC > absCThreshold ||
               d.maxAbsDeltaLambda > absDeltaLambdaThreshold ||
               d.degenerateCount > 0 ||
               d.nanInfCount > 0;
    }

    void DrawNodeEntry(int index, Node node, NodeCache cache, DebugData d, bool hasIssues) {
        var bgColor = hasIssues ? new Color(1f, 0.85f, 0.85f) : new Color(0.85f, 1f, 0.85f);
        var oldColor = GUI.backgroundColor;
        GUI.backgroundColor = bgColor;

        EditorGUILayout.BeginVertical(EditorStyles.helpBox);
        GUI.backgroundColor = oldColor;

        bool expanded = selectedNode == index;

        string header =
            $"Node {index} | MaxΔx {d.maxPositionDelta:F4} | Max|C| {d.maxAbsC:F4} | Max|Δλ| {d.maxAbsDeltaLambda:F4}";

        if (d.degenerateCount > 0) header += $" | Degen {d.degenerateCount}";
        if (d.nanInfCount > 0) header += $" | NaN {d.nanInfCount}";
        if (d.plasticFlowCount > 0) header += $" | Plastic {d.plasticFlowCount}";

        if (GUILayout.Button(header, expanded ? EditorStyles.boldLabel : EditorStyles.label)) {
            selectedNode = expanded ? -1 : index;
        }

        if (expanded) {
            EditorGUI.indentLevel++;

            if (showNodeInfo) DrawNodeInfo(node, index);
            if (showDeformationInfo) DrawDeformationInfo(node, cache);
            if (showNeighborInfo && cache.neighbors != null) DrawNeighborInfo(index, cache);
            if (showLambdaInfo) DrawXPBIInfo(d);

            EditorGUI.indentLevel--;
        }

        EditorGUILayout.EndVertical();
    }

    void DrawXPBIInfo(DebugData d) {
        EditorGUILayout.LabelField("XPBI Debug", EditorStyles.boldLabel);

        EditorGUILayout.LabelField($"  Last C: {d.lastC:F6}");
        EditorGUILayout.LabelField($"  Avg |C|: {d.avgAbsC:F6} | Max |C|: {d.maxAbsC:F6}",
            d.maxAbsC > absCThreshold ? GetErrorStyle() : EditorStyles.label);

        EditorGUILayout.LabelField($"  Last λ: {d.lastLambda:F6} | Last Δλ: {d.lastDeltaLambda:F6}",
            Mathf.Abs(d.lastDeltaLambda) > absDeltaLambdaThreshold ? GetWarningStyle() : EditorStyles.label);

        float minDen = float.IsInfinity(d.minDenominator) ? 0f : d.minDenominator;
        EditorGUILayout.LabelField($"  Denom: last {d.lastDenominator:F6} | min {minDen:F6} | max {d.maxDenominator:F6}",
            minDen < Const.Eps ? GetWarningStyle() : EditorStyles.label);

        EditorGUILayout.LabelField($"  |∇C_vi|² proxy: last {d.lastGradCViLenSq:F6} | max {d.maxGradCViLenSq:F6}");
        EditorGUILayout.LabelField($"  Iterations to converge (last update): {d.iterationsToConverge}");
        EditorGUILayout.LabelField($"  Constraint Energy (accum |C|): {d.constraintEnergy:F6}");

        if (d.degenerateCount > 0) EditorGUILayout.LabelField($"  Degenerate: {d.degenerateCount}", GetWarningStyle());
        if (d.nanInfCount > 0) EditorGUILayout.LabelField($"  NaN/Inf: {d.nanInfCount}", GetErrorStyle());
        if (d.plasticFlowCount > 0) EditorGUILayout.LabelField($"  Plastic events: {d.plasticFlowCount}", GetInfoStyle());

        EditorGUILayout.Space(5);
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

    void DrawDeformationInfo(Node node, NodeCache cache) {
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

    void DrawNeighborInfo(int nodeIndex, NodeCache cache) {
        EditorGUILayout.LabelField($"Neighbors ({cache.neighbors.Count})", EditorStyles.boldLabel);
        for (int i = 0; i < cache.neighbors.Count; i++) {
            EditorGUILayout.LabelField($"  [{i}] Node {cache.neighbors[i]}");
        }
        EditorGUILayout.Space(5);
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
