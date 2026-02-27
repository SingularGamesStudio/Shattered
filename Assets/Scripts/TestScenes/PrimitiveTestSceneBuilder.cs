using Unity.Mathematics;
using UnityEngine;

public sealed class PrimitiveTestSceneBuilder : MonoBehaviour {
    public enum Scenario {
        SlimeHang2Points = 1,
        SlimeWind = 2,
        CantileverSeries = 3,
        MultiForceReadbackStress = 4,
        SlimeDropClusteredBoxes = 5,
    }

    [Header("Setup")]
    public Scenario scenario = Scenario.SlimeHang2Points;
    public bool generateOnStart = true;

    [Header("Common Box")]
    public int pointCount = 450;
    public int maxLayer = 2;
    public float layerRatio = 0.12f;
    public float radiusScale = 0.85f;

    [Header("Scenario 1")]
    public Vector2 hangCenter = new Vector2(0f, 0f);
    public Vector2 hangSize = new Vector2(3.4f, 2.2f);
    public bool addBigForcePulse = true;

    [Header("Scenario 2")]
    public Vector2 windCenter = new Vector2(0f, 0f);
    public Vector2 windSize = new Vector2(2.8f, 2f);

    [Header("Scenario 3")]
    [Min(2)] public int cantileverCount = 5;
    public Vector2 cantileverStart = new Vector2(-5f, 1.5f);
    public Vector2 cantileverStep = new Vector2(0f, -1.5f);
    public Vector2 cantileverSize = new Vector2(4.4f, 0.7f);
    [Range(0.01f, 0.6f)] public float fixedSideWidthFraction = 0.10f;

    [Header("Scenario 5")]
    public Vector2 dropSlimeCenter = new Vector2(0f, 2.8f);
    public Vector2 dropSlimeSize = new Vector2(2.4f, 2f);
    [Min(1)] public int obstacleCount = 5;
    [Min(16)] public int obstaclePointCount = 96;
    [Min(0f)] public float obstaclePointJitterFraction = 0.2f;
    public Vector2 obstacleClusterCenter = new Vector2(0f, 0.3f);
    public Vector2 obstacleClusterHalfSpread = new Vector2(1.1f, 0.55f);
    public Vector2 obstacleBaseSize = new Vector2(0.7f, 0.45f);
    public Vector2 obstacleSizeJitter = new Vector2(0.2f, 0.12f);
    [Range(0f, 80f)] public float obstacleMaxRotationDeg = 35f;
    [Min(1)] public uint obstacleSeed = 7;
    public bool addFloorPlane = true;
    public Vector2 floorCenter = new Vector2(0f, -2.2f);
    public Vector2 floorSize = new Vector2(8f, 1.2f);
    [Min(16)] public int floorPointCount = 180;

    void Start() {
        if (!generateOnStart)
            return;

        Build();
    }

    [ContextMenu("Build")]
    public void Build() {
        var library = MaterialLibrary.Instance;
        if (library == null) {
            Debug.LogError("PrimitiveTestSceneBuilder: MaterialLibrary.Instance is required in scene.");
            return;
        }

        switch (scenario) {
            case Scenario.SlimeHang2Points:
                BuildSlimeHang2Points(library);
                break;
            case Scenario.SlimeWind:
                BuildSlimeWind(library);
                break;
            case Scenario.CantileverSeries:
                BuildCantileverSeries(library);
                break;
            case Scenario.MultiForceReadbackStress:
                BuildMultiForceReadbackStress(library);
                break;
            case Scenario.SlimeDropClusteredBoxes:
                BuildSlimeDropClusteredBoxes(library);
                break;
        }
    }

    void BuildSlimeHang2Points(MaterialLibrary library) {
        var baseMat = GetMaterialSafe(library, 0);
        var box = CreateBox("SlimeHang2", hangCenter, hangSize, pointCount, baseMat);

        float2 topLeft = new float2(hangCenter.x - hangSize.x * 0.45f, hangCenter.y + hangSize.y * 0.45f);
        float2 topRight = new float2(hangCenter.x + hangSize.x * 0.45f, hangCenter.y + hangSize.y * 0.45f);
        box.FixClosestNode(topLeft);
        box.FixClosestNode(topRight);
        box.RecomputeMassFromDensity();

        if (addBigForcePulse) {
            var pulse = gameObject.GetComponent<PulseForceController>();
            if (pulse == null)
                pulse = gameObject.AddComponent<PulseForceController>();

            pulse.targetMeshless = box;
            pulse.worldCenter = hangCenter;
            pulse.radius = Mathf.Max(0.8f, hangSize.x * 0.5f);
            pulse.direction = new Vector2(1f, 0.15f);
            pulse.strength = 22f;
            pulse.startDelay = 0.8f;
            pulse.duration = 0.06f;
            pulse.repeat = true;
            pulse.repeatInterval = 3f;
        }
    }

    void BuildSlimeWind(MaterialLibrary library) {
        var baseMat = GetMaterialSafe(library, 0);
        var box = CreateBox("SlimeWind", windCenter, windSize, pointCount, baseMat);
        box.FixClosestNode(new float2(windCenter.x, windCenter.y + windSize.y * 0.45f));
        box.RecomputeMassFromDensity();

        var stress = gameObject.GetComponent<ReadbackMultiForceController>();
        if (stress == null)
            stress = gameObject.AddComponent<ReadbackMultiForceController>();

        stress.targetMeshless = box;
        stress.pullStrength = 0.1f;
        stress.cornerBoost = 0.02f;
        stress.oscillationAmplitude = 0.1f;
        stress.oscillationFrequency = 1.1f;
    }

    void BuildMultiForceReadbackStress(MaterialLibrary library) {
        var baseMat = GetMaterialSafe(library, 0);
        var box = CreateBox("MultiForceStress", windCenter, windSize, pointCount, baseMat);

        box.FixClosestNode(new float2(windCenter.x, windCenter.y + windSize.y * 0.46f));
        box.RecomputeMassFromDensity();

        var stress = gameObject.GetComponent<ReadbackMultiForceController>();
        if (stress == null)
            stress = gameObject.AddComponent<ReadbackMultiForceController>();

        stress.targetMeshless = box;
        stress.pullStrength = 0.04f;
        stress.cornerBoost = 0.03f;
        stress.oscillationAmplitude = 0.5f;
        stress.oscillationFrequency = 1.35f;
    }

    void BuildCantileverSeries(MaterialLibrary library) {
        var seed = GetMaterialSafe(library, 0);

        for (int i = 0; i < cantileverCount; i++) {
            float t = cantileverCount <= 1 ? 0f : i / (float)(cantileverCount - 1);

            MaterialDef runtimeMat = CreateRuntimeVariant(seed, t, i);
            int materialIndex = library.AddRuntimeMaterial(runtimeMat);
            var materialRef = library.materials[materialIndex];

            Vector2 center = cantileverStart + cantileverStep * i;
            var box = CreateBox($"Cantilever_{i}", center, cantileverSize, pointCount, materialRef);
            FixLeftSide(box, fixedSideWidthFraction);
            box.RecomputeMassFromDensity();
        }
    }

    void BuildSlimeDropClusteredBoxes(MaterialLibrary library) {
        var baseMat = GetMaterialSafe(library, 0);
        var obstacleMat = GetMaterialSafe(library, 1);

        var slime = CreateBox("SlimeDropBlob", dropSlimeCenter, dropSlimeSize, pointCount, baseMat);
        slime.RecomputeMassFromDensity();

        var rnd = new Unity.Mathematics.Random(obstacleSeed == 0 ? 1u : obstacleSeed);
        for (int i = 0; i < obstacleCount; i++) {
            float2 centerJitter = new float2(
                rnd.NextFloat(-obstacleClusterHalfSpread.x, obstacleClusterHalfSpread.x),
                rnd.NextFloat(-obstacleClusterHalfSpread.y, obstacleClusterHalfSpread.y)
            );

            Vector2 center = obstacleClusterCenter + new Vector2(centerJitter.x, centerJitter.y);
            Vector2 size = new Vector2(
                Mathf.Max(0.12f, obstacleBaseSize.x + rnd.NextFloat(-obstacleSizeJitter.x, obstacleSizeJitter.x)),
                Mathf.Max(0.12f, obstacleBaseSize.y + rnd.NextFloat(-obstacleSizeJitter.y, obstacleSizeJitter.y))
            );

            int pointJitter = Mathf.RoundToInt(obstaclePointCount * obstaclePointJitterFraction);
            int points = Mathf.Max(16, obstaclePointCount + rnd.NextInt(-pointJitter, pointJitter + 1));
            float rotationDeg = rnd.NextFloat(-obstacleMaxRotationDeg, obstacleMaxRotationDeg);

            CreateBox($"DropObstacle_{i}", center, size, points, obstacleMat, true, rotationDeg);
        }

        if (addFloorPlane)
            CreateBox("DropFloor", floorCenter, floorSize, floorPointCount, obstacleMat, true);
    }

    MaterialDef CreateRuntimeVariant(MaterialDef seed, float t, int index) {
        var def = ScriptableObject.CreateInstance<MaterialDef>();
        def.name = $"RuntimeCantileverMaterial_{index}";
        def.hideFlags = HideFlags.DontSave;
        def.sprite = seed != null ? seed.sprite : null;

        MaterialParams src = seed != null ? seed.physical : new MaterialParams {
            youngModulus = 15000f,
            poissonRatio = 0.3f,
            yieldHencky = 0.08f,
            volumetricHenckyLimit = 0.35f,
            density = 1f,
        };

        src.youngModulus = Mathf.Lerp(7000f, 70000f, t);
        src.poissonRatio = Mathf.Lerp(0.22f, 0.42f, t);
        src.yieldHencky = Mathf.Lerp(0.04f, 0.12f, t);
        src.volumetricHenckyLimit = Mathf.Lerp(0.2f, 0.45f, t);
        src.density = Mathf.Lerp(0.5f, 2.2f, t);
        def.physical = src;

        return def;
    }

    Box CreateBox(string objectName, Vector2 center, Vector2 size, int points, MaterialDef baseMaterial, bool fixedObject = false, float rotationDeg = 0f) {
        var go = new GameObject(objectName);
        go.transform.SetParent(transform, false);
        go.transform.position = new Vector3(center.x, center.y, 0f);

        var box = go.AddComponent<Box>();
        box.size = new float2(size.x, size.y);
        box.pointCount = points;
        box.maxLayer = maxLayer;
        box.layerRatio = layerRatio;
        box.radiusScale = radiusScale;
        box.defaultAnchorMode = Box.AnchorMode.None;
        box.baseMaterialDef = baseMaterial;
        box.fixedObject = fixedObject;
        box.generateOnStart = false;
        box.Generate(points, 0);

        if (Mathf.Abs(rotationDeg) > 0.001f) {
            RotateNodes(box, center, rotationDeg);
            box.Build();
        }

        return box;
    }

    static void RotateNodes(Box box, Vector2 center, float rotationDeg) {
        if (box == null || box.nodes == null || box.nodes.Count == 0)
            return;

        float radians = rotationDeg * Mathf.Deg2Rad;
        float s = Mathf.Sin(radians);
        float c = Mathf.Cos(radians);
        float2 pivot = new float2(center.x, center.y);

        for (int i = 0; i < box.nodes.Count; i++) {
            var node = box.nodes[i];
            float2 d = node.pos - pivot;
            float2 rotated = new float2(d.x * c - d.y * s, d.x * s + d.y * c) + pivot;
            node.pos = rotated;
            node.originalPos = rotated;
            box.nodes[i] = node;
        }
    }

    static void FixLeftSide(Box box, float fraction) {
        if (box == null || box.nodes == null || box.nodes.Count == 0)
            return;

        float minX = box.nodes[0].pos.x;
        float maxX = minX;

        for (int i = 1; i < box.nodes.Count; i++) {
            float x = box.nodes[i].pos.x;
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
        }

        float threshold = minX + (maxX - minX) * Mathf.Clamp01(fraction);
        for (int i = 0; i < box.nodes.Count; i++) {
            if (box.nodes[i].pos.x <= threshold)
                box.FixNode(i);
        }
    }

    static MaterialDef GetMaterialSafe(MaterialLibrary library, int index) {
        if (library == null || library.materials == null || library.materials.Length == 0)
            return null;

        if (index < 0 || index >= library.materials.Length)
            return library.materials[0];

        return library.materials[index];
    }
}
