using UnityEngine;

[CreateAssetMenu(menuName = "Shattered/Meshless Material", fileName = "MeshlessMaterialDef")]
public sealed class MaterialDef : ScriptableObject {
    [Header("Rendering")]
    public Sprite sprite;

    [Header("Solver params (used by XPBI)")]
    public MaterialParams physical;
}

[System.Serializable]
public struct MaterialParams {
    public float youngModulus;
    public float poissonRatio;
    public float yieldHencky;
    public float volumetricHenckyLimit;
}
