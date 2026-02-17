using UnityEngine;

[CreateAssetMenu(menuName = "Shattered/Meshless Material", fileName = "MeshlessMaterialDef")]
public sealed class MaterialDef : ScriptableObject {
    [Header("Rendering")]
    public Sprite sprite;

    [Header("Physical params (not used by renderer yet)")]
    public MaterialParams physical;
}

[System.Serializable]
public struct MaterialParams {
    public float density;
    public float youngModulus;
    public float poissonRatio;

    public float friction;
    public float restitution;
    public float damping;
}
