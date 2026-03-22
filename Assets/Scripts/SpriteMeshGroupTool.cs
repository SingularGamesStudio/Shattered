using UnityEngine;

[DisallowMultipleComponent]
public sealed class SpriteMeshGroupTool : MonoBehaviour {
    [Header("Scope")]
    public bool includeInactiveChildren = true;

    [Header("Generation")]
    public bool regenerateExistingOnGenerate = true;

    [Header("Material")]
    public MaterialDef materialTemplate;
    public bool overwriteMaterialOnExisting = true;

    [Header("Collision Grouping")]
    public bool applyCollisionOwnerOverride = true;
    public Transform collisionOwnerOverride;
    public bool fallbackToThisTransform = true;
}
