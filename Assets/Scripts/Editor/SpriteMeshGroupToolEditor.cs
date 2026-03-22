#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(SpriteMeshGroupTool))]
public sealed class SpriteMeshGroupToolEditor : Editor {
    SpriteMeshGroupTool Tool => (SpriteMeshGroupTool)target;

    public override void OnInspectorGUI() {
        DrawDefaultInspector();

        EditorGUILayout.Space();

        if (GUILayout.Button("Generate / Add SpriteMesh On Child Sprites")) {
            GenerateOrUpdateChildren(Tool);
        }

        if (GUILayout.Button("Set Material On Existing Child SpriteMesh")) {
            ApplyMaterialToExisting(Tool);
        }

        if (GUILayout.Button("Set Collision Owner Override On Existing Child SpriteMesh")) {
            ApplyCollisionOwnerOverrideToExisting(Tool);
        }
    }

    static void GenerateOrUpdateChildren(SpriteMeshGroupTool tool) {
        if (tool == null)
            return;

        SpriteRenderer[] renderers = tool.GetComponentsInChildren<SpriteRenderer>(tool.includeInactiveChildren);
        int undoGroup = Undo.GetCurrentGroup();
        Undo.SetCurrentGroupName("Generate Child SpriteMesh");

        int added = 0;
        int regenerated = 0;
        int skipped = 0;

        for (int i = 0; i < renderers.Length; i++) {
            SpriteRenderer sr = renderers[i];
            if (sr == null || sr.sprite == null) {
                skipped++;
                continue;
            }

            GameObject go = sr.gameObject;
            SpriteMesh sm = go.GetComponent<SpriteMesh>();
            bool isNew = sm == null;

            if (isNew) {
                Meshless existingMeshless = go.GetComponent<Meshless>();
                if (existingMeshless != null) {
                    skipped++;
                    continue;
                }

                sm = Undo.AddComponent<SpriteMesh>(go);
                added++;
            }

            Undo.RecordObject(sm, "Configure SpriteMesh");
            sm.spriteRenderer = sr;

            if (tool.applyCollisionOwnerOverride) {
                sm.collisionOwnerOverride = ResolveCollisionOwnerOverride(tool);
            }

            if (tool.materialTemplate != null && (isNew || tool.overwriteMaterialOnExisting)) {
                sm.physicsTemplate = tool.materialTemplate;
                sm.baseMaterialDef = tool.materialTemplate;
            }

            if (isNew || tool.regenerateExistingOnGenerate) {
                sm.Generate(0, 0);
                regenerated++;
            }

            EditorUtility.SetDirty(sm);
        }

        Undo.CollapseUndoOperations(undoGroup);
        EditorUtility.SetDirty(tool);
        MarkSceneDirty(tool.gameObject.scene);

        Debug.Log($"SpriteMeshGroupTool: added {added}, regenerated {regenerated}, skipped {skipped} child sprite objects.", tool);
    }

    static void ApplyMaterialToExisting(SpriteMeshGroupTool tool) {
        if (tool == null)
            return;

        if (tool.materialTemplate == null) {
            Debug.LogWarning("SpriteMeshGroupTool: materialTemplate is null.", tool);
            return;
        }

        SpriteMesh[] meshes = tool.GetComponentsInChildren<SpriteMesh>(tool.includeInactiveChildren);
        int undoGroup = Undo.GetCurrentGroup();
        Undo.SetCurrentGroupName("Apply Material To Child SpriteMesh");

        int updated = 0;
        for (int i = 0; i < meshes.Length; i++) {
            SpriteMesh sm = meshes[i];
            if (sm == null)
                continue;

            Undo.RecordObject(sm, "Set SpriteMesh Material Template");
            sm.physicsTemplate = tool.materialTemplate;
            sm.baseMaterialDef = tool.materialTemplate;
            EditorUtility.SetDirty(sm);
            updated++;
        }

        Undo.CollapseUndoOperations(undoGroup);
        EditorUtility.SetDirty(tool);
        MarkSceneDirty(tool.gameObject.scene);

        Debug.Log($"SpriteMeshGroupTool: applied material template to {updated} SpriteMesh components.", tool);
    }

    static void ApplyCollisionOwnerOverrideToExisting(SpriteMeshGroupTool tool) {
        if (tool == null)
            return;

        if (!tool.applyCollisionOwnerOverride) {
            Debug.LogWarning("SpriteMeshGroupTool: applyCollisionOwnerOverride is disabled.", tool);
            return;
        }

        Transform overrideTarget = ResolveCollisionOwnerOverride(tool);
        SpriteMesh[] meshes = tool.GetComponentsInChildren<SpriteMesh>(tool.includeInactiveChildren);
        int undoGroup = Undo.GetCurrentGroup();
        Undo.SetCurrentGroupName("Apply Collision Owner Override To Child SpriteMesh");

        int updated = 0;
        for (int i = 0; i < meshes.Length; i++) {
            SpriteMesh sm = meshes[i];
            if (sm == null)
                continue;

            Undo.RecordObject(sm, "Set SpriteMesh Collision Owner Override");
            sm.collisionOwnerOverride = overrideTarget;
            EditorUtility.SetDirty(sm);
            updated++;
        }

        Undo.CollapseUndoOperations(undoGroup);
        EditorUtility.SetDirty(tool);
        MarkSceneDirty(tool.gameObject.scene);

        Debug.Log($"SpriteMeshGroupTool: applied collision owner override to {updated} SpriteMesh components.", tool);
    }

    static Transform ResolveCollisionOwnerOverride(SpriteMeshGroupTool tool) {
        if (tool == null)
            return null;

        if (tool.collisionOwnerOverride != null)
            return tool.collisionOwnerOverride;

        return tool.fallbackToThisTransform ? tool.transform : null;
    }

    static void MarkSceneDirty(UnityEngine.SceneManagement.Scene scene) {
        if (!scene.IsValid())
            return;
        UnityEditor.SceneManagement.EditorSceneManager.MarkSceneDirty(scene);
    }
}
#endif
