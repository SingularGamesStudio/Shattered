#if UNITY_EDITOR
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

[CustomEditor(typeof(SpriteMesh))]
public sealed class SpriteMeshEditor : Editor {
    Texture2D lastSeenMaterialMap;

    public override void OnInspectorGUI() {
        SpriteMesh spriteMesh = (SpriteMesh)target;
        serializedObject.Update();

        SerializedProperty mapProp = serializedObject.FindProperty("materialMap");
        SerializedProperty assignmentsProp = serializedObject.FindProperty("materialMapAssignments");

        DrawPropertiesExcluding(serializedObject, "m_Script", "materialMapAssignments");

        Texture2D currentMap = mapProp != null ? (Texture2D)mapProp.objectReferenceValue : null;
        if (currentMap != lastSeenMaterialMap) {
            serializedObject.ApplyModifiedProperties();
            serializedObject.Update();
            SyncAssignmentsFromMap(spriteMesh);
            lastSeenMaterialMap = currentMap;
            serializedObject.Update();
            assignmentsProp = serializedObject.FindProperty("materialMapAssignments");
        }

        DrawMaterialMapAssignments(spriteMesh, assignmentsProp, currentMap);

        serializedObject.ApplyModifiedProperties();
    }

    void DrawMaterialMapAssignments(SpriteMesh spriteMesh, SerializedProperty assignmentsProp, Texture2D currentMap) {
        EditorGUILayout.Space();
        EditorGUILayout.LabelField("Material Map Colors", EditorStyles.boldLabel);

        if (currentMap == null) {
            EditorGUILayout.HelpBox("Assign materialMap to map sprite colors to different point materials.", MessageType.Info);
            return;
        }

        EditorGUILayout.BeginHorizontal();
        if (GUILayout.Button("Rescan Colors")) {
            SyncAssignmentsFromMap(spriteMesh);
            serializedObject.Update();
            assignmentsProp = serializedObject.FindProperty("materialMapAssignments");
        }

        if (GUILayout.Button("Generate")) {
            spriteMesh.Generate(0, 0);
            EditorUtility.SetDirty(spriteMesh);
        }
        EditorGUILayout.EndHorizontal();

        if (assignmentsProp == null || assignmentsProp.arraySize == 0) {
            EditorGUILayout.HelpBox("No colors were found in materialMap.", MessageType.Warning);
            return;
        }

        for (int i = 0; i < assignmentsProp.arraySize; i++) {
            SerializedProperty element = assignmentsProp.GetArrayElementAtIndex(i);
            SerializedProperty colorProp = element.FindPropertyRelative("color");
            SerializedProperty materialProp = element.FindPropertyRelative("material");

            EditorGUILayout.BeginHorizontal();

            using (new EditorGUI.DisabledScope(true)) {
                Color32 c32 = (Color32)colorProp.colorValue;
                EditorGUILayout.ColorField(new GUIContent(), c32, true, false, false, GUILayout.MaxWidth(64f));
                EditorGUILayout.TextField($"#{c32.r:X2}{c32.g:X2}{c32.b:X2}{c32.a:X2}", GUILayout.MaxWidth(96f));
            }

            EditorGUILayout.PropertyField(materialProp, GUIContent.none);
            EditorGUILayout.EndHorizontal();
        }

        EditorGUILayout.HelpBox("Unassigned colors fall back to base/default material.", MessageType.None);
    }

    static void SyncAssignmentsFromMap(SpriteMesh spriteMesh) {
        if (spriteMesh == null)
            return;

        Undo.RecordObject(spriteMesh, "Sync Material Map Colors");

        Color32[] colors = spriteMesh.CollectMaterialMapColors();
        var previous = new Dictionary<Color32, MaterialDef>();
        if (spriteMesh.materialMapAssignments != null) {
            for (int i = 0; i < spriteMesh.materialMapAssignments.Count; i++) {
                SpriteMesh.ColorMaterialAssignment a = spriteMesh.materialMapAssignments[i];
                previous[a.color] = a.material;
            }
        }

        var rebuilt = new List<SpriteMesh.ColorMaterialAssignment>(colors.Length);
        for (int i = 0; i < colors.Length; i++) {
            Color32 color = colors[i];
            previous.TryGetValue(color, out MaterialDef assigned);

            rebuilt.Add(new SpriteMesh.ColorMaterialAssignment {
                color = color,
                material = assigned
            });
        }

        spriteMesh.materialMapAssignments = rebuilt;
        EditorUtility.SetDirty(spriteMesh);
    }
}
#endif
