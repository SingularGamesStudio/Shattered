using UnityEditor;
using UnityEditor.UIElements;
using UnityEngine;
using UnityEngine.UIElements;

[CustomEditor(typeof(Box), editorForChildClasses: true)]
public class BoxGenerator : Editor {
    Box Target => target as Box;

    public override VisualElement CreateInspectorGUI() {
        var root = new VisualElement();

        root.Add(new IMGUIContainer(base.OnInspectorGUI));

        root.Add(new Button(() => {
            Undo.RecordObject(Target, "Generate Box");
            Target.Generate(Target.pointCount, 0);
            EditorUtility.SetDirty(Target);
        }) { text = "Generate" });

        return root;
    }
}