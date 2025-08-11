using System.Collections;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;

[CustomEditor(typeof(Box), editorForChildClasses: true)]
public class BoxGenerator: UnityEditor.Editor
{
	private Box Target => target as Box;

	public override VisualElement CreateInspectorGUI()
	{
		var root = new VisualElement();

		root.Add(new IMGUIContainer(base.OnInspectorGUI));

		var generateButton = new Button(() =>
		{
			Target.Generate(Target.pointCount, 0);
			EditorUtility.SetDirty(Target);
		}) { text = "Generate" };

		root.Add(generateButton);

		return root;
	}
}
