#if UNITY_EDITOR
using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

public static class BuildAndRunStandalone {
    const string BuildDir = "Builds/Win64Dev";
    const string ExeName = "ShatteredDev.exe";

    [MenuItem("Tools/Build/Build & Run Win64 Dev _F5")]
    public static void BuildRunWin64Dev() {
        Directory.CreateDirectory(BuildDir);

        var scenes = EditorBuildSettingsScene.GetActiveSceneList(EditorBuildSettings.scenes);

        var opts = new BuildPlayerOptions {
            scenes = scenes,
            locationPathName = Path.Combine(BuildDir, ExeName),
            target = BuildTarget.StandaloneWindows64,
            options =
                BuildOptions.Development |
                BuildOptions.AllowDebugging |
                BuildOptions.AutoRunPlayer
        };

        BuildReport report = BuildPipeline.BuildPlayer(opts);
        Debug.Log($"Build result: {report.summary.result}, time: {report.summary.totalTime}");
    }
}
#endif
