#if UNITY_EDITOR
using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEngine;

public static class BuildAndRunStandalone {
    const string BuildDir = "Build";
    const string ExeName = "ShatteredDev.exe";

    [MenuItem("Tools/Build/Build & Run Win64 Dev _F5")]
    public static void BuildRunWin64Dev() {
        Directory.CreateDirectory(BuildDir);

        // 1. Disable asynchronous shader compilation to make the build wait for all shaders.
        bool previousAsyncSetting = ShaderUtil.allowAsyncCompilation;
        ShaderUtil.allowAsyncCompilation = false;

        var scenes = EditorBuildSettingsScene.GetActiveSceneList(EditorBuildSettings.scenes);
        var opts = new BuildPlayerOptions {
            scenes = scenes,
            locationPathName = Path.Combine(BuildDir, ExeName),
            target = BuildTarget.StandaloneWindows64,
            options = BuildOptions.Development | BuildOptions.AllowDebugging | BuildOptions.AutoRunPlayer
        };

        BuildReport report = BuildPipeline.BuildPlayer(opts);
        Debug.Log($"Build result: {report.summary.result}, time: {report.summary.totalTime}");

        // 2. Restore the original async setting.
        ShaderUtil.allowAsyncCompilation = previousAsyncSetting;

        // 3. (Optional) Wait for any remaining compilation that might have been triggered post‑build.
        while (ShaderUtil.anythingCompiling) {
            System.Threading.Thread.Sleep(100);
        }
    }
}
#endif