using System.Collections.Generic;

public static class MeshlessForceControllerRegistry {
    static readonly List<IMeshlessForceController> controllers = new List<IMeshlessForceController>(32);

    public static IReadOnlyList<IMeshlessForceController> Controllers => controllers;

    public static void Register(IMeshlessForceController controller) {
        if (controller == null)
            return;

        if (!controllers.Contains(controller))
            controllers.Add(controller);
    }

    public static void Unregister(IMeshlessForceController controller) {
        if (controller == null)
            return;

        controllers.Remove(controller);
    }
}
