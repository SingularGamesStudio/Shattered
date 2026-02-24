using System.Collections.Generic;
using GPU.Solver;
using UnityEngine;

public abstract class MeshlessForceControllerBase : MonoBehaviour, IMeshlessForceController {
    public bool IsActive => isActiveAndEnabled;

    protected virtual void OnEnable() {
        MeshlessForceControllerRegistry.Register(this);
    }

    protected virtual void OnDisable() {
        MeshlessForceControllerRegistry.Unregister(this);
    }

    public abstract void GatherForceEvents(Meshless target, float dtPerTick, int ticksToRun, List<XPBISolver.ForceEvent> eventsOut);
}
