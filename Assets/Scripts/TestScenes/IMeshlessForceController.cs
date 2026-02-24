using System.Collections.Generic;
using GPU.Solver;

public interface IMeshlessForceController {
    bool IsActive { get; }
    void GatherForceEvents(Meshless target, float dtPerTick, int ticksToRun, List<XPBISolver.ForceEvent> eventsOut);
}
