using System.Collections.Generic;
using Unity.VisualScripting;
namespace Physics {
    public interface Constraint {
        // Initialise stores needed cached data before making an iteration step.
        // For example, it can calculate the non-deformed neighbor distances before the position shifts.
        void Initialise(List<Node> nodes) { }
        // Relax evaluates the constraint, and relaxes the positions to satisfy it.
        // It returns the updated total Lagrange Multiplier.
        void Relax(List<Node> nodes, float stiffness, float timeStep) { }
    }

    public class ConstraintCache {
        public List<int> neighbors;
        public List<float> neighborDistances;
        public List<float> neighborLambdas;
        public float tensionLambda;
        public float avgEdgeLen;
    }
}