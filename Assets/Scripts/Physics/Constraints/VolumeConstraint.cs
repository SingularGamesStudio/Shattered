using Unity.Mathematics;

namespace Physics {
    public class VolumeConstraint : Constraint {
        public float Compliance = 0f;
        public float Damping = 0f;
        public string GetConstraintType() => "VolumeConstraint";

        public void Initialise(NodeBatch data) {
            data.CacheVolume();
            data.ResetDebugData(GetConstraintType());
        }

        public void Relax(NodeBatch data, float stiffness, float timeStep) {
            float alphaTilde = stiffness / (timeStep * timeStep);
            float gammaDt = Damping * timeStep;
            string constraintType = GetConstraintType();

            for (int i = 0; i < data.Count; i++) {
                var ni = data.nodes[i];
                var cache = data.caches[i];
                var debug = data.GetOrCreateDebugData(i, constraintType);
                float wi = ni.isFixed ? 0f : ni.invMass;

                for (int k = 0; k < cache.neighbors.Count; k++) {
                    float2 oldPos = ni.predPos;

                    var nb = data.nodes[cache.neighbors[k]];
                    var nc = data.nodes[cache.neighbors[(k + 1) % cache.neighbors.Count]];
                    float wj = nb.isFixed ? 0f : nb.invMass;
                    float wk = nc.isFixed ? 0f : nc.invMass;

                    float2 gA = 0.5f * Perp(nb.predPos - nc.predPos);
                    float2 gB = 0.5f * Perp(nc.predPos - ni.predPos);
                    float2 gC = 0.5f * Perp(ni.predPos - nb.predPos);

                    float denom = wi * math.lengthsq(gA) + wj * math.lengthsq(gB) + wk * math.lengthsq(gC) + alphaTilde + gammaDt;

                    if (denom <= Const.Eps) {
                        debug.degenerateCount++;
                        continue;
                    }

                    float restVol = cache.leafVolumes[k];

                    float dLambda = -(0.5f * Cross(nb.predPos - ni.predPos, nc.predPos - ni.predPos) - restVol +
                                     alphaTilde * cache.lambdas.volume[k]) / denom;

                    if (float.IsNaN(dLambda) || float.IsInfinity(dLambda)) {
                        debug.nanInfCount++;
                        continue;
                    }

                    ni.predPos -= wi * dLambda * gA;
                    nb.predPos -= wj * dLambda * gB;
                    nc.predPos -= wk * dLambda * gC;

                    cache.lambdas.volume[k] += dLambda;
                    debug.RecordPositionUpdate(ni.predPos - oldPos);
                }
            }

            data.FinalizeDebugData(constraintType);
        }

        // XPBI plastic flow for area/volume constraint
        public void PlasticFlow(NodeBatch data, float dt) {
            const float yieldAreaFrac = 0.03f; // e.g., allow 7% excess area before plasticity

            for (int i = 0; i < data.Count; ++i) {
                var node = data.nodes[i];
                var cache = data.caches[i];
                if (cache?.neighbors == null) continue;

                int n = cache.neighbors.Count;
                float2 areaPlasticShift = float2.zero;
                int numPlastic = 0;

                var pi = node.plasticReferencePos;

                for (int k = 0; k < n; ++k) {
                    int j0 = cache.neighbors[k];
                    int j1 = cache.neighbors[(k + 1) % n];
                    var pj0 = data.nodes[j0].plasticReferencePos;
                    var pj1 = data.nodes[j1].plasticReferencePos;
                    float restVol = 0.5f * Cross(pj0 - pi, pj1 - pi);

                    var ni = node.predPos;
                    var nb = data.nodes[j0].predPos;
                    var nc = data.nodes[j1].predPos;
                    float curVol = 0.5f * Cross(nb - ni, nc - ni);
                    float relAreaChange = (curVol - restVol) / (math.abs(restVol) + Const.Eps);

                    if (math.abs(relAreaChange) > yieldAreaFrac) {
                        // Plastic shift towards current config in case of yielding
                        areaPlasticShift += (ni - pi) * 0.3f;
                        numPlastic++;
                    }
                }
                if (numPlastic > 0) {
                    node.plasticReferencePos += areaPlasticShift / numPlastic;
                }
            }
        }

        static float Cross(in float2 a, in float2 b) => a.x * b.y - a.y * b.x;
        static float2 Perp(in float2 v) => new float2(-v.y, v.x);
    }
}