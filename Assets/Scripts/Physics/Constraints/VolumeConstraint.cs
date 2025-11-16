using Unity.Mathematics;

namespace Physics {
    public class VolumeConstraint : Constraint {
        public float Compliance = 0f;
        public float Damping = 0f;
        public string GetConstraintType() => "VolumeConstraint";

        public void Initialise(NodeBatch data) {
            //data.CacheVolume();
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

                    // On-the-fly rest area from Fp
                    float2 xi0 = ni.originalPos;
                    float2 xj0 = nb.originalPos;
                    float2 xk0 = nc.originalPos;
                    float2 restEdge_j = math.mul(ni.Fp, xj0 - xi0);
                    float2 restEdge_k = math.mul(ni.Fp, xk0 - xi0);
                    float restVol = 0.5f * Cross(restEdge_j, restEdge_k);

                    float dLambda = -(0.5f * Cross(nb.predPos - ni.predPos, nc.predPos - ni.predPos) - restVol + alphaTilde * cache.lambdas.volume[k]) / denom;
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
            const float yieldAreaFrac = 1.05f; // stretch threshold for area

            for (int i = 0; i < data.Count; ++i) {
                var node = data.nodes[i];
                var cache = data.caches[i];
                if (cache?.neighbors == null || cache.neighbors.Count < 2) continue;

                int n = cache.neighbors.Count;
                float2 xi0 = node.originalPos;
                float2 xi = node.predPos;

                // Build local rest/cur edges: from node to neighbors
                float2[] restEdges = new float2[n];
                float2[] curEdges = new float2[n];
                for (int k = 0; k < n; ++k) {
                    int j = cache.neighbors[k];
                    float2 xj0 = data.nodes[j].originalPos;
                    float2 xj = data.nodes[j].predPos;
                    restEdges[k] = math.mul(node.Fp, xj0 - xi0);
                    curEdges[k] = xj - xi;
                }

                // Fit deformation gradient F mapping restEdges to curEdges
                float2x2 F = DeformationUtils.FitDeformationGradient(restEdges, curEdges);
                float2x2 R, S;
                float s1, s2;
                DeformationUtils.PolarDecompose2D(F, out R, out S, out s1, out s2);

                // Area stretch: product of principal stretches
                float areaStretch = s1 * s2;

                bool yielded = (areaStretch > yieldAreaFrac) || (areaStretch < 1f / yieldAreaFrac);
                if (yielded) {
                    // Return-mapping: clamp area stretch (isotropic projection)
                    float clampedAreaStretch = math.clamp(areaStretch, 1f / yieldAreaFrac, yieldAreaFrac);
                    float proj_s = math.sqrt(clampedAreaStretch);

                    // For anisotropy: adaptively clamp s1 and s2 (optional: you may want isotropic area flow only)
                    float clamped_s1 = math.clamp(s1, 1f / yieldAreaFrac, yieldAreaFrac);
                    float clamped_s2 = math.clamp(s2, 1f / yieldAreaFrac, yieldAreaFrac);

                    // Build projected S_p with same eigenvectors
                    float2x2 V;
                    float a = S.c0.x, b = S.c0.y, c = S.c1.x, d = S.c1.y;
                    if (math.abs(b) > 1e-5f) {
                        float2 v1 = math.normalize(new float2(s1 * s1 - d * d, b));
                        float2 v2 = math.normalize(new float2(s2 * s2 - d * d, b));
                        V = new float2x2(v1, v2);
                    } else {
                        V = float2x2.identity;
                    }
                    float2x2 Dcl = new float2x2(new float2(clamped_s1, 0f), new float2(0f, clamped_s2));
                    float2x2 Sp = math.mul(math.mul(V, Dcl), math.transpose(V));

                    node.Fp = math.mul(Sp, node.Fp); // update plastic gradient
                }
            }
        }

        static float Cross(in float2 a, in float2 b) => a.x * b.y - a.y * b.x;
        static float2 Perp(in float2 v) => new float2(-v.y, v.x);
    }
}
