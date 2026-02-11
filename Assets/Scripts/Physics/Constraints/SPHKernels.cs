using Unity.Mathematics;

namespace Physics
{

    /// <summary>
    /// Wendland C2 kernel (2D) with compact support radius 2h.
    /// Used by XPBI-style corrected kernel operators for stable ∇v estimation on irregular samples.
    /// </summary>
    public static class SPHKernels
    {

        // 2D normalization for Wendland C2 with support 2h.
        // W(q) = α (1 - q/2)^4 (2q + 1), q = r/h, q ∈ [0, 2].
        static float Alpha2D(float h) => 7f / (4f * math.PI * h * h);

        public static float WendlandC2(float r, float h)
        {
            if (h <= Const.Eps) return 0f;

            float q = r / h;
            if (q >= 2f) return 0f;

            float s = 1f - 0.5f * q;
            float s2 = s * s;
            float s4 = s2 * s2;

            return Alpha2D(h) * s4 * (2f * q + 1f);
        }

        /// <summary>
        /// ∇_{x_i} W(x_j - x_i, h).
        /// With xij = xj - xi and r = |xij|:
        /// ∇_{x_i} W = dW/dr * (x_i - x_j)/r = -(dW/dr) * xij / r.
        /// </summary>
        public static float2 GradWendlandC2(float2 xij, float h)
        {
            float r2 = math.lengthsq(xij);
            if (h <= Const.Eps || r2 <= Const.Eps * Const.Eps) return float2.zero;

            float r = math.sqrt(r2);
            float q = r / h;
            if (q >= 2f) return float2.zero;

            float alpha = Alpha2D(h);

            float s = 1f - 0.5f * q;
            float s2 = s * s;
            float s3 = s2 * s;
            float s4 = s2 * s2;

            // d/dq [s^4 (2q+1)], s = 1 - 0.5 q
            // = 4 s^3 (ds/dq)(2q+1) + s^4 * 2, ds/dq = -0.5
            float dFdq = 4f * s3 * (-0.5f) * (2f * q + 1f) + s4 * 2f;

            // dW/dr = α * dF/dq * (1/h)
            float dWdr = alpha * dFdq / h;

            // ∇_{x_i} W = -(dW/dr) * xij / r
            return -(dWdr / r) * xij;
        }
    }
}
