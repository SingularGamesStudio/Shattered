using Unity.Mathematics;

namespace GPU.Delaunay {
    public static partial class DTBuilder {
        // Sign conventions here are intentionally matched: Orient(...) and InCircle(...) must agree.
        // This is the same convention used in known-good ports.
        static bool Orient(double px, double py, double qx, double qy, double rx, double ry) =>
            (qy - py) * (rx - qx) - (qx - px) * (ry - qy) < 0.0;

        static bool InCircle(double ax, double ay, double bx, double by, double cx, double cy, double px, double py) {
            double dx = ax - px;
            double dy = ay - py;
            double ex = bx - px;
            double ey = by - py;
            double fx = cx - px;
            double fy = cy - py;

            double ap = dx * dx + dy * dy;
            double bp = ex * ex + ey * ey;
            double cp = fx * fx + fy * fy;

            return dx * (ey * cp - bp * fy) -
                   dy * (ex * cp - bp * fx) +
                   ap * (ex * fy - ey * fx) < 0.0;
        }

        static double Circumradius(double ax, double ay, double bx, double by, double cx, double cy) {
            double dx = bx - ax;
            double dy = by - ay;
            double ex = cx - ax;
            double ey = cy - ay;
            double bl = dx * dx + dy * dy;
            double cl = ex * ex + ey * ey;
            double d = 0.5 / (dx * ey - dy * ex);
            double x = (ey * bl - dy * cl) * d;
            double y = (dx * cl - ex * bl) * d;
            return x * x + y * y;
        }

        static void Circumcenter(double ax, double ay, double bx, double by, double cx, double cy, out double x, out double y) {
            double dx = bx - ax;
            double dy = by - ay;
            double ex = cx - ax;
            double ey = cy - ay;
            double bl = dx * dx + dy * dy;
            double cl = ex * ex + ey * ey;
            double d = 0.5 / (dx * ey - dy * ex);
            x = ax + (ey * bl - dy * cl) * d;
            y = ay + (dx * cl - ex * bl) * d;
        }

        static double PseudoAngle(double dx, double dy) {
            double p = dx / (math.abs(dx) + math.abs(dy));
            return (dy > 0 ? 3 - p : 1 + p) / 4;
        }
    }
}
