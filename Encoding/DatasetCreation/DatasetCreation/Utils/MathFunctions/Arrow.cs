namespace DatasetCreation.Utils.MathFunctions
{
    internal class Arrow : IMathFunction
    {
        const double h = 1e-4;

        double m = 0.01;

        public double mu { get; set; }

        public double sigma { get; set; }

        public double b1 { get; set; }

        public double b2 { get; set; }

        public void Compute1D() { return; }
        public void Compute2D()
        {
            double factor = 1.0f + Math.PI / 13.0;
            for (int i = 0; i < 2; i++)
            {
                Func<double, double, double, double> alpha = (x, y, eps) => Math.Exp(-(y - x) * (y - x) / eps);
                Func<double, double, double> rho = (z, eps) => z - (Math.Exp((z - 1.0f) / eps) - (Math.Exp(-1.0f / eps) / (1.0 - Math.Exp(-1.0f / eps))));
                Func<double, double, double> rho180 = (z, eps) => 1.0f - z - (Math.Exp(-z / eps) - (Math.Exp(-1.0f / eps) / (1.0 - Math.Exp(-1.0f / eps))));
                Func<double, double, double> delta = (z, eps) => 1.0f + (-Math.Exp(-z / eps) + Math.Exp(-1.0f / eps) - Math.Exp(-(1.0f - z) / eps));
                Func<double, double, double> delta180 = (z, eps) => 1.0f + (-Math.Exp(-(1.0f - z) / eps) + Math.Exp(-1 / eps) - Math.Exp(-z / eps));

                Func<double, double, double> uh = (x, y) =>
                    (alpha(x, y, m) + rho(x, m) * rho(y, m)) * delta(x, m) * delta(y, m);

                Evaluator.EvalAndSave2DFunction((float x, float y) =>
                {
                    double u = uh(x, y);
                    double u_px = uh(x + h, y);
                    double u_mx = uh(x - h, y);
                    double u_py = uh(x, y + h);
                    double u_my = uh(x, y - h);

                    double dx_u = (u_px - u_mx) / (2.0 * h);
                    double dy_u = (u_py - u_my) / (2.0 * h);
                    double dxx_u = (u_px - 2.0 * u + u_mx) / (h * h);
                    double dyy_u = (u_py - 2.0 * u + u_my) / (h * h);

                    return (float)(-mu * (dxx_u + dyy_u) + b1 * dx_u + b2 * dy_u + sigma * u);
                });
                Evaluator.EvalAndSave2DFunction((float x, float y) =>
                {
                    double u = uh(x, y);
                    double u_px = uh(x + h, y);
                    double u_mx = uh(x - h, y);
                    double u_py = uh(x, y + h);
                    double u_my = uh(x, y - h);

                    double dx_u = (u_px - u_mx) / (2.0 * h);
                    double dy_u = (u_py - u_my) / (2.0 * h);
                    double dxx_u = (u_px - 2.0 * u + u_mx) / (h * h);
                    double dyy_u = (u_py - 2.0 * u + u_my) / (h * h);

                    return (float)(mu * (dxx_u + dyy_u) - b1 * dx_u - b2 * dy_u - sigma * u);
                });

                Func<double, double, double> uh180 = (x, y) =>
                    (alpha(x, y, m) + rho180(x, m) * rho180(y, m)) * delta180(x, m) * delta180(y, m);

                Evaluator.EvalAndSave2DFunction((float x, float y) =>
                {
                    double u = uh180(x, y);
                    double u_px = uh180(x + h, y);
                    double u_mx = uh180(x - h, y);
                    double u_py = uh180(x, y + h);
                    double u_my = uh180(x, y - h);

                    double dx_u = (u_px - u_mx) / (2.0 * h);
                    double dy_u = (u_py - u_my) / (2.0 * h);
                    double dxx_u = (u_px - 2.0 * u + u_mx) / (h * h);
                    double dyy_u = (u_py - 2.0 * u + u_my) / (h * h);

                    return (float)(-mu * (dxx_u + dyy_u) + b1 * dx_u + b2 * dy_u + sigma * u);
                });
                Evaluator.EvalAndSave2DFunction((float x, float y) =>
                {
                    double u = uh180(x, y);
                    double u_px = uh180(x + h, y);
                    double u_mx = uh180(x - h, y);
                    double u_py = uh180(x, y + h);
                    double u_my = uh180(x, y - h);

                    double dx_u = (u_px - u_mx) / (2.0 * h);
                    double dy_u = (u_py - u_my) / (2.0 * h);
                    double dxx_u = (u_px - 2.0 * u + u_mx) / (h * h);
                    double dyy_u = (u_py - 2.0 * u + u_my) / (h * h);

                    return (float)(mu * (dxx_u + dyy_u) - b1 * dx_u - b2 * dy_u - sigma * u);
                });

                m *= factor + RandomNoise.GetRandomNoise();
            }
        }

        public static IMathFunction[] GetInstances(int n = 600, float min = 0.0f, float max = 1e2f)
        {
            double[] mus = { 5e-4f, 1e-3f, 1e-2f };
            double[] sigmas = { 0, 0.5f };
            double[] bs = { -1, -0.5f, 0.0f, 0.5f, 1 };

            var instances = new IMathFunction[mus.Length * sigmas.Length * bs.Length * bs.Length];
            var idx = 0;
            foreach (var mu in mus)
            {
                foreach (var sigma in sigmas)
                {
                    foreach (var b1 in bs)
                    {
                        foreach (var b2 in bs)
                        {
                            instances[idx++] = new Arrow
                            {
                                mu = mu,
                                sigma = sigma,
                                b1 = b1,
                                b2 = b2
                            };
                        }
                    }
                }
            }

            return instances;
        }
    }
}
