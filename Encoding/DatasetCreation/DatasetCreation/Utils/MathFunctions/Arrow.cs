namespace DatasetCreation.Utils.MathFunctions
{
    internal class Arrow : IMathFunction
    {
        float h = 5e-5f;

        float m = 0.01f;

        public float mu { get; set; }

        public float sigma { get; set; }

        public float b1 { get; set; }

        public float b2 { get; set; }

        public void Compute1D() { return; }
        public void Compute2D()
        {
            for (int i = 0; i < 3; i++)
            {
                Func<float, float, float, float> alpha = (x, y, eps) => (float)Math.Exp(-(y - x) * (y - x) / eps);
                Func<float, float, float> rho = (z, eps) => z - ((float)Math.Exp((z - 1.0f) / eps) - (float)(Math.Exp(-1.0f / eps) / (1.0 - Math.Exp(-1.0f / eps))));
                Func<float, float, float> rho180 = (z, eps) => 1.0f - z - ((float)Math.Exp(-z / eps) - (float)(Math.Exp(-1.0f / eps) / (1.0 - Math.Exp(-1.0f / eps))));
                Func<float, float, float> delta = (z, eps) => 1.0f + (float)(-Math.Exp(-z / eps) + Math.Exp(-1.0f / eps) - Math.Exp(-(1.0f - z) / eps));
                Func<float, float, float> delta180 = (z, eps) => 1.0f + (float)(-Math.Exp(-(1.0f - z) / eps) + Math.Exp(-1 / eps) - Math.Exp(-z / eps));

                Func<float, float, float> uh = (x, y) =>
                    (alpha(x, y, m) + rho(x, m) * rho(y, m)) * delta(x, m) * delta(y, m);

                Evaluator.EvalAndSave2DFunction((float x, float y) =>
                {
                    float u = uh(x, y);
                    float u_px = uh(x + h, y);
                    float u_mx = uh(x - h, y);
                    float u_py = uh(x, y + h);
                    float u_my = uh(x, y - h);

                    float dx_u = (u_px - u_mx) / (2.0f * h);
                    float dy_u = (u_py - u_my) / (2.0f * h);
                    float dxx_u = (u_px - 2.0f * u + u_mx) / (h * h);
                    float dyy_u = (u_py - 2.0f * u + u_my) / (h * h);

                    return -mu * (dxx_u + dyy_u) + b1 * dx_u + b2 * dy_u + sigma * u;
                });
                Evaluator.EvalAndSave2DFunction((float x, float y) =>
                {
                    float u = uh(x, y);
                    float u_px = uh(x + h, y);
                    float u_mx = uh(x - h, y);
                    float u_py = uh(x, y + h);
                    float u_my = uh(x, y - h);

                    float dx_u = (u_px - u_mx) / (2.0f * h);
                    float dy_u = (u_py - u_my) / (2.0f * h);
                    float dxx_u = (u_px - 2.0f * u + u_mx) / (h * h);
                    float dyy_u = (u_py - 2.0f * u + u_my) / (h * h);

                    return mu * (dxx_u + dyy_u) - b1 * dx_u - b2 * dy_u - sigma * u;
                });

                Func<float, float, float> uh180 = (x, y) =>
                    (alpha(x, y, m) + rho180(x, m) * rho180(y, m)) * delta180(x, m) * delta180(y, m);

                Evaluator.EvalAndSave2DFunction((float x, float y) =>
                {
                    float u = uh180(x, y);
                    float u_px = uh180(x + h, y);
                    float u_mx = uh180(x - h, y);
                    float u_py = uh180(x, y + h);
                    float u_my = uh180(x, y - h);

                    float dx_u = (u_px - u_mx) / (2.0f * h);
                    float dy_u = (u_py - u_my) / (2.0f * h);
                    float dxx_u = (u_px - 2.0f * u + u_mx) / (h * h);
                    float dyy_u = (u_py - 2.0f * u + u_my) / (h * h);

                    return -mu * (dxx_u + dyy_u) + b1 * dx_u + b2 * dy_u + sigma * u;
                });
                Evaluator.EvalAndSave2DFunction((float x, float y) =>
                {
                    float u = uh180(x, y);
                    float u_px = uh180(x + h, y);
                    float u_mx = uh180(x - h, y);
                    float u_py = uh180(x, y + h);
                    float u_my = uh180(x, y - h);

                    float dx_u = (u_px - u_mx) / (2.0f * h);
                    float dy_u = (u_py - u_my) / (2.0f * h);
                    float dxx_u = (u_px - 2.0f * u + u_mx) / (h * h);
                    float dyy_u = (u_py - 2.0f * u + u_my) / (h * h);

                    return mu * (dxx_u + dyy_u) - b1 * dx_u - b2 * dy_u - sigma * u;
                });

                m *= 1.3f;
            }
        }

        public static IMathFunction[] GetInstances(int n = 600, float min = 0.0f, float max = 1e2f)
        {
            float[] mus = { 5e-4f, 1e-3f, 1e-2f, 1e-1f };
            float[] sigmas = { 0, 0.5f, 1 };
            float[] bs = { -1, -0.5f, 0.0f, 0.5f, 1 };

            var instances = new IMathFunction[2 * mus.Length * sigmas.Length * bs.Length * bs.Length];
            var idx = 0;
            for (int i = 0; i < 2; i++)
            {
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
            }

            return instances;
        }
    }
}
