namespace DatasetCreation.Utils.MathFunctions
{
    internal class BoundaryLayerForcingTerm : IMathFunction
    {
        public double mu { get; set; }

        public double sigma { get; set; }

        public double b1 { get; set; }

        public double b2 { get; set; }

        public void Compute1D() { return; }
        public void Compute2D()
        {
            if (mu == 0.0f) return;
            double factor = 1.0f + Math.PI / 17.0;
            var mag = 10.0;
            for (int i = 0; i < 3; i++)
            {
                if (mu <= 1e-3f) return;

                Evaluator.EvalAndSave2DFunction((float x, float y) =>
                {
                    double term1 = Math.Exp(-x) * ((1.0 - y) * ((2 * mu + b1 * (1 - x)) * y + (sigma - mu) * x * y) + (2 * mu + b2 * (1.0 - 2 * y)) * x);
                    double term2 = Math.Exp(-1.0 + (x - 1.0) / mu) * ((1.0 - y) * ((2 - b1 * (1 + x / mu)) * y + (1.0 / mu - sigma) * x * y) - (2 * mu + b2 * (1.0 - 2 * y)) * x);
                    return (float)(mag * (term1 + term2));
                });
                Evaluator.EvalAndSave2DFunction((float x, float y) =>
                {
                    double term1 = Math.Exp(-x) * ((1.0 - y) * ((2 * mu + b1 * (1 - x)) * y + (sigma - mu) * x * y) + (2 * mu + b2 * (1.0 - 2 * y)) * x);
                    double term2 = Math.Exp(-1.0f + (x - 1.0) / mu) * ((1.0 - y) * ((2 - b1 * (1 + x / mu)) * y + (1.0 / mu - sigma) * x * y) - (2 * mu + b2 * (1.0 - 2 * y)) * x);
                    return (float)(-mag * (term1 + term2));
                });

                Evaluator.EvalAndSave2DFunction((float x, float y) =>
                {
                    double term1 = Math.Exp(-y) * (y * (b1 + 2 * mu - 2 * b1 * x + (x - 1) * x * (b2 + mu - sigma)) - (x - 1) * x * (b2 + 2 * mu));
                    double term2 = Math.Exp(-1.0 + (y - 1.0) / mu) * (-(x * x * ((b2 - 2) * mu + y * (b2 + mu * sigma - 1))) + x * ((b2 - 2) * mu + y * (-2 * b1 * mu + b2 + mu * sigma - 1)) + mu * sigma * (b1 + 2 * mu)) / mu;
                    return (float)(mag * (term1 - term2));
                });
                Evaluator.EvalAndSave2DFunction((float x, float y) =>
                {
                    double term1 = Math.Exp(-y) * (y * (b1 + 2 * mu - 2 * b1 * x + (x - 1) * x * (b2 + mu - sigma)) - (x - 1) * x * (b2 + 2 * mu));
                    double term2 = Math.Exp(-1.0 + (y - 1.0) / mu) * (-(x * x * ((b2 - 2) * mu + y * (b2 + mu * sigma - 1))) + x * ((b2 - 2) * mu + y * (-2 * b1 * mu + b2 + mu * sigma - 1)) + mu * sigma * (b1 + 2 * mu)) / mu;
                    return (float)(-mag * (term1 - term2));
                });

                Evaluator.EvalAndSave2DFunction((float x, float y) =>
                {
                    double magnitude = (mag / mu) * Math.Exp(-1.0 - x / mu);
                    double expTerm = Math.Exp(x * (1.0f + 1.0f / mu));

                    double part1 = -mu * (x - 1) * (b2 + 2 * mu) * (expTerm - 1.0);
                    double part2 = y * y * (-b1 * mu - b1 + mu * sigma - 2 * mu - mu * expTerm * (mu + sigma - x * (b1 - mu + sigma)) + b1 * x - mu * sigma * x + x - 1.0);
                    double part3 = y * (b1 * mu + b1 + 2 * b2 * mu - mu * sigma + 2 * mu + mu * expTerm * (-2 * b2 + mu + sigma - x * (b1 - 2 * b2 - mu + sigma)) - b1 * x - 2 * b2 * mu * x + mu * sigma * x + 1.0);
                    return (float)(magnitude * (part1 + part2 + part3));
                });
                Evaluator.EvalAndSave2DFunction((float x, float y) =>
                {
                    double magnitude = (mag / mu) * Math.Exp(-1.0 - x / mu);
                    double expTerm = Math.Exp(x * (1.0f + 1.0f / mu));

                    double part1 = -mu * (x - 1) * (b2 + 2 * mu) * (expTerm - 1.0);
                    double part2 = y * y * (-b1 * mu - b1 + mu * sigma - 2 * mu - mu * expTerm * (mu + sigma - x * (b1 - mu + sigma)) + b1 * x - mu * sigma * x + x - 1.0);
                    double part3 = y * (b1 * mu + b1 + 2 * b2 * mu - mu * sigma + 2 * mu + mu * expTerm * (-2 * b2 + mu + sigma - x * (b1 - 2 * b2 - mu + sigma)) - b1 * x - 2 * b2 * mu * x + mu * sigma * x + 1.0);
                    return (float)(-magnitude * (part1 + part2 + part3));
                });

                mu *= factor + RandomNoise.GetRandomNoise();
                sigma *= factor + RandomNoise.GetRandomNoise();
                b1 *= factor + RandomNoise.GetRandomNoise();
                b2 *= factor + RandomNoise.GetRandomNoise();
                mag /= factor + RandomNoise.GetRandomNoise();
            }
        }

        public static IMathFunction[] GetInstances(int n = 600, float min = 0.0f, float max = 1e2f)
        {
            double[] values = { 1e-2, 1e-1, 0.5, 1.0 };

            var instances = new IMathFunction[values.Length * values.Length * values.Length * values.Length];
            var idx = 0;
            foreach (var mu in values)
            {
                foreach (var sigma in values)
                {
                    foreach (var b1 in values)
                    {
                        foreach (var b2 in values)
                        {
                            instances[idx++] = new BoundaryLayerForcingTerm
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

