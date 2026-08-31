namespace DatasetCreation.Utils.MathFunctions
{
    internal class Panettone : IMathFunction
    {
        public float mu { get; set; }

        public float sigma { get; set; }

        public float b1 { get; set; }

        public float b2 { get; set; }

        public void Compute1D() { return; }
        public void Compute2D()
        {
            if (mu == 0.0f) return;
            for (int i = 0; i < 3; i++)
            {
                Evaluator.EvalAndSave2DFunction((float x, float y) =>
                {
                    float term1 = (float)Math.Exp(-x) * ((1.0f - y) * ((2 * mu + b1 * (1 - x)) * y + (sigma - mu) * x * y) + (2 * mu + b2 * (1.0f - 2 * y)) * x);
                    float term2 = (float)Math.Exp(-1.0f + (x - 1.0f) / mu) * ((1.0f - y) * ((2 - b1 * (1 + x / mu)) * y + (1.0f / mu - sigma) * x * y) - (2 * mu + b2 * (1.0f - 2 * y)) * x);
                    return 10.0f * (term1 + term2);
                });
                Evaluator.EvalAndSave2DFunction((float x, float y) =>
                {
                    float term1 = (float)Math.Exp(-x) * ((1.0f - y) * ((2 * mu + b1 * (1 - x)) * y + (sigma - mu) * x * y) + (2 * mu + b2 * (1.0f - 2 * y)) * x);
                    float term2 = (float)Math.Exp(-1.0f + (x - 1.0f) / mu) * ((1.0f - y) * ((2 - b1 * (1 + x / mu)) * y + (1.0f / mu - sigma) * x * y) - (2 * mu + b2 * (1.0f - 2 * y)) * x);
                    return -10.0f * (term1 + term2);
                });

                Evaluator.EvalAndSave2DFunction((float x, float y) =>
                {
                    float term1 = (float)Math.Exp(-y) * (y * (b1 + 2 * mu - 2 * b1 * x + (x - 1) * x * (b2 + mu - sigma)) - (x - 1) * x * (b2 + 2 * mu));
                    float term2 = (float)Math.Exp(-1.0f + (y - 1.0f) / mu) * (-(x * x * ((b2 - 2) * mu + y * (b2 + mu * sigma - 1))) + x * ((b2 - 2) * mu + y * (-2 * b1 * mu + b2 + mu * sigma - 1)) + mu * sigma * (b1 + 2 * mu)) / mu;
                    return 10.0f * (term1 - term2);
                });
                Evaluator.EvalAndSave2DFunction((float x, float y) =>
                {
                    float term1 = (float)Math.Exp(-y) * (y * (b1 + 2 * mu - 2 * b1 * x + (x - 1) * x * (b2 + mu - sigma)) - (x - 1) * x * (b2 + 2 * mu));
                    float term2 = (float)Math.Exp(-1.0f + (y - 1.0f) / mu) * (-(x * x * ((b2 - 2) * mu + y * (b2 + mu * sigma - 1))) + x * ((b2 - 2) * mu + y * (-2 * b1 * mu + b2 + mu * sigma - 1)) + mu * sigma * (b1 + 2 * mu)) / mu;
                    return -10.0f * (term1 - term2);
                });

                if (mu <= 1e-3f) return;

                Evaluator.EvalAndSave2DFunction((float x, float y) =>
                {
                    float factor = (10.0f / mu) * (float)Math.Exp(-1.0f - x / mu);
                    float expTerm = (float)Math.Exp(x * (1.0f + 1.0f / mu));

                    float part1 = -mu * (x - 1) * (b2 + 2 * mu) * (expTerm - 1.0f);
                    float part2 = y * y * (-b1 * mu - b1 + mu * sigma - 2 * mu - mu * expTerm * (mu + sigma - x * (b1 - mu + sigma)) + b1 * x - mu * sigma * x + x - 1.0f);
                    float part3 = y * (b1 * mu + b1 + 2 * b2 * mu - mu * sigma + 2 * mu + mu * expTerm * (-2 * b2 + mu + sigma - x * (b1 - 2 * b2 - mu + sigma)) - b1 * x - 2 * b2 * mu * x + mu * sigma * x + 1.0f);
                    return factor * (part1 + part2 + part3);
                });
                Evaluator.EvalAndSave2DFunction((float x, float y) =>
                {
                    float factor = (10.0f / mu) * (float)Math.Exp(-1.0f - x / mu);
                    float expTerm = (float)Math.Exp(x * (1.0f + 1.0f / mu));

                    float part1 = -mu * (x - 1) * (b2 + 2 * mu) * (expTerm - 1.0f);
                    float part2 = y * y * (-b1 * mu - b1 + mu * sigma - 2 * mu - mu * expTerm * (mu + sigma - x * (b1 - mu + sigma)) + b1 * x - mu * sigma * x + x - 1.0f);
                    float part3 = y * (b1 * mu + b1 + 2 * b2 * mu - mu * sigma + 2 * mu + mu * expTerm * (-2 * b2 + mu + sigma - x * (b1 - 2 * b2 - mu + sigma)) - b1 * x - 2 * b2 * mu * x + mu * sigma * x + 1.0f);
                    return -factor * (part1 + part2 + part3);
                });

                mu *= 1.2f;
                sigma *= 1.2f;
                b1 *= 1.2f;
                b2 *= 1.2f;
            }
        }

        public static IMathFunction[] GetInstances(int n = 600, float min = 0.0f, float max = 1e2f)
        {
            float[] values = { 2e-4f, 1e-3f, 1e-1f, 0.5f, 1.0f };

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
                            instances[idx++] = new Panettone
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

