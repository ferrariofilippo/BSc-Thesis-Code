namespace DatasetCreation.Utils.MathFunctions
{
    internal class Linear : IMathFunction
    {
        public float m { get; set; }
        public float q { get; set; }

        public void Compute1D()
        {
            if (m == 0.0) return;
            Evaluator.EvalAndSave1DFunction(x => m * x + q);
        }

        public void Compute2D()
        {
            if (m == 0.0) return;
            float factor = 1.0f + (float)(Math.PI / 10.0);
            for (int i = 0; i < 3; i++)
            {
                Evaluator.EvalAndSave2DFunction((x, y) => m * x + q);
                Evaluator.EvalAndSave2DFunction((x, y) => m * x + q * y);
                Evaluator.EvalAndSave2DFunction((x, y) => m * y + q);
                Evaluator.EvalAndSave2DFunction((x, y) => m * y + m * x + q);
                m *= factor + RandomNoise.GetRandomNoise();
                q *= factor + RandomNoise.GetRandomNoise();
            }
        }

        public static IMathFunction[] GetInstances(int n = 824, float min = 0.0f, float max = 3e0f)
        {
            int howMany = (int)(Math.Sqrt(n)) / 4 * 4;
            var instances = new IMathFunction[howMany * howMany];
            var step = (max - min) / (howMany / 4 - 1) / (float)Math.Pow(10.0, 3);
            var stepA = step;
            var stepB = step;
            var m = min;
            var q = min;
            int idx = 0;
            for (int i = 0; i < howMany / 2; i++)
            {
                stepB = step;
                for (int j = 0; j < howMany / 2; j++)
                {
                    instances[idx++] = new Linear { m = m, q = q };
                    instances[idx++] = new Linear { m = -m, q = q };
                    instances[idx++] = new Linear { m = m, q = -q };
                    instances[idx++] = new Linear { m = -m, q = -q };

                    if (j % 4 == 0)
                        stepB *= 10.0f;

                    q += stepB;
                }

                if (i % 4 == 0)
                    stepA *= 10.0f;

                m += stepA;
                q = min;
            }

            return instances;
        }
    }
}
