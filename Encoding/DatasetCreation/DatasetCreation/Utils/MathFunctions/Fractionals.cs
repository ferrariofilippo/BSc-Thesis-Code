namespace DatasetCreation.Utils.MathFunctions
{
    internal class Fractionals : IMathFunction
    {
        public float a { get; set; }

        public void Compute1D()
        {
            if (a == 0.0f) return;
            Evaluator.EvalAndSave1DFunction(x => (1.0f + a) / (1.0f + x));
            Evaluator.EvalAndSave1DFunction(x => (1.0f + a) / (1.0f + x * x));
            Evaluator.EvalAndSave1DFunction(x => (1.0f + a) / (2.0f - x));
            Evaluator.EvalAndSave1DFunction(x => 1.0f / (1.0f + a * x));
        }

        public void Compute2D()
        {
            if (a == 0.0f) return;
            for (int i = 0; i < 3; i++)
            {
                Evaluator.EvalAndSave2DFunction((x, y) => (1.0f + a) / (1.0f + x + y));
                Evaluator.EvalAndSave2DFunction((x, y) => (1.0f + a) / (1.0f + x * x + y * y));
                Evaluator.EvalAndSave2DFunction((x, y) => (1.0f + a) / (3.0f - x - y));
                Evaluator.EvalAndSave2DFunction((x, y) => 1.0f / (1.0f + a * (x + y)));
                Evaluator.EvalAndSave2DFunction((x, y) => 1.0f / (1.0f + a + x - y));

                a *= 0.9f;
            }
        }

        public static IMathFunction[] GetInstances(int n = 500, float min = 0.0f, float max = 1.0f)
        {
            var instances = new IMathFunction[n];
            var step = (max - min) / (n - 1) / (float)Math.Pow(10.0, 3);
            var a = min;
            for (int i = 0; i < n / 2; i++)
            {
                instances[2 * i] = new Fractionals { a = a };
                instances[2 * i + 1] = new Fractionals { a = -a };

                a += step;
                if (i % 400 == 0)
                    step *= 10.0f;
            }

            return instances;
        }
    }
}
