namespace DatasetCreation.Utils.MathFunctions
{
    internal class Cbrt : IMathFunction
    {
        public float Alpha { get; set; }

        public float Beta { get; set; }

        public void Compute1D()
        {
            Evaluator.EvalAndSave1DFunction(x => (float)Math.Cbrt(x + Alpha));
            if (Alpha != 0.0f)
            {
                Evaluator.EvalAndSave1DFunction(x => (float)Math.Cbrt(x * Alpha));
                Evaluator.EvalAndSave1DFunction(x => Alpha * (float)Math.Cbrt(x));
                Evaluator.EvalAndSave1DFunction(x => -Alpha * (float)Math.Cbrt(x));
            }
        }

        public void Compute2D()
        {
            Alpha *= 3e2f;
            Beta *= 3e2f;
            for (int i = 0; i < 3; i++)
            {
                if (Alpha != 0.0f || Beta != 0.0f)
                {
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Cbrt(Alpha * x + Beta * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Cbrt(Alpha * x + Beta * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Cbrt(Alpha * x * y));
                }

                if (Alpha != 0.0f)
                {
                    Evaluator.EvalAndSave2DFunction((x, y) => Alpha * (float)Math.Cbrt(x + y));
                    Evaluator.EvalAndSave2DFunction((x, y) => Alpha * (float)Math.Cbrt(x - y));
                    Evaluator.EvalAndSave2DFunction((x, y) => -Alpha * (float)Math.Cbrt(x + y));
                    Evaluator.EvalAndSave2DFunction((x, y) => -Alpha * (float)Math.Cbrt(x - y));
                    Evaluator.EvalAndSave2DFunction((x, y) => Alpha * (float)Math.Cbrt(x * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => Alpha * (float)Math.Cbrt(x));
                    Evaluator.EvalAndSave2DFunction((x, y) => Alpha * (float)Math.Cbrt(1.0 - x));
                    Evaluator.EvalAndSave2DFunction((x, y) => Alpha * (float)Math.Cbrt((1.0 + x) / (1.0 + y)));
                }

                if (Beta != 0.0f)
                    Evaluator.EvalAndSave2DFunction((x, y) => Beta * (float)Math.Cbrt(y));

                Alpha *= 1.5f;
                Beta *= 1.5f;
            }
        }

        public static IMathFunction[] GetInstances(int n = 600, float min = 0.0f, float max = 5e1f)
        {
            int howMany = 32;
            var instances = new IMathFunction[howMany * howMany / 16 * 4];
            var step = (max - min) / (howMany / 8 - 1) / (float)Math.Pow(10.0, 4);
            var stepA = step;
            var stepB = step;
            var a = min;
            var b = min;
            int idx = 0;
            for (int i = 0; i < howMany / 4; i++)
            {
                stepB = step;
                for (int j = 0; j < howMany / 4; j++)
                {
                    instances[idx++] = new Cbrt { Alpha = a, Beta = b };
                    instances[idx++] = new Cbrt { Alpha = -a, Beta = b };
                    instances[idx++] = new Cbrt { Alpha = -a, Beta = -b };
                    instances[idx++] = new Cbrt { Alpha = a, Beta = -b };

                    if (j % 2 == 0)
                        stepB *= 10.0f;

                    b += step;
                }

                if (i % 2 == 0)
                    stepA *= 10.0f;

                a += step;
                b = min;
            }

            return instances;
        }
    }
}
