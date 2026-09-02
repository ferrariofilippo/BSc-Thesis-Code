namespace DatasetCreation.Utils.MathFunctions
{
    internal class Sqrt : IMathFunction
    {
        public float Alpha { get; set; }

        public float Beta { get; set; }

        public void Compute1D()
        {
            Evaluator.EvalAndSave1DFunction(x => (float)Math.Sqrt(x + Alpha));
            if (Alpha != 0.0f)
            {
                Evaluator.EvalAndSave1DFunction(x => Alpha * (float)Math.Sqrt(x));
                Evaluator.EvalAndSave1DFunction(x => (float)Math.Sqrt(x * Alpha));
                Evaluator.EvalAndSave1DFunction(x => -Alpha * (float)Math.Sqrt(x));
            }
        }

        public void Compute2D()
        {
            float factor = 1.0f + (float)(Math.PI / 10.0);
            for (int i = 0; i < 3; i++)
            {
                if (Alpha != 0.0f)
                {
                    Evaluator.EvalAndSave2DFunction((x, y) => Alpha * (float)Math.Sqrt(x + y));
                    Evaluator.EvalAndSave2DFunction((x, y) => -Alpha * (float)Math.Sqrt(x + y));
                    Evaluator.EvalAndSave2DFunction((x, y) => Alpha * (float)Math.Sqrt(x * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => Alpha * x * (float)Math.Sqrt(x * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => Alpha * y * (float)Math.Sqrt(x * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => Alpha * x * y * (float)Math.Sqrt(x * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => Alpha * (float)Math.Sqrt(x));
                    Evaluator.EvalAndSave2DFunction((x, y) => Alpha * (float)Math.Sqrt(2.0 - x - y));
                    Evaluator.EvalAndSave2DFunction((x, y) => Beta * (float)Math.Sqrt(y));
                }

                if (Alpha != 0.0f || Beta != 0.0f)
                {
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Sqrt(Alpha * x + Beta * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Sqrt(Alpha * x * Beta * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Sqrt(Alpha + Beta * x * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Sqrt(Alpha + Beta / (1.0f + x + y)));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Sqrt(Alpha + Beta / (2.0f + x - y)));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Sqrt(Alpha + Beta / (3.0f - x - y)));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Sqrt(Alpha * (1.0f - x) + Beta / (1.0f + y)));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Sqrt(Alpha * (1.0f - y) + Beta / (1.0f + x)));
                }

                Alpha *= factor + RandomNoise.GetRandomNoise();
                Beta *= factor + RandomNoise.GetRandomNoise();
            }
        }

        public static IMathFunction[] GetInstances(int n = 400, float min = 0.0f, float max = 5e1f)
        {
            int howMany = (int)Math.Sqrt(n) / 2 * 2;
            var instances = new IMathFunction[howMany * howMany / 4];
            var step = (max - min) / (howMany - 1) / (float)Math.Pow(10.0, 3);
            var stepA = step;
            var stepB = step;
            var a = min;
            var b = min;
            int idx = 0;
            for (int i = 0; i < howMany / 2; i++)
            {
                stepB = step;
                for (int j = 0; j < howMany / 2; j++)
                {
                    instances[idx++] = new Sqrt { Alpha = a, Beta = b };
                    if (j % 4 == 0)
                        stepB *= 10.0f;
                
                    b += stepB;
                }

                if (i % 4 == 0)
                    stepA *= 10.0f;
                
                a += stepA;
                b = min;
            }

            return instances;
        }
    }
}
