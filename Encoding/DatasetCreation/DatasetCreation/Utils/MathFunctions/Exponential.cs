namespace DatasetCreation.Utils.MathFunctions
{
    internal class Exponential : IMathFunction
    {
        public float Alpha { get; set; }
        public float Beta { get; set; }

        public void Compute1D()
        {
            Evaluator.EvalAndSave1DFunction(x => (float)Math.Exp(Beta + x));
            Evaluator.EvalAndSave1DFunction(x => x * (float)Math.Exp(Beta + x));
            Evaluator.EvalAndSave1DFunction(x => (float)Math.Log(0.05f + x) * (float)Math.Exp(Alpha * x));
            if (Alpha != 0.0f)
            {
                Evaluator.EvalAndSave1DFunction(x => (float)Math.Exp(Alpha * x * x));
                Evaluator.EvalAndSave1DFunction(x => Alpha * (float)Math.Exp(Beta * x));
                Evaluator.EvalAndSave1DFunction(x => Alpha * (float)Math.Exp(x));
                Evaluator.EvalAndSave1DFunction(x => (float)Math.Exp(Math.Sin(2 * Math.PI * Alpha * x)));
                Evaluator.EvalAndSave1DFunction(x => (float)Math.Exp(Math.Cos(2 * Math.PI * Alpha * x)));
                Evaluator.EvalAndSave1DFunction(x => (float)Math.Exp(Alpha / (1.0f + x)));
                Evaluator.EvalAndSave1DFunction(x => (float)Math.Exp(Alpha / (1.0f + x * x)));
            }
        }

        public void Compute2D()
        {
            for (int i = 0; i < 3; i++)
            {
                Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Exp(Beta + x));
                Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Exp(Beta + y));
                Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Exp(Alpha * x * x + y * y));
                if (Alpha != 0.0f || Beta != 0.0f)
                {
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Exp(Alpha * x + Beta * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Exp(Alpha * x) + (float)Math.Exp(Beta * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Exp(Alpha * x) - (float)Math.Exp(Beta * y));
                }

                if (Alpha != 0.0f)
                {
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Exp(Alpha * x * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Log(0.05f + y) * (float)Math.Exp(Alpha * x));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Exp(Alpha / (1.0f + x + y)));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Exp(Alpha / (1.0f + x * x + y * y)));

                    if (Beta != 0.0f)
                    {
                        Evaluator.EvalAndSave2DFunction((x, y) => Alpha * (float)Math.Exp(Beta * x));
                        Evaluator.EvalAndSave2DFunction((x, y) => Alpha * (float)Math.Exp(Beta * y));
                        Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Exp(Math.Sin(2 * Math.PI * Beta * x) + Math.Sin(2 * Math.PI * Alpha * y)));
                        Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Exp(Math.Cos(2 * Math.PI * Alpha * x) + Math.Cos(2 * Math.PI * Beta * y)));
                    }
                }

                Alpha *= 1.2f;
                Beta *= 1.2f;
            }
        }

        public static IMathFunction[] GetInstances(int n = 576, float min = 0.0f, float max = 5.0f)
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
                    instances[idx++] = new Exponential { Alpha = m, Beta = q };
                    instances[idx++] = new Exponential { Alpha = -m, Beta = q };
                    instances[idx++] = new Exponential { Alpha = m, Beta = -q };
                    instances[idx++] = new Exponential { Alpha = -m, Beta = -q };

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
