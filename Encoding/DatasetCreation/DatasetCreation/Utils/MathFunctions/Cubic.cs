namespace DatasetCreation.Utils.MathFunctions
{
    internal class Cubic : IMathFunction
    {
        public float a { get; set; }
        public float b { get; set; }
        public float c { get; set; }
        public float d { get; set; }

        public void Compute1D()
        {
            if (a == 0.0f) return;
            Evaluator.EvalAndSave1DFunction(x => a * x * x * x + b * x * x + c * x + d);
        }

        public void Compute2D()
        {
            if (a == 0.0f) return;
            float factor = 1.0f + (float)(Math.PI / 7.0);
            for (int i = 0; i < 3; i++)
            {
                Evaluator.EvalAndSave2DFunction((x, y) => a * x * x * x + b * x * x + c * x + d);
                Evaluator.EvalAndSave2DFunction((x, y) => a * y * x * x + b * y * x + c * y + d);
                Evaluator.EvalAndSave2DFunction((x, y) => a * y * y * y + b * x * x + c * y + d);
                Evaluator.EvalAndSave2DFunction((x, y) => a * x * x * x + b * x * y + c * y + d);
                Evaluator.EvalAndSave2DFunction((x, y) => a * y * y * y + b * y * y + c * y + d);
                Evaluator.EvalAndSave2DFunction((x, y) => a * y * y * x + c * x + d);
                Evaluator.EvalAndSave2DFunction((x, y) => a * x * x * x + b * y * y + c * x * x + d);
                Evaluator.EvalAndSave2DFunction((x, y) => a * x * x * x + b * y * y * y + c * x + d * y);
                Evaluator.EvalAndSave2DFunction((x, y) => a * x * x * y + b * x * y + c * x * x + d);
                Evaluator.EvalAndSave2DFunction((x, y) => a * x * y * y + b * y * y + c * x * y + d);
                Evaluator.EvalAndSave2DFunction((x, y) => a * y * x * x + b * x * y + c + x + d);
                Evaluator.EvalAndSave2DFunction((x, y) => a * y * y * x + b * x * x * y + c * x + d * y);

                a *= factor + RandomNoise.GetRandomNoise();
                b *= factor + RandomNoise.GetRandomNoise();
                c *= factor + RandomNoise.GetRandomNoise();
                d *= factor + RandomNoise.GetRandomNoise();
            }
        }

        public static IMathFunction[] GetInstances(int n = 1296, float min = 0.0f, float max = 5e2f)
        {
            int howMany = (int)(Math.Sqrt(Math.Sqrt(n)));
            var instances = new IMathFunction[howMany * howMany * howMany * howMany];
            var step = (max - min) / (howMany - 1) / (float)Math.Pow(10.0, 4);
            var stepA = step;
            var stepB = step;
            var stepC = step;
            var stepD = step;
            var a = min;
            var b = min;
            var c = min;
            var d = min;
            int idx = 0;
            for (int i = 0; i < howMany / 2; i++)
            {
                stepB = step;
                for (int j = 0; j < howMany / 2; j++)
                {
                    stepC = step;
                    for (int k = 0; k < howMany / 2; k++)
                    {
                        stepD = step;
                        for (int l = 0; l < howMany / 2; l++)
                        {
                            instances[idx++] = new Cubic { a = a, b = b, c = c, d = d };
                            instances[idx++] = new Cubic { a = a, b = b, c = -c, d = -d };
                            instances[idx++] = new Cubic { a = a, b = b, c = c, d = -d };
                            instances[idx++] = new Cubic { a = a, b = b, c = -c, d = d };
                            instances[idx++] = new Cubic { a = a, b = -b, c = c, d = d };
                            instances[idx++] = new Cubic { a = a, b = -b, c = -c, d = -d };
                            instances[idx++] = new Cubic { a = a, b = -b, c = c, d = -d };
                            instances[idx++] = new Cubic { a = a, b = -b, c = -c, d = d };
                            instances[idx++] = new Cubic { a = -a, b = b, c = c, d = d };
                            instances[idx++] = new Cubic { a = -a, b = b, c = -c, d = -d };
                            instances[idx++] = new Cubic { a = -a, b = b, c = c, d = -d };
                            instances[idx++] = new Cubic { a = -a, b = b, c = -c, d = d };
                            instances[idx++] = new Cubic { a = -a, b = -b, c = c, d = d };
                            instances[idx++] = new Cubic { a = -a, b = -b, c = -c, d = -d };
                            instances[idx++] = new Cubic { a = -a, b = -b, c = c, d = -d };
                            instances[idx++] = new Cubic { a = -a, b = -b, c = -c, d = d };

                            stepD *= 10.0f;
                            d += stepD;
                        }

                        stepC *= 10.0f;
                        c += stepC;
                        d = min;
                    }

                    stepB *= 10.0f;
                    b += stepB;
                    c = min;
                }

                stepA *= 10.0f;
                a += stepA;
                b = min;
            }

            return instances;
        }
    }
}
