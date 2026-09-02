namespace DatasetCreation.Utils.MathFunctions
{
    internal class Trigonometric : IMathFunction
    {
        public float a { get; set; }
        public float b { get; set; }
        public float c { get; set; }

        private static Func<float, float> safeTan = (val) => (float)Math.Tan(Math.PI * 0.49 * Math.Tanh(val));

        public void Compute1D()
        {
            if (a != 0.0f)
            {
                Evaluator.EvalAndSave1DFunction(x => c + a * (float)Math.Sin(x));
                Evaluator.EvalAndSave1DFunction(x => c + a * (float)Math.Cos(x));
                Evaluator.EvalAndSave1DFunction(x => c + a * safeTan(x));
                Evaluator.EvalAndSave1DFunction(x => (float)Math.Sin(a * 1e2 * x));
                Evaluator.EvalAndSave1DFunction(x => (float)Math.Cos(a * 1e2 * x));
                Evaluator.EvalAndSave1DFunction(x => (float)Math.Sin(2 * Math.PI * a * x));
                Evaluator.EvalAndSave1DFunction(x => (float)Math.Sin(Math.PI * a * x));
                Evaluator.EvalAndSave1DFunction(x => (float)Math.Cos(2 * Math.PI * a * x));
                Evaluator.EvalAndSave1DFunction(x => (float)Math.Cos(Math.PI * a * x));
                Evaluator.EvalAndSave1DFunction(x => a * (float)Math.Sin(2 * Math.PI * x));
                Evaluator.EvalAndSave1DFunction(x => a * (float)Math.Cos(2 * Math.PI * a * x));
            } 
            else
            {
                Evaluator.EvalAndSave1DFunction(x => (float)Math.Sin(Math.PI * x));
                Evaluator.EvalAndSave1DFunction(x => (float)Math.Cos(Math.PI * x));
                Evaluator.EvalAndSave1DFunction(x => (float)Math.Sin(2 * Math.PI * x));
                Evaluator.EvalAndSave1DFunction(x => (float)Math.Cos(2 * Math.PI * x));
                Evaluator.EvalAndSave1DFunction(x => (float)Math.Sin(4 * Math.PI * x));
                Evaluator.EvalAndSave1DFunction(x => (float)Math.Cos(4 * Math.PI * x));
                Evaluator.EvalAndSave1DFunction(x => (float)Math.Sin(16 * Math.PI * x));
                Evaluator.EvalAndSave1DFunction(x => (float)Math.Cos(16 * Math.PI * x));
                Evaluator.EvalAndSave1DFunction(x => (float)Math.Sin(128 * Math.PI * x));
                Evaluator.EvalAndSave1DFunction(x => (float)Math.Cos(128 * Math.PI * x));
            }

            if (a != 0.0f || c != 0.0f)
            {
                Evaluator.EvalAndSave1DFunction(x => a * (float)Math.Sin(b * x) + c * (float)Math.Cos(x));
                Evaluator.EvalAndSave1DFunction(x => a * (float)Math.Sin(x) + c * (float)Math.Cos(b * x));
                Evaluator.EvalAndSave1DFunction(x => a * (float)Math.Sin(b + x) + c * (float)Math.Cos(x));
            }

            if (b != 0.0f)
            {
                Evaluator.EvalAndSave1DFunction(x => a + b * (float)Math.Sin(c * x));
                Evaluator.EvalAndSave1DFunction(x => b * (float)Math.Sin(a * x));
                Evaluator.EvalAndSave1DFunction(x => b * (float)Math.Cos(a * x));
                Evaluator.EvalAndSave1DFunction(x => b * (float)Math.Sin(a * 2 * Math.PI * x));
                Evaluator.EvalAndSave1DFunction(x => b * (float)Math.Cos(a * 2 * Math.PI * x));
                Evaluator.EvalAndSave1DFunction(x => b * (float)Math.Cos(Math.PI * x));
                Evaluator.EvalAndSave1DFunction(x => b * (float)Math.Sin(Math.PI * x));
                Evaluator.EvalAndSave1DFunction(x => a + b * (float)Math.Cos(c * x));
                Evaluator.EvalAndSave1DFunction(x => a + b * safeTan(c * x));
                Evaluator.EvalAndSave1DFunction(x => b * safeTan(a * x));
                Evaluator.EvalAndSave1DFunction(x => (float)Math.Sin(a + b * x) + (float)Math.Cos(c + x));
                Evaluator.EvalAndSave1DFunction(x => (float)Math.Sin(a + b * x) + (float)Math.Cos(c + x));
                Evaluator.EvalAndSave1DFunction(x => (float)Math.Sin(a + b * x) * (float)Math.Cos(c + x));
                Evaluator.EvalAndSave1DFunction(x => (float)Math.Sin(a + b * 2 * Math.PI * x) + (float)Math.Cos(c + 2 * Math.PI * x));
                Evaluator.EvalAndSave1DFunction(x => b * (float)Math.Sin(Math.PI * x));
                Evaluator.EvalAndSave1DFunction(x => b * (float)Math.Cos(Math.PI * x));
                Evaluator.EvalAndSave1DFunction(x => b * (float)Math.Sin(2 * Math.PI * x));
                Evaluator.EvalAndSave1DFunction(x => b * (float)Math.Cos(2 * Math.PI * x));
                Evaluator.EvalAndSave1DFunction(x => b * (float)Math.Sin(4 * Math.PI * x));
                Evaluator.EvalAndSave1DFunction(x => b * (float)Math.Cos(4 * Math.PI * x));
            }
        }

        public void Compute2D()
        {
            float factor = 1.0f + (float)(Math.PI / 10.0);

            for (int i = 0; i < 3; i++)
            {
                if (a != 0.0f && b != 0.0f)
                {
                    Evaluator.EvalAndSave2DFunction((x, y) => b * safeTan(a * (x + y)));
                    Evaluator.EvalAndSave2DFunction((x, y) => b * (float)Math.Sin(a * x * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => b * (float)Math.Sin(a * x) + b * (float)Math.Sin(a * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => b * (float)Math.Sin(a * 2 * Math.PI * x) + b * (float)Math.Sin(a * 2 * Math.PI * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => b * (float)Math.Cos(a * 2 * Math.PI * x) + b * (float)Math.Cos(a * 2 * Math.PI * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => b * (float)Math.Cos(a * 2 * Math.PI * x) + b * (float)Math.Sin(a * 2 * Math.PI * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => b * (float)Math.Sin(a * 2 * Math.PI * Math.Sqrt(x * x + y * y)));
                    Evaluator.EvalAndSave2DFunction((x, y) => b * (float)Math.Sin(a * Math.Sqrt(x * x + y * y)));
                }

                if (a != 0.0f)
                {
                    Evaluator.EvalAndSave2DFunction((x, y) => (c + a * (float)Math.Sin(x)) * (c + a * (float)Math.Cos(y)));
                    Evaluator.EvalAndSave2DFunction((x, y) => c + a * safeTan(x) * safeTan(y));
                }
                else
                {
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Sin(Math.PI * x));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Cos(Math.PI * x));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Sin(2 * Math.PI * x));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Cos(2 * Math.PI * x));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Sin(4 * Math.PI * x));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Cos(4 * Math.PI * x));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Sin(2 * Math.PI * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Cos(2 * Math.PI * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Sin(4 * Math.PI * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Cos(4 * Math.PI * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Sin(2 * Math.PI * x) * (float)Math.Sin(2 * Math.PI * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Cos(2 * Math.PI * y) * (float)Math.Cos(2 * Math.PI * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Sin(2 * Math.PI * x) * (float)Math.Cos(2 * Math.PI * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Cos(2 * Math.PI * x) * (float)Math.Sin(2 * Math.PI * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Sin(16 * Math.PI * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Cos(16 * Math.PI * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Sin(4 * Math.PI * x) * (float)Math.Sin(2 * Math.PI * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Cos(4 * Math.PI * x) * (float)Math.Cos(16 * Math.PI * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Sin(16 * Math.PI * x) * (float)Math.Cos(2 * Math.PI * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Cos(16 * Math.PI * x) * (float)Math.Sin(4 * Math.PI * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Sin(8 * Math.PI * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Cos(8 * Math.PI * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Sin(8 * Math.PI * x) * (float)Math.Sin(16 * Math.PI * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Cos(4 * Math.PI * x) * (float)Math.Cos(8 * Math.PI * y));
                }

                if (b != 0.0f && c != 0.0f)
                    Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Cos(c * Math.Sqrt(x * x + y * y)));

                if (a != 0.0f && b != 0.0f && c != 0.0f)
                {
                    Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Sin(c * (x + y)));
                    Evaluator.EvalAndSave2DFunction((x, y) => a * (float)Math.Sin(b * x) + c * (float)Math.Cos(y));
                    Evaluator.EvalAndSave2DFunction((x, y) => a * (float)Math.Sin(x) * c * (float)Math.Cos(b * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Sin(a + b * x) * (float)Math.Cos(c + y));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Sin(a * x + b * y) + (float)Math.Cos(c * x + y));
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Sin(a * x - b * y) * (float)Math.Cos(c * y - x));
                    Evaluator.EvalAndSave2DFunction((x, y) => a + b * safeTan(c * x) + b * safeTan(c * y));
                    Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Sin(c * Math.PI * (x + y) / (3.0f - x - y)));
                    Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Sin(c * Math.PI * (x - y)));
                    Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Sin(-c * Math.PI * (x + y)));
                    Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Sin(c * Math.PI * (-x + y)));
                    Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Cos(c * Math.PI * (x + y) / (3.0f - x - y)));
                    Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Cos(c * Math.PI * (x - y)));
                    Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Cos(-c * Math.PI * (x + y)));
                    Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Cos(c * Math.PI * (-x + y)));
                    Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Sin(c * Math.PI / (1.0f + x + y)));
                    Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Cos(c * Math.PI / (1.0f + x + y)));
                    Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Sin(c * Math.PI / (2.0f + x - y)));
                    Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Cos(c * Math.PI / (2.0f + x - y)));
                    Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Cos(c * Math.Exp(x - y)));
                    Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Cos(-c * Math.Exp(x + y)));
                }

                a *= factor + RandomNoise.GetRandomNoise();
                b *= factor + RandomNoise.GetRandomNoise();
                c *= factor + RandomNoise.GetRandomNoise();
            }
        }

        public static IMathFunction[] GetInstances(int n = 512, float min = 0.0f, float max = 1e1f)
        {
            int howMany = (int)(Math.Cbrt(n));
            var instances = new IMathFunction[howMany * howMany * howMany];
            var step = (max - min) / (howMany - 1) / (float)Math.Pow(10.0, 3);
            var stepA = step;
            var stepB = step;
            var stepC = step;
            var a = min;
            var b = min;
            var c = min;
            int idx = 0;
            for (int i = 0; i < howMany / 2; i++)
            {
                stepB = step;
                for (int j = 0; j < howMany / 2; j++)
                {
                    stepC = step;
                    for (int k = 0; k < howMany / 2; k++)
                    {
                        instances[idx++] = new Trigonometric { a = a, b = b, c = c };
                        instances[idx++] = new Trigonometric { a = a, b = b, c = -c };
                        instances[idx++] = new Trigonometric { a = a, b = -b, c = c };
                        instances[idx++] = new Trigonometric { a = a, b = -b, c = -c };
                        instances[idx++] = new Trigonometric { a = -a, b = b, c = c };
                        instances[idx++] = new Trigonometric { a = -a, b = b, c = -c };
                        instances[idx++] = new Trigonometric { a = -a, b = -b, c = c };
                        instances[idx++] = new Trigonometric { a = -a, b = -b, c = -c };

                        stepC *= 10.0f;
                        c += stepC;
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
