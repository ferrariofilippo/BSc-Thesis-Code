namespace DatasetCreation.Utils.MathFunctions
{
    internal class Hyperbolic : IMathFunction
    {
        public float a { get; set; }
        public float b { get; set; }
        public float c { get; set; }

        private float SafeHyperbolicInput(float value)
        {
            return (float)Math.Max(Math.Min(value, 10.0f), -10.0f) / 1.5f;
        }

        public void Compute1D()
        {
            if (b != 0.0f)
            {
                Evaluator.EvalAndSave1DFunction(x => (float)Math.Sinh(SafeHyperbolicInput(a + b * x)) * (float)Math.Cosh(SafeHyperbolicInput(c * x)));
                Evaluator.EvalAndSave1DFunction(x => b * (float)Math.Sinh(SafeHyperbolicInput(a * x)));
                Evaluator.EvalAndSave1DFunction(x => a + b * (float)Math.Sinh(SafeHyperbolicInput(c * x)));
                Evaluator.EvalAndSave1DFunction(x => b * (float)Math.Cosh(SafeHyperbolicInput(a * x)));
                Evaluator.EvalAndSave1DFunction(x => a + b * (float)Math.Cosh(SafeHyperbolicInput(c * x)));
                Evaluator.EvalAndSave1DFunction(x => b * (float)Math.Tanh(a * x));
                Evaluator.EvalAndSave1DFunction(x => a + b * (float)Math.Tanh(c * x));
            }

            if (a != 0.0f)
            {
                Evaluator.EvalAndSave1DFunction(x => c + a * (float)Math.Sinh(SafeHyperbolicInput(x)));
                Evaluator.EvalAndSave1DFunction(x => c + a * (float)Math.Cosh(SafeHyperbolicInput(x)));
                Evaluator.EvalAndSave1DFunction(x => a * (float)Math.Sinh(SafeHyperbolicInput(b * x)) + c * (float)Math.Cosh(SafeHyperbolicInput(x)));
                Evaluator.EvalAndSave1DFunction(x => c + a * (float)Math.Tanh(x));
            }
        }

        public void Compute2D()
        {
            for (int i = 0; i < 3; i++)
            {
                if (a != 0.0f || b != 0.0f || c != 0.0f)
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Sinh(SafeHyperbolicInput(a * x + b * y)) + (float)Math.Cosh(SafeHyperbolicInput(c * x + y)));

                if (a != 0.0f)
                {
                    Evaluator.EvalAndSave2DFunction((x, y) => (c + a * (float)Math.Cosh(SafeHyperbolicInput(x))) * (c + a * (float)Math.Cosh(SafeHyperbolicInput(y))));
                    Evaluator.EvalAndSave2DFunction((x, y) => c + a * (float)Math.Tanh(x) * (float)Math.Tanh(y));
                }

                if (b != 0.0f)
                {
                    if (a != 0.0f)
                    {
                        Evaluator.EvalAndSave2DFunction((x, y) => b * (float)Math.Sinh(SafeHyperbolicInput(a * x)) + b * (float)Math.Sinh(SafeHyperbolicInput(a * y)));
                        Evaluator.EvalAndSave2DFunction((x, y) => b * (float)Math.Sinh(SafeHyperbolicInput(a * x * y)));
                        Evaluator.EvalAndSave2DFunction((x, y) => b * (float)Math.Cosh(SafeHyperbolicInput(a * (float)Math.Sqrt(x * x + y * y))));
                        Evaluator.EvalAndSave2DFunction((x, y) => b * (float)Math.Tanh(a * (x + y)));
                    }

                    if (c != 0.0f)
                    {
                        Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Sinh(SafeHyperbolicInput(c * (x + y))));
                        Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Tanh(c * x) + b * (float)Math.Tanh(c * y));
                    }
                }

                a *= 1.2f;
                b *= 1.2f;
                c *= 1.2f;
            }
        }

        public static IMathFunction[] GetInstances(int n = 512, float min = 0.0f, float max = 1e1f)
        {
            int howMany = (int)Math.Cbrt(n);
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
                        instances[idx++] = new Hyperbolic { a = a, b = b, c = c };
                        instances[idx++] = new Hyperbolic { a = a, b = b, c = -c };
                        instances[idx++] = new Hyperbolic { a = a, b = -b, c = c };
                        instances[idx++] = new Hyperbolic { a = a, b = -b, c = -c };
                        instances[idx++] = new Hyperbolic { a = -a, b = b, c = c };
                        instances[idx++] = new Hyperbolic { a = -a, b = b, c = -c };
                        instances[idx++] = new Hyperbolic { a = -a, b = -b, c = c };
                        instances[idx++] = new Hyperbolic { a = -a, b = -b, c = -c };

                        stepC *= 10.0f;
                        c += stepC;
                    }

                    stepB *= 10.0f;
                    b += stepB;
                    c = min;
                }

                stepA *= 10.0f;
                a += step;
                b = min;
            }

            return instances;
        }
    }
}

