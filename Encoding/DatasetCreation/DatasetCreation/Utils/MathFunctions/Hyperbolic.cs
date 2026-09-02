namespace DatasetCreation.Utils.MathFunctions
{
    internal class Hyperbolic : IMathFunction
    {
        public float a { get; set; }
        public float b { get; set; }
        public float c { get; set; }

        public float SafeCoefficient(float value, float max = 3.0f)
        {
            return (float)(Math.Tanh(value) * max + RandomNoise.GetRandomNoise(-0.25f, 0.25f));
        }

        public void Compute1D()
        {
            float bClippedAtTwo = SafeCoefficient(b, 2.0f);
            if (b != 0.0f)
            {
                float aClippedAtOne = SafeCoefficient(a, 1.0f);
                float aClippedAtThree = SafeCoefficient(a, 3.0f);
                float cClippedAtThreeHalves = SafeCoefficient(c, 1.5f);
                float cClippedAtThree = SafeCoefficient(c, 3.0f);
                Evaluator.EvalAndSave1DFunction(x => (float)Math.Sinh(aClippedAtOne + bClippedAtTwo * x) * (float)Math.Cosh(cClippedAtThreeHalves * x));
                Evaluator.EvalAndSave1DFunction(x => b * (float)Math.Sinh(aClippedAtThree * x));
                Evaluator.EvalAndSave1DFunction(x => a + b * (float)Math.Sinh(cClippedAtThree * x));
                Evaluator.EvalAndSave1DFunction(x => b * (float)Math.Cosh(aClippedAtThree * x));
                Evaluator.EvalAndSave1DFunction(x => a + b * (float)Math.Cosh(cClippedAtThree * x));
                Evaluator.EvalAndSave1DFunction(x => b * (float)Math.Tanh(a * x));
                Evaluator.EvalAndSave1DFunction(x => a + b * (float)Math.Tanh(c * x));
            }

            if (a != 0.0f)
            {
                Evaluator.EvalAndSave1DFunction(x => c + a * (float)Math.Sinh(x));
                Evaluator.EvalAndSave1DFunction(x => c + a * (float)Math.Cosh(x));
                Evaluator.EvalAndSave1DFunction(x => a * (float)Math.Sinh(bClippedAtTwo * x) + c * (float)Math.Cosh(x));

                Evaluator.EvalAndSave1DFunction(x => c + a * (float)Math.Tanh(x));
            }
        }

        public void Compute2D()
        {
            float factor = 1.0f + (float)(Math.PI / 17.0);
            for (int i = 0; i < 3; i++)
            {
                float cClippedAtOne = SafeCoefficient(c, 1.0f);
                float aClippedAtThreeHalves = SafeCoefficient(a, 1.5f);
                float bClippedAtThreeHalves = SafeCoefficient(b, 1.5f);
                float cClippedAtThreeHalves = SafeCoefficient(c, 1.5f);
                float aClippedAtThree = SafeCoefficient(a, 3.0f);
                float cClippedAtThree = SafeCoefficient(c, 3.0f);

                if (a != 0.0f || b != 0.0f || c != 0.0f)
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Sinh(aClippedAtThreeHalves * x + bClippedAtThreeHalves * y) + (float)Math.Cosh(cClippedAtThreeHalves * x + y));

                if (a != 0.0f)
                {
                    Evaluator.EvalAndSave2DFunction((x, y) => (c + a * (float)Math.Cosh(x)) * (c + a * (float)Math.Cosh(y)));
                    Evaluator.EvalAndSave2DFunction((x, y) => c + a * (float)Math.Tanh(x) * (float)Math.Tanh(y));
                }

                if (b != 0.0f)
                {
                    if (a != 0.0f)
                    {
                        Evaluator.EvalAndSave2DFunction((x, y) => b * (float)Math.Sinh(aClippedAtThreeHalves * x) + b * (float)Math.Sinh(aClippedAtThreeHalves * y));
                        Evaluator.EvalAndSave2DFunction((x, y) => b * (float)Math.Cosh(aClippedAtThreeHalves * x) + b * (float)Math.Cosh(aClippedAtThreeHalves * y));
                        Evaluator.EvalAndSave2DFunction((x, y) => b * (float)Math.Tanh(a * x) + b * (float)Math.Sinh(aClippedAtThree * y));
                        Evaluator.EvalAndSave2DFunction((x, y) => b * (float)Math.Sinh(aClippedAtThree * x * y));
                        Evaluator.EvalAndSave2DFunction((x, y) => b * (float)Math.Cosh(aClippedAtThreeHalves * (float)Math.Sqrt(x * x + y * y)));
                        Evaluator.EvalAndSave2DFunction((x, y) => b * (float)Math.Tanh(a * (x + y)));
                    }

                    if (c != 0.0f)
                    {
                        Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Sinh(cClippedAtThreeHalves * (x + y)));
                        Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Cosh(cClippedAtThreeHalves * (x + y)));
                        Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Sinh(cClippedAtThree * (x - y)));
                        Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Cosh(cClippedAtThree * (x - y)));
                        Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Sinh(cClippedAtThree * (-x + y)));
                        Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Cosh(cClippedAtThree * (-x + y)));
                        Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Sinh(cClippedAtThree * (1.0f - x * y)));
                        Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Cosh(cClippedAtThree * (1.0f - x * y)));
                        Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Sinh(cClippedAtThree * (x * y)));
                        Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Cosh(cClippedAtThree * (x * y)));

                        Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Tanh(c * x) + b * (float)Math.Tanh(c * y));
                        Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Tanh(c * (x + y)));
                        Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Tanh(c * (x - y)));
                        Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Tanh(c * (-x + y)));
                        Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Tanh(c * (1.0f - x * y)));
                        Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Tanh(c * (x * y)));

                        Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Sinh(cClippedAtOne * (2.0f + x - y)));
                        Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Cosh(cClippedAtOne * (2.0f + x - y)));
                        Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Sinh(cClippedAtOne * (2.0f - x + y)));
                        Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Cosh(cClippedAtOne * (2.0f - x + y)));
                        Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Tanh(c * (2.0f + x - y)));
                        Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Tanh(c * (2.0f - x + y)));
                    }
                }

                a *= factor + RandomNoise.GetRandomNoise();
                b *= factor + RandomNoise.GetRandomNoise();
                c *= factor + RandomNoise.GetRandomNoise();
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
                a += stepA;
                b = min;
            }

            return instances;
        }
    }
}

