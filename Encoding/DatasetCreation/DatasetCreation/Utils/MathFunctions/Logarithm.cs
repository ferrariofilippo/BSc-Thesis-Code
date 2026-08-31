namespace DatasetCreation.Utils.MathFunctions
{
    internal class Logarithm : IMathFunction
    {
        public float a { get; set; }
        public float b { get; set; }
        public float c { get; set; }
        public float d { get; set; }

        private float SafeLogInput(float value)
        {
            return Math.Abs(value) + 0.0001f;
        }

        public void Compute1D()
        {
            if (b != 0.0f || d != 0.0f)
                Evaluator.EvalAndSave1DFunction(x => (float)Math.Log(SafeLogInput(a + b * x)) * (float)Math.Log10(SafeLogInput(c + d * x)));

            if (b != 0.0f)
            {
                Evaluator.EvalAndSave1DFunction(x => b * (float)Math.Log(SafeLogInput(a * x)));
                Evaluator.EvalAndSave1DFunction(x => a + b * (float)Math.Log(SafeLogInput(c * x + d)));
                Evaluator.EvalAndSave1DFunction(x => b * (float)Math.Log10(SafeLogInput(a * x)));
                Evaluator.EvalAndSave1DFunction(x => a + b * (float)Math.Log10(SafeLogInput(c * x + d)));
            }

            if (a != 0.0f)
            {
                Evaluator.EvalAndSave1DFunction(x => c + a * (float)Math.Log(SafeLogInput(x)));
                Evaluator.EvalAndSave1DFunction(x => c + a * (float)Math.Log10(SafeLogInput(x)));
                Evaluator.EvalAndSave1DFunction(x => a * (float)Math.Log(SafeLogInput(b * x)) + c * (float)Math.Log10(SafeLogInput(d * x)));
            }
        }

        public void Compute2D()
        {
            for (int i = 0; i < 3; i++)
            {
                if (a != 0.0f || b != 0.0f || c != 0.0f || d != 0.0f)
                    Evaluator.EvalAndSave2DFunction((x, y) => (float)Math.Log(SafeLogInput(a * x + b * y)) + (float)Math.Log10(SafeLogInput(c * x + d * y)));

                if (a != 0.0f)
                    Evaluator.EvalAndSave2DFunction((x, y) => (c + a * (float)Math.Log10(SafeLogInput(x))) * (c + a * (float)Math.Log10(SafeLogInput(y))));

                if (b != 0.0f)
                {
                    if (a != 0.0f)
                    {
                        Evaluator.EvalAndSave2DFunction((x, y) => b * (float)Math.Log(SafeLogInput(a * x)) + b * (float)Math.Log(SafeLogInput(a * y)));
                        Evaluator.EvalAndSave2DFunction((x, y) => b * (float)Math.Log(SafeLogInput(a * x * y)));
                        Evaluator.EvalAndSave2DFunction((x, y) => b * (float)Math.Log(SafeLogInput(a * (float)Math.Sqrt(x * x + y * y))));
                    }

                    if (c != 0.0f)
                    {
                        Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Log(SafeLogInput(c * (x + y) + d)));
                        Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Log10(SafeLogInput(c * (float)Math.Sqrt(x * x + y * y) + d)));
                        Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Log10(SafeLogInput(c * (float)Math.Sqrt(x * x * y * y) + d)));
                        Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Log10(SafeLogInput(c * (float)Math.Sqrt(x * x * y * y + x) + d)));
                        Evaluator.EvalAndSave2DFunction((x, y) => a + b * (float)Math.Log10(SafeLogInput(c * (float)Math.Sqrt(x * x * y * y + x * y) + d)));
                    }
                }

                a *= 1.3f;
                b *= 1.3f;
                c *= 1.3f;
                d *= 1.3f;
            }
        }

        public static IMathFunction[] GetInstances(int n = 400, float min = 0.0f, float max = 1e2f)
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
                            instances[idx++] = new Logarithm { a = a, b = b, c = c, d = d };
                            instances[idx++] = new Logarithm { a = a, b = b, c = -c, d = -d };
                            instances[idx++] = new Logarithm { a = a, b = b, c = c, d = -d };
                            instances[idx++] = new Logarithm { a = a, b = b, c = -c, d = d };
                            instances[idx++] = new Logarithm { a = a, b = -b, c = c, d = d };
                            instances[idx++] = new Logarithm { a = a, b = -b, c = -c, d = -d };
                            instances[idx++] = new Logarithm { a = a, b = -b, c = c, d = -d };
                            instances[idx++] = new Logarithm { a = a, b = -b, c = -c, d = d };
                            instances[idx++] = new Logarithm { a = -a, b = b, c = c, d = d };
                            instances[idx++] = new Logarithm { a = -a, b = b, c = -c, d = -d };
                            instances[idx++] = new Logarithm { a = -a, b = b, c = c, d = -d };
                            instances[idx++] = new Logarithm { a = -a, b = b, c = -c, d = d };
                            instances[idx++] = new Logarithm { a = -a, b = -b, c = c, d = d };
                            instances[idx++] = new Logarithm { a = -a, b = -b, c = -c, d = -d };
                            instances[idx++] = new Logarithm { a = -a, b = -b, c = c, d = -d };
                            instances[idx++] = new Logarithm { a = -a, b = -b, c = -c, d = d };

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
