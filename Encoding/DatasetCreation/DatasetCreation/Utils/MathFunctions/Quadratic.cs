namespace DatasetCreation.Utils.MathFunctions
{
    internal class Quadratic : IMathFunction
    {
        public float a { get; set; }
        public float b { get; set; }
        public float c { get; set; }

        public void Compute1D()
        {
            if (a == 0.0f) return;
            Evaluator.EvalAndSave1DFunction(x => a * x * x + b * x + c);
        }

        public void Compute2D()
        {
            float factor = 1.0f + (float)(Math.PI / 17.0);

            a *= 0.6f;
            b *= 0.6f;
            c *= 0.6f;
            if (a == 0.0f) return;
            for (int i = 0; i < 3; i++)
            { 
                Evaluator.EvalAndSave2DFunction((x, y) => a * x * x + b * x + c);
                Evaluator.EvalAndSave2DFunction((x, y) => a * y * y + b * y + c);
                Evaluator.EvalAndSave2DFunction((x, y) => a * x * x + c);
                Evaluator.EvalAndSave2DFunction((x, y) => a * y * y + b);
                Evaluator.EvalAndSave2DFunction((x, y) => a * x * y + b);
                Evaluator.EvalAndSave2DFunction((x, y) => a * y * y + b * x + c);
                Evaluator.EvalAndSave2DFunction((x, y) => a * x * y + b * (x + y) + c);
                Evaluator.EvalAndSave2DFunction((x, y) => a * x * y + b * y + c * x);

                a *= factor + RandomNoise.GetRandomNoise();
                b *= factor + RandomNoise.GetRandomNoise();
                c *= factor + RandomNoise.GetRandomNoise();
            }
        }

        public static IMathFunction[] GetInstances(int n = 1000, float min = 0.0f, float max = 5e1f)
        {
            int howMany = (int)(Math.Cbrt(n)) / 2 * 2;
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
                        instances[idx++] = new Quadratic { a = a, b = b, c = c };
                        instances[idx++] = new Quadratic { a = a, b = -b, c = c };
                        instances[idx++] = new Quadratic { a = a, b = -b, c = -c };
                        instances[idx++] = new Quadratic { a = a, b = b, c = -c };
                        instances[idx++] = new Quadratic { a = -a, b = b, c = c };
                        instances[idx++] = new Quadratic { a = -a, b = -b, c = c };
                        instances[idx++] = new Quadratic { a = -a, b = b, c = -c };
                        instances[idx++] = new Quadratic { a = -a, b = -b, c = -c };
                        
                        if (k < 3)
                            stepC *= 10.0f;
                        
                        c += stepC;
                    }

                    if (j < 3)
                        stepB *= 10.0f;
                    
                    b += stepB;
                    c = min;
                }

                if (i < 3)
                    stepA *= 10.0f;
                
                a += stepA;
                b = min;
            }

            return instances;
        }
    }
}
