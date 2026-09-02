namespace DatasetCreation.Utils.MathFunctions
{
    internal class Constant : IMathFunction
    {
        public float Magnitude { get; set; } = 1.0f;

        public void Compute1D()
        {
            int max = Magnitude == 0.0f ? 30 : 3;
            for (int i = 0; i < max; i++)
            { 
                Evaluator.EvalAndSave1DFunction(x => Magnitude);
                Magnitude *= 1.1f;
            }
        }

        public void Compute2D()
        {
            float factor = 1.0f + (float)(Math.PI / 10.0);
            for (int i = 0; i < 6; i++)
            {
                Evaluator.EvalAndSave2DFunction((x, y) => Magnitude);

                Magnitude *= factor + RandomNoise.GetRandomNoise();
            }
        }

        public static IMathFunction[] GetInstances(int n = 2000, float min = 0.0f, float max = 1.5e1f)
        {
            n = n / 2 * 2;
            var offset = 0;
            var instances = new IMathFunction[n];
            var step = (max - min) / (n / 2 - 1) / (float)Math.Pow(10.0, 3);
            var m = min;
            for (int i = 0; i < n / 2; i++)
            {
                instances[offset + 2 * i] = new Constant { Magnitude = m };
                instances[offset + 2 * i + 1] = new Constant { Magnitude = -m };
                if (i % 450 == 0)
                    step *= 10.0f;

                m += step;
            }

            offset += n;

            return instances;
        }
    }
}
