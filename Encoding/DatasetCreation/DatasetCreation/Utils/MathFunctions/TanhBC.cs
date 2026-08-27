namespace DatasetCreation.Utils.MathFunctions
{
    internal class TanhBC : IMathFunction
    {
        public float mu { get; set; }

        public void Compute1D()
        {
            Evaluator.EvalAndSave1DFunction((float x) => 0.5f * ((float)Math.Tanh((2 * x - 1) / mu) + 1.0f));
            Evaluator.EvalAndSave1DFunction((float x) => (float)Math.Tanh(2 * (1 - x) / mu));
        }

        public void Compute2D()
        {
            return;
        }

        public static IMathFunction[] GetInstances(int n = 600, float min = 0.0f, float max = 1e2f)
        {
            float[] values = { 1e-5f, 1e-4f, 1e-3f, 1e-2f, 1e-1f, 1.0f };

            var instances = new IMathFunction[10 * values.Length];
            var idx = 0;
            for ( int i = 0; i < 10; i++)
            {
                foreach (var mu in values)
                    instances[idx++] = new TanhBC() { mu = mu };
            }

            return instances;
        }
    }
}
