using System.Text;

namespace DatasetCreation.Utils
{
    internal static class Evaluator
    {
        private const string CoefficientFormat = ",{0:F7}";

        private static int oneDimID = 1;
        private static int twoDimID = 1;

        private static readonly Chebyshev _chebyshev = new();

        public static void EvalAndSave1DFunction(
            Func<float, float> f,
            float start = 0.0f,
            float end = 1.0f,
            int intervals = 128
        )
        {
            float[] values = new float[intervals];
            for (int i = 0; i < intervals; i++)
            {
                float x = (float)Chebyshev.GetGaussLobattoNode(i, intervals, start, end);
                values[i] = f(x);
            }

            var coefficients = _chebyshev.Compute1DCoefficients(values);
            var builder = new StringBuilder(coefficients.Length * 11);
            builder.Append(oneDimID);
            for (int i = 0; i < coefficients.Length; i++)
            {
                builder.Append(
                    string.Format(
                        System.Globalization.CultureInfo.InvariantCulture,
                        CoefficientFormat,
                        coefficients[i]
                    )
                );
            }

            using (var writer = FileUtil.GetDatasetWriter(true, true))
                writer.WriteLine(builder.ToString());

            ++oneDimID;
        }

        public static void EvalAndSave2DFunction(
            Func<float, float, float> f,
            float start = 0.0f,
            float end = 1.0f,
            int intervals = 64
        )
        {
            float[,] values = new float[intervals, intervals];
            for (int i = 0; i < intervals; i++)
            {
                float x = (float)Chebyshev.GetGaussLobattoNode(i, intervals, start, end);
                for (int j = 0; j < intervals; j++)
                {
                    float y = (float)Chebyshev.GetGaussLobattoNode(j, intervals, start, end);
                    values[i, j] = f(x, y);
                }
            }

            var coefficients = _chebyshev.Compute2DCoefficients(values);
            var builder = new StringBuilder(coefficients.Length * 11);
            builder.Append(twoDimID);
            for (int i = 0; i < coefficients.GetLength(0); i++)
            {
                for (int j = 0; j < coefficients.GetLength(1); j++)
                {
                    builder.Append(
                        string.Format(
                            System.Globalization.CultureInfo.InvariantCulture,
                            CoefficientFormat,
                            coefficients[i, j]
                        )
                    );
                }
            }

            using (var writer = FileUtil.GetDatasetWriter(false, true))
                writer.WriteLine(builder.ToString());

            ++twoDimID;
        }
    }
}
