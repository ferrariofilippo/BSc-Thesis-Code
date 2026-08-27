namespace DatasetCreation.Utils
{
    /// <summary>
    /// Computes truncated Chebyshev expansions from values sampled at
    /// Chebyshev-Gauss-Lobatto nodes.
    /// </summary>
    internal sealed class Chebyshev
    {
        public int InputLength1D { get; }
        public int InputLength2D { get; }
        public int OutputLength1D { get; }
        public int OutputLength2D { get; }

        private readonly double[,] _transform1D;
        private readonly double[,] _transform2D;

        public Chebyshev(
            int inputLength1D = 128,
            int maxDegree1D = 15,
            int inputLength2D = 64,
            int maxDegree2D = 15
        )
        {
            ValidateDimensions(inputLength1D, maxDegree1D, nameof(inputLength1D));
            ValidateDimensions(inputLength2D, maxDegree2D, nameof(inputLength2D));

            InputLength1D = inputLength1D;
            InputLength2D = inputLength2D;
            OutputLength1D = maxDegree1D + 1;
            OutputLength2D = maxDegree2D + 1;

            // Only the requested low-degree rows are needed. Precomputing them
            // also makes the tensor-product 2D transform substantially cheaper
            // than computing a full DCT and discarding most of it afterwards.
            _transform1D = BuildTruncatedDctI(inputLength1D, OutputLength1D);
            _transform2D = BuildTruncatedDctI(inputLength2D, OutputLength2D);
        }

        public double[] Compute1DCoefficients(float[] input)
        {
            ArgumentNullException.ThrowIfNull(input);
            if (input.Length != InputLength1D)
            {
                throw new ArgumentException(
                    $"1D input must contain exactly {InputLength1D} samples, but received {input.Length}.",
                    nameof(input)
                );
            }

            var coefficients = new double[OutputLength1D];
            for (int degree = 0; degree < OutputLength1D; degree++)
            {
                double sum = 0.0;
                for (int sample = 0; sample < InputLength1D; sample++)
                    sum += _transform1D[degree, sample] * input[sample];

                coefficients[degree] = sum;
            }

            return coefficients;
        }

        public double[,] Compute2DCoefficients(float[,] input)
        {
            ArgumentNullException.ThrowIfNull(input);
            if (input.GetLength(0) != InputLength2D || input.GetLength(1) != InputLength2D)
            {
                throw new ArgumentException(
                    $"2D input must be exactly {InputLength2D}x{InputLength2D}, but received " +
                    $"{input.GetLength(0)}x{input.GetLength(1)}.",
                    nameof(input)
                );
            }

            // A 2D Chebyshev transform is separable: transform the x axis,
            // then transform the y axis. The default result is [0, 10]^2.
            var alongX = new double[OutputLength2D, InputLength2D];
            for (int xDegree = 0; xDegree < OutputLength2D; xDegree++)
            {
                for (int ySample = 0; ySample < InputLength2D; ySample++)
                {
                    double sum = 0.0;
                    for (int xSample = 0; xSample < InputLength2D; xSample++)
                        sum += _transform2D[xDegree, xSample] * input[xSample, ySample];

                    alongX[xDegree, ySample] = sum;
                }
            }

            var coefficients = new double[OutputLength2D, OutputLength2D];
            for (int xDegree = 0; xDegree < OutputLength2D; xDegree++)
            {
                for (int yDegree = 0; yDegree < OutputLength2D; yDegree++)
                {
                    double sum = 0.0;
                    for (int ySample = 0; ySample < InputLength2D; ySample++)
                        sum += _transform2D[yDegree, ySample] * alongX[xDegree, ySample];

                    coefficients[xDegree, yDegree] = sum;
                }
            }

            return coefficients;
        }

        /// <summary>
        /// Maps cos(pi*i/(count-1)) from [-1, 1] onto [start, end].
        /// Nodes are ordered from end to start.
        /// </summary>
        public static double GetGaussLobattoNode(int index, int count, double start, double end)
        {
            if (count < 2)
                throw new ArgumentOutOfRangeException(nameof(count), "At least two nodes are required.");
            if ((uint)index >= (uint)count)
                throw new ArgumentOutOfRangeException(nameof(index));

            double standardNode = Math.Cos(Math.PI * index / (count - 1));
            return 0.5 * (start + end) + 0.5 * (end - start) * standardNode;
        }

        private static double[,] BuildTruncatedDctI(int inputLength, int outputLength)
        {
            int maxTransformDegree = inputLength - 1;
            var transform = new double[outputLength, inputLength];

            for (int degree = 0; degree < outputLength; degree++)
            {
                // Equivalent to scipy.fft.dct(values, type=1) / (n - 1),
                // followed by halving the first and last coefficients.
                double outputScale = degree == 0 || degree == maxTransformDegree ? 0.5 : 1.0;
                for (int sample = 0; sample < inputLength; sample++)
                {
                    double inputWeight = sample == 0 || sample == maxTransformDegree ? 1.0 : 2.0;
                    transform[degree, sample] =
                        outputScale * inputWeight *
                        Math.Cos(Math.PI * degree * sample / maxTransformDegree) /
                        maxTransformDegree;
                }
            }

            return transform;
        }

        private static void ValidateDimensions(int inputLength, int maxDegree, string parameterName)
        {
            if (inputLength < 2)
                throw new ArgumentOutOfRangeException(parameterName, "DCT-I requires at least two samples.");
            if (maxDegree < 0 || maxDegree >= inputLength)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(maxDegree),
                    $"The maximum degree must be between 0 and {inputLength - 1}."
                );
            }
        }
    }
}
