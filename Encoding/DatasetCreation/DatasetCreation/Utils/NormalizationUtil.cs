namespace DatasetCreation.Utils
{
    internal static class NormalizationUtil
    {
        public static void Compute1DNormalizationFactors()
        {
            const int coefficientCount = 16;
            const int csvFieldCount = 1 + coefficientCount;
            var (means, stds, nonZero, validRowCount) =
                ComputeNormalizationFactors(true, csvFieldCount);

            using (var writer = FileUtil.GetNormalizations1DWriter(true, append: false))
                WriteFactors(writer, means, stds);

            ReportNonZeroCoefficients(
                "1D",
                nonZero,
                validRowCount,
                coefficientIndex => $"c[{coefficientIndex}]"
            );
        }

        public static void Compute2DNormalizationFactors()
        {
            const int coefficientAxisLength = 16;
            const int coefficientCount = coefficientAxisLength * coefficientAxisLength;
            const int csvFieldCount = 1 + coefficientCount;
            var (means, stds, nonZero, validRowCount) =
                ComputeNormalizationFactors(false, csvFieldCount);

            using (var writer = FileUtil.GetNormalizations1DWriter(false, append: false))
                WriteFactors(writer, means, stds);

            ReportNonZeroCoefficients(
                "2D",
                nonZero,
                validRowCount,
                coefficientIndex =>
                    $"c[{coefficientIndex / coefficientAxisLength},{coefficientIndex % coefficientAxisLength}]"
            );
        }

        private static (double[] Means, double[] Stds, bool[] NonZero, int ValidRowCount)
            ComputeNormalizationFactors(bool is1D, int csvFieldCount)
        {
            int coefficientCount = csvFieldCount - 1;
            var means = new double[coefficientCount];
            var squaredDifferences = new double[coefficientCount];
            var nonZero = new bool[coefficientCount];

            int validRowCount = 0;
            using (var reader = FileUtil.GetDatasetReader(is1D))
            {
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    if (line is null)
                        continue;

                    var values = line.Split(',');
                    if (values.Length != csvFieldCount)
                        continue;

                    int nextRowCount = validRowCount + 1;
                    for (int fieldIndex = 1; fieldIndex < csvFieldCount; fieldIndex++)
                    {
                        int coefficientIndex = fieldIndex - 1;
                        double value = double.Parse(
                            values[fieldIndex],
                            System.Globalization.CultureInfo.InvariantCulture
                        );

                        nonZero[coefficientIndex] |= value != 0.0;

                        // Welford's online update avoids the cancellation in
                        // E[x^2] - E[x]^2 while computing the same population std.
                        double delta = value - means[coefficientIndex];
                        means[coefficientIndex] += delta / nextRowCount;
                        double updatedDelta = value - means[coefficientIndex];
                        squaredDifferences[coefficientIndex] += delta * updatedDelta;
                    }

                    validRowCount = nextRowCount;
                }
            }

            var stds = new double[coefficientCount];
            if (validRowCount > 0)
            {
                for (int i = 0; i < coefficientCount; i++)
                    stds[i] = Math.Sqrt(Math.Max(0.0, squaredDifferences[i] / validRowCount));
            }

            return (means, stds, nonZero, validRowCount);
        }

        private static void WriteFactors(StreamWriter writer, double[] means, double[] stds)
        {
            for (int i = 0; i < means.Length; i++)
            {
                writer.WriteLine(
                    string.Format(
                        System.Globalization.CultureInfo.InvariantCulture,
                        "{0},{1:F7},{2:F7}",
                        i + 1,
                        means[i],
                        stds[i]
                    )
                );
            }
        }

        private static void ReportNonZeroCoefficients(
            string dimension,
            bool[] nonZero,
            int validRowCount,
            Func<int, string> formatCoefficient
        )
        {
            var labels = new List<string>();
            for (int i = 0; i < nonZero.Length; i++)
            {
                if (nonZero[i])
                    labels.Add(formatCoefficient(i));
            }

            Console.WriteLine(
                $"{dimension} sanity check: {labels.Count}/{nonZero.Length} coefficients " +
                $"were non-zero at least once across {validRowCount} valid rows."
            );

            if (labels.Count == 0)
            {
                Console.WriteLine("  None");
                return;
            }

            const int labelsPerLine = 12;
            for (int start = 0; start < labels.Count; start += labelsPerLine)
                Console.WriteLine("  " + string.Join(", ", labels.Skip(start).Take(labelsPerLine)));
        }
    }
}
