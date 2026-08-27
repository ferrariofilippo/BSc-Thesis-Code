namespace DatasetCreation.Utils
{
    internal static class FileUtil
    {
        private const string DATASET_1D = "../../../dataset/1d_function_dataset.csv";
        private const string NORMALIZATIONS_1D = "../../../dataset/1d_normalizations.csv";
        private const string DATASET_2D = "../../../dataset/2d_function_dataset.csv";
        private const string NORMALIZATIONS_2D = "../../../dataset/2d_normalizations.csv";

        public static StreamWriter GetDatasetWriter(bool is1D, bool append)
            => new StreamWriter(is1D ? DATASET_1D : DATASET_2D, append: append);

        public static StreamReader GetDatasetReader(bool is1D) 
            => new StreamReader(is1D ? DATASET_1D : DATASET_2D);

        public static StreamWriter GetNormalizations1DWriter(bool is1D, bool append)
            => new StreamWriter(is1D ? NORMALIZATIONS_1D : NORMALIZATIONS_2D, append: append);
    }
}
