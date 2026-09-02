namespace DatasetCreation.Utils
{
    internal static class RandomNoise
    {
        static Random random = new Random(53);

        public static float GetRandomNoise(float min = -0.05f, float max = 0.05f)
        {
            return (float)(random.NextDouble() * (max - min) + min);
        }
    }
}
