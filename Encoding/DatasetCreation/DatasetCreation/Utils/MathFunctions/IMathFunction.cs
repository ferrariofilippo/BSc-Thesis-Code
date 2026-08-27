namespace DatasetCreation.Utils.MathFunctions
{
    interface IMathFunction
    {
        void Compute1D();

        void Compute2D();

        abstract static IMathFunction[] GetInstances(int n = 500, float min = 0.0f, float max = 1e2f);
    }
}
