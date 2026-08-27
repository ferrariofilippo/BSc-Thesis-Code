using DatasetCreation.Utils;
using DatasetCreation.Utils.MathFunctions;

void CreateDataset()
{
    Func<IMathFunction[]>[] functionFamilies =
    [
        () => Constant.GetInstances(),
        () => Linear.GetInstances(),
        () => Quadratic.GetInstances(),
        () => Cubic.GetInstances(),
        () => Sqrt.GetInstances(),
        () => Cbrt.GetInstances(),
        () => Arrow.GetInstances(),
        () => Panettone.GetInstances(),
        () => Exponential.GetInstances(),
        () => TanhBC.GetInstances(),
        () => Logarithm.GetInstances(),
        () => Trigonometric.GetInstances(),
        () => Hyperbolic.GetInstances()
    ];

    Console.WriteLine("---Starting---");
    for (int familyIndex = 0; familyIndex < functionFamilies.Length; familyIndex++)
    {
        Console.WriteLine($"- {familyIndex}/{functionFamilies.Length}");
        foreach (var function in functionFamilies[familyIndex]())
        {
            function.Compute1D();
            function.Compute2D();
        }
    }

    Console.WriteLine($"- {functionFamilies.Length}/{functionFamilies.Length}");
    Console.WriteLine("---Finished---");
}

void GetNormalizationFactors()
{
    Console.WriteLine("---Initializing Normalization Factors---");
    NormalizationUtil.Compute1DNormalizationFactors();
    Console.WriteLine("---Finished 1D Normalization Factors---");
    Console.WriteLine("---Initializing 2D Normalization Factors---");
    NormalizationUtil.Compute2DNormalizationFactors();
    Console.WriteLine("---Finished 2D Normalization Factors---");
}

CreateDataset();
GetNormalizationFactors();
