using System;

namespace JamSys.NeuralNetwork.ActivationFunctions
{
    public class TanhActivation : IActivationFunction
    {
        public double Calculate(double input)
        {
            return 2 / (1 + Math.Pow(Math.E, -(2 * input))) - 1;
        }

        public double CalculateDerivative(double x)
        {
            return 1 - Math.Pow(Calculate(x), 2);
        }
    }
}
