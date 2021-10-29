using System;
using System.Collections.Generic;
using System.Text;

namespace JamSys.NeuralNetwork.ActivationFunctions
{
    public class ReLUActivation : IActivationFunction
    {
        public double Calculate(double input)
        {
            return input < 0 ? 0 : input;
        }

        public double CalculateDerivative(double input)
        {
            return input < 0 ? 0 : 1;
        }
    }
}
