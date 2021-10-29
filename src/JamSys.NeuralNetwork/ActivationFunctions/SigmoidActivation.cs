using System;
using System.Collections.Generic;
using System.Text;

namespace JamSys.NeuralNetwork.ActivationFunctions
{
    class SigmoidActivation : IActivationFunction
    {
        public double Calculate(double input)
        {
            return 1.0 / (1.0 + Math.Exp(-input));
        }

        public double CalculateDerivative(double x)
        {
            double fx = Calculate(x);
            return fx * (1.0 - fx);
        }
    }
}
