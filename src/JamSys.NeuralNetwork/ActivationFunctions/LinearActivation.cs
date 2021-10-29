using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JamSys.NeuralNetwork.ActivationFunctions
{
    public class LinearActivation : IActivationFunction
    {
        public double Calculate(double input)
        {
            return input;
        }

        public double CalculateDerivative(double fx)
        {
            return 1;
        }
    }
}
