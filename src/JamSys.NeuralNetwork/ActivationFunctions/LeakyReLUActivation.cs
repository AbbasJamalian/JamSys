using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JamSys.NeuralNetwork.ActivationFunctions
{
    public class LeakyReLUActivation : IActivationFunction
    {
        public double Calculate(double input)
        {
            return input < 0 ? 0.01 * input : input;
        }

        public double CalculateDerivative(double input)
        {
            return input < 0 ? 0.01 : 1;
        }
    }
}
