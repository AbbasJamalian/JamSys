using System;
using System.Collections.Generic;
using System.Text;

namespace JamSys.NeuralNetwork.ActivationFunctions
{
    public interface IActivationFunction
    {
        double Calculate(double input);
        double CalculateDerivative(double x);
    }
}
