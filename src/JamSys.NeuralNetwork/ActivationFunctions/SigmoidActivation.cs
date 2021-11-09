#region License
/*
 * Copyright (c) 2020 - Abbas Jamalian
 * This file is part of JamSys Project and is licensed under the MIT License. 
 * For more details see the License file provided with the software
 */
#endregion License

using System;

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
