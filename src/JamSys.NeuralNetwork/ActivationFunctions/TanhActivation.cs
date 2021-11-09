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
