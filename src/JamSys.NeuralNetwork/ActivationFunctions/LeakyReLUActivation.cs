#region License
/*
 * Copyright (c) 2020 - Abbas Jamalian
 * This file is part of JamSys Project and is licensed under the MIT License. 
 * For more details see the License file provided with the software
 */
#endregion License

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
