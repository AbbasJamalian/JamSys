#region License
/*
 * Copyright (c) 2020 - Abbas Jamalian
 * This file is part of JamSys Project and is licensed under the MIT License. 
 * For more details see the License file provided with the software
 */
#endregion License

namespace JamSys.NeuralNetwork.ActivationFunctions
{
    /// <summary>
    /// Interface for Activation Function which provides calculation of activation function and its derivative
    /// </summary>
    public interface IActivationFunction
    {
        double Calculate(double input);
        double CalculateDerivative(double x);
    }
}
