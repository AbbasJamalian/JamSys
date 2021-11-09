#region License
/*
 * Copyright (c) 2020 - Abbas Jamalian
 * This file is part of JamSys Project and is licensed under the MIT License. 
 * For more details see the License file provided with the software
 */
#endregion License

namespace JamSys.NeuralNetwork.LossFunctions
{
    /// <summary>
    /// Interface for Loss Function which is used by training and measuring the accuracy 
    /// </summary>
    public interface ILossFunction
    {
        /// <summary>
        /// Calculates the total loss or error based on the expected value and current output of a network
        /// </summary>
        /// <param name="networkOutput"></param>
        /// <param name="expectedOutput"></param>
        /// <returns></returns>
        double CalculateTotalLoss(Tensor networkOutput, Tensor expectedOutput);

        /// <summary>
        /// Calculates the gradients which will be used for backpropagation for the last layer based on the expected output and current output of the network
        /// </summary>
        /// <param name="networkOutput"></param>
        /// <param name="expectedOutput"></param>
        /// <returns></returns>
        Tensor CalculateGradients(Tensor networkOutput, Tensor expectedOutput);
    }
}
