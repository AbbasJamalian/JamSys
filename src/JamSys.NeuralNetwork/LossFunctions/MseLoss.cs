#region License
/*
 * Copyright (c) 2020 - Abbas Jamalian
 * This file is part of JamSys Project and is licensed under the MIT License. 
 * For more details see the License file provided with the software
 */
#endregion License

using System;

namespace JamSys.NeuralNetwork.LossFunctions
{
    /// <summary>
    /// Implementation of the Mean squared error loss function
    /// </summary>
    public class MseLoss : ILossFunction
    {
        public Tensor CalculateGradients(Tensor networkOutput, Tensor expectedOutput)
        {
            Tensor gradients = networkOutput.CreateSimilar();

            for (int i = 0; i < networkOutput.Width; i++)
            {
                gradients[i] = networkOutput[i] - expectedOutput[i];
            }

            return gradients;
        }

        public double CalculateTotalLoss(Tensor networkOutput, Tensor expectedOutput)
        {
            double error = 0;

            for (int i = 0; i < networkOutput.Width; i++)
            {
                error += Math.Pow(networkOutput[i] - expectedOutput[i], 2);
            }

            error /= 2;

            return error;
        }
    }
}
