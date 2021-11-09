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
    /// Implementation of Cross Entropy Loss Function
    /// </summary>
    public class CrossEntropyLoss : ILossFunction
    {
        public Tensor CalculateGradients(Tensor networkOutput, Tensor expectedOutput)
        {
            Tensor gradients = networkOutput.CreateSimilar();

            for (int x = 0; x < networkOutput.Width; x++)
            {
                gradients[x] = -expectedOutput[x] / networkOutput[x];
            }

            return gradients;
        }

        public double CalculateTotalLoss(Tensor networkOutput, Tensor expectedOutput)
        {
            double totalError = 0;

            for (int x = 0; x < networkOutput.Width; x++)
            {
                totalError += -expectedOutput[x] * Math.Log(networkOutput[x]);
            }

            return totalError;
        }
    }
}
