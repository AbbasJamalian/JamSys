using System;

namespace JamSys.NeuralNetwork.LossFunctions
{
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
