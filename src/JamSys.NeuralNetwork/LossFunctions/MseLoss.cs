using System;

namespace JamSys.NeuralNetwork.LossFunctions
{
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
            //TODO: Check Tensors are single dimentional
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
