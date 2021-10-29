using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JamSys.NeuralNetwork.LossFunctions
{
    public interface ILossFunction
    {
        double CalculateTotalLoss(Tensor networkOutput, Tensor expectedOutput);

        Tensor CalculateGradients(Tensor networkOutput, Tensor expectedOutput);
    }
}
