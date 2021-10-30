using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JamSys.NeuralNetwork.Training
{
    public class SGDTrainerConfig : ITrainerConfig
    {
        public int BatchSize { get; set; } = 1;
        public double LearningRate { get; set; } = 0.01;
        public LossFunctionEnum LossFunction { get; set; } = LossFunctionEnum.MSE;
    }
}
