using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JamSys.NeuralNetwork
{
    public interface ITrainerConfig
    {
        public int BatchSize { get; set; }
        
        public double LearningRate { get; set; }

        public LossFunctionEnum LossFunction { get; set; }
    }
}
