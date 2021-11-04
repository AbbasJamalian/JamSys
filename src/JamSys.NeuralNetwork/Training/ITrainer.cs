using JamSys.NeuralNetwork.DataSet;
using System;
using System.Linq.Expressions;

namespace JamSys.NeuralNetwork.Training
{
    public interface ITrainer
    {
        public ITrainer Configure(Action<ITrainerConfig> config);

        void Train(INetwork network, IDataSetProvider dataSet, int epoches);

        double Validate(INetwork network, IDataSetProvider dataSet);
    }
}
