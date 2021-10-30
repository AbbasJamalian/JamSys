using JamSys.NeuralNetwork.DataSet;
using System;
using System.Linq.Expressions;

namespace JamSys.NeuralNetwork
{
    public interface ITrainer : IDisposable
    {
        public ITrainer Configure(Action<ITrainerConfig> config);

        void Train(INetwork network, IDataSetProvider dataSet, int epoches);

        double Validate(INetwork network, IDataSetProvider dataSet);
    }
}
