using JamSys.NeuralNetwork.DataSet;
using System;

namespace JamSys.NeuralNetwork
{
    public interface ITrainer : IDisposable
    {
        ITrainer Initialize(int batchSize, double learningRate, LossFunctionEnum lossFunction);

        void Train(INetwork network, IDataSetProvider dataSet, int epoches);

        double Validate(INetwork network, IDataSetProvider dataSet);
    }
}
