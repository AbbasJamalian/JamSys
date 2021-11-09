#region License
/*
 * Copyright (c) 2020 - Abbas Jamalian
 * This file is part of JamSys Project and is licensed under the MIT License. 
 * For more details see the License file provided with the software
 */
#endregion License

using JamSys.NeuralNetwork.DataSet;
using JamSys.NeuralNetwork.LossFunctions;
using System;

namespace JamSys.NeuralNetwork.Training
{
    /// <summary>
    /// Gradient Decent trainer with parameters to set the batch size, loss function and learning rate
    /// </summary>
    public class SGDTrainer : ITrainer
    {
        private int _batchSize;
        private double _learningRate;
        private ILossFunction _lossFunction;

        public SGDTrainer()
        {
        }

        public void Train(INetwork network, IDataSetProvider dataSet, int epoches)
        {
            //TODO: Check Input and Output sizes with the dataset data and label sizes
            for (int epoche = 0; epoche < epoches; epoche++)
            {
                dataSet.Shuffle();
                for (int index = 0; index < dataSet.Size; index += _batchSize)
                {
                    var delta = network.Output.CreateSimilar();
                    for (int x = 0; x < delta.Width; x++)
                        delta[x] = 0;


                    for (int offset = 0; offset < _batchSize && offset + index < dataSet.Size; offset++)
                    {
                        var output = network.Run(dataSet[index + offset].data);

                            delta.Add(_lossFunction.CalculateGradients(output, dataSet[index + offset].label));
                    }

                    for (int x = 0; x < delta.Width; x++)
                        delta[x] = (delta[x] / _batchSize);

                    network.Backpropagate(delta, _learningRate);
                }
            }
        }

        public double Validate(INetwork network, IDataSetProvider dataSet)
        {
            double totalError = 0;

            for (int index = 0; index < dataSet.Size; index++)
            {
                var output = network.Run(dataSet[index].data);

                totalError += _lossFunction.CalculateTotalLoss(output, dataSet[index].label);
            }

            totalError /= dataSet.Size;

            return totalError;
        }

        public ITrainer Configure(Action<ITrainerConfig> config)
        {
            ITrainerConfig trainerConfig = new SGDTrainerConfig();
            config.Invoke(trainerConfig);

            _batchSize = trainerConfig.BatchSize;
            _learningRate = trainerConfig.LearningRate;
            _lossFunction = Factory.Instance.ResolveNamed<ILossFunction>(trainerConfig.LossFunction);

            return this;
        }
    }
}
