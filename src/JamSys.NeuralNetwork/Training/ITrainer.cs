#region License
/*
 * Copyright (c) 2020 - Abbas Jamalian
 * This file is part of JamSys Project and is licensed under the MIT License. 
 * For more details see the License file provided with the software
 */
#endregion License

using JamSys.NeuralNetwork.DataSet;
using System;

namespace JamSys.NeuralNetwork.Training
{
    /// <summary>
    /// Interface for the trainers 
    /// </summary>
    public interface ITrainer
    {
        /// <summary>
        /// Configures the trainer. The parameters are provided by trainer itself
        /// </summary>
        /// <param name="config"></param>
        /// <returns></returns>
        public ITrainer Configure(Action<ITrainerConfig> config);

        /// <summary>
        /// Trains the network for the given dataset and number of epoches
        /// </summary>
        /// <param name="network"></param>
        /// <param name="dataSet"></param>
        /// <param name="epoches"></param>
        void Train(INetwork network, IDataSetProvider dataSet, int epoches);

        /// <summary>
        /// Validates the training of a network. It runs the dataset on the network and measures the total loss
        /// </summary>
        /// <param name="network"></param>
        /// <param name="dataSet"></param>
        /// <returns>Total loss of the network</returns>
        double Validate(INetwork network, IDataSetProvider dataSet);
    }
}
