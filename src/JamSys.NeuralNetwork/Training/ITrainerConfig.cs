#region License
/*
 * Copyright (c) 2020 - Abbas Jamalian
 * This file is part of JamSys Project and is licensed under the MIT License. 
 * For more details see the License file provided with the software
 */
#endregion License

namespace JamSys.NeuralNetwork.Training
{
    /// <summary>
    /// Interface for configurations of the trainer
    /// </summary>
    public interface ITrainerConfig
    {
        /// <summary>
        /// Batch size - number of elements in a batch. minimum value is 1
        /// </summary>
        public int BatchSize { get; set; }
        
        /// <summary>
        /// Learning rate - usually a small value less than one
        /// </summary>
        public double LearningRate { get; set; }

        /// <summary>
        /// Loss function to be used for training
        /// </summary>
        public LossFunctionEnum LossFunction { get; set; }
    }
}
