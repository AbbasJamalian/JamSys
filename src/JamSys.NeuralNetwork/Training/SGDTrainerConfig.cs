using System;
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
    /// Configuration parameters used by SGDTrainer
    /// </summary>
    public class SGDTrainerConfig : ITrainerConfig
    {
        public int BatchSize { get; set; } = 1;
        public double LearningRate { get; set; } = 0.01;
        public LossFunctionEnum LossFunction { get; set; } = LossFunctionEnum.MSE;
    }
}
