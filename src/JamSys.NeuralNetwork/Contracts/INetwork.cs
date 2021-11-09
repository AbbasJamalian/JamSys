#region License
/*
 * Copyright (c) 2020 - Abbas Jamalian
 * This file is part of JamSys Project and is licensed under the MIT License. 
 * For more details see the License file provided with the software
 */
#endregion License

using JamSys.NeuralNetwork.Layers;
using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace JamSys.NeuralNetwork
{

    /// <summary>
    /// Main Interface representing a neural network
    /// </summary>
    public interface INetwork
    {
        /// <summary>
        /// The output of the network which is valid after the Run is executed
        /// </summary>
        [JsonIgnore]
        public Tensor Output { get; }

        /// <summary>
        /// Saves the content of the network including all of its parameters to a JSON string
        /// </summary>
        /// <returns></returns>
        string Save();

        /// <summary>
        /// Load the network including all of its parameters from a JSON string
        /// </summary>
        /// <param name="jsonValue"></param>
        /// <returns></returns>
        INetwork Load(string jsonValue);

        /// <summary>
        /// Adds an Input Layer to the network. This must be the first layer in the network,
        /// </summary>
        /// <param name="width">Width of Input</param>
        /// <param name="height">Height of input. Must be 1 if the input is single dimensional</param>
        /// <param name="depth">Depth of input. Must be 1 if the input is single dimensional</param>
        /// <returns></returns>
        INetwork AddInputLayer(int width, int height = 1, int depth = 1);

        /// <summary>
        /// Adds a Dense (Fully connected) layer to network. Network should have one input layer to let dense layers to be added
        /// </summary>
        /// <param name="numNeurons">Number of neurons in the layer</param>
        /// <param name="activation">Activation function</param>
        /// <returns></returns>
        INetwork AddDenseLayer(int numNeurons, ActivationFunctionEnum activation);

        /// <summary>
        /// Adds a Dense Layer as output layer to the network
        /// </summary>
        /// <param name="numNeurons"></param>
        /// <param name="activation"></param>
        /// <returns></returns>
        INetwork AddOutputLayer(int numNeurons, ActivationFunctionEnum activation);

        /// <summary>
        /// Adds a Softmax layer as output layer to the network
        /// </summary>
        /// <param name="outputCount"></param>
        /// <returns></returns>
        INetwork AddSoftmaxLayer(int outputCount);

        /// <summary>
        /// Initializes the network for execution or training. It must be called after the layers are added to network. 
        /// </summary>
        /// <returns></returns>
        INetwork Build();

        /// <summary>
        /// Performs the forward run on the network and returns the result
        /// </summary>
        /// <param name="input"></param>
        /// <returns></returns>
        Tensor Run(Tensor input);

        /// <summary>
        /// Used for training and is called by Trainer object
        /// </summary>
        /// <param name="delta"></param>
        /// <param name="rate"></param>
        void Backpropagate(Tensor delta, double rate);
    }
}
