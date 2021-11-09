#region License
/*
 * Copyright (c) 2020 - Abbas Jamalian
 * This file is part of JamSys Project and is licensed under the MIT License. 
 * For more details see the License file provided with the software
 */
#endregion License

using System.Text.Json.Serialization;

namespace JamSys.NeuralNetwork.Layers
{
    /// <summary>
    /// Interface for a network layer
    /// </summary>
    public interface ILayer
    {
        /// <summary>
        /// Input of the Layer. The Input cannot be set after the layer is built. The layer user need to modify the contents of the Input
        /// </summary>
        [JsonIgnore]
        public Tensor Input { get; }

        /// <summary>
        /// Output Tensor of the Layer. The values of the output Tensor will be modified after Run method is executed
        /// </summary>
        [JsonIgnore]
        public Tensor Output { get; }

        /// <summary>
        /// Initializes the Layer and binds it to its previous layer
        /// </summary>
        /// <param name="previousLayer">null if the layer is input layer, otherwise the previos layer in the network</param>
        public void Build(ILayer previousLayer);

        /// <summary>
        /// clears the contents of the Tensors to free up memory
        /// </summary>
        public void Clear();

        /// <summary>
        /// Runs the forward path for the layer. uses the values of the Input Tensor to generate values for the output Tensor
        /// </summary>
        void Run();

        /// <summary>
        /// Called by trainer from last layer back to the first layer
        /// </summary>
        /// <param name="delta">Gradients to be used for back propagation</param>
        /// <param name="rate">Learning rate</param>
        /// <returns>Gradients for the previous layers</returns>
        Tensor Backpropagate(Tensor delta, double rate);

    }
}
