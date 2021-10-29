using JamSys.NeuralNetwork.Layers;
using System;
using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace JamSys.NeuralNetwork
{
    public interface INetwork : IDisposable
    {
        [JsonIgnore]
        public Tensor Input { get; }

        [JsonIgnore]
        public Tensor Output { get; }

        public List<ILayer> Layers { get; set; }

        string Save();

        INetwork Load(string jsonValue);

        INetwork AddInputLayer(int width, int height = 1, int depth = 1);

        INetwork AddDenseLayer(int numNeurons, ActivationFunctionEnum activation);

        INetwork AddOutputLayer(int numNeurons, ActivationFunctionEnum activation);

        INetwork AddSoftmaxLayer(int outputCount);

        INetwork Build();

        Tensor Run(Tensor input);

        void Backpropagate(Tensor delta, double rate);
    }
}
