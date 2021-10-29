using JamSys.NeuralNetwork.Nodes;
using System.Collections.Generic;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace JamSys.NeuralNetwork.Layers
{
    public class DenseLayer : ILayer
    {
        public List<Neuron> Neurons { get; set; }

        [JsonIgnore]
        public Tensor Input { get; private set; }

        [JsonIgnore]
        public Tensor Output { get; private set; }

        public ActivationFunctionEnum Activation { get; private set; }
        public int NeuronCount { get; private set; }

        public DenseLayer(int neuronCount, ActivationFunctionEnum activation)
        {
            NeuronCount = neuronCount;
            Activation = activation;
        }

        public void Build(ILayer previousLayer)
        {
            //TODO: Check if output of previous layer is one dimentional - just in case automatic flattening
            Input = previousLayer.Output;
            Output = new Tensor(NeuronCount);

            if (Neurons == null)
            {
                Neurons = new List<Neuron>();
                for (int index = 0; index < NeuronCount; index++)
                {
                    var neuron = new Neuron(Activation, index);
                    Neurons.Add(neuron);
                }
            }

            Neurons.ForEach(n => n.Build(Input, Output));
        }

        public void Run()
        {
            Parallel.ForEach(Neurons, n => n.Run());
            //_neurons.ForEach(n => n.Run());
        }

        public void Dispose()
        {
            if (Output != null)
            {
                Output.Dispose();
                Output = null;
            }

            if (Neurons != null)
            {
                Neurons.ForEach(n => n.Dispose());
                Neurons.Clear();
                Neurons = null;
            }
        }

        public void Clear()
        {
            Output?.Clear();
        }

        public Tensor Backpropagate(Tensor delta, double rate)
        {
            //TODO: Check delta is one dimensional with the width equal to number of neurons
            Tensor layerDelta = Input.CreateSimilar();

            for (int index = 0; index < NeuronCount; index++)
            {
                layerDelta.Add(Neurons[index].Backpropagate(delta[index], rate));
            }
            return layerDelta;
        }
    }
}
