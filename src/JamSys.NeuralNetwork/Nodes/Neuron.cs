#region License
/*
 * Copyright (c) 2020 - Abbas Jamalian
 * This file is part of JamSys Project and is licensed under the MIT License. 
 * For more details see the License file provided with the software
 */
#endregion License

using JamSys.NeuralNetwork.ActivationFunctions;
using JamSys.NeuralNetwork.Tensors;
using JamSys.NeuralNetwork.Initializers;
using System;
using System.Text.Json.Serialization;

namespace JamSys.NeuralNetwork.Nodes
{
    /// <summary>
    /// Implementation of a Neuron 
    /// </summary>
    public class Neuron
    {
        private Tensor _input;
        private Tensor _output;

        private IActivationFunction _activationFunction;


        /// <summary>
        /// This enumeration specifies which activation function will be used by neuron
        /// </summary>
        public ActivationFunctionEnum ActivationType { get; private set; }

        public int Index { get; private set; }

        /// <summary>
        /// A Tensor for the weights. This Tensor must be one-dimensional
        /// </summary>
        [JsonConverter(typeof(TensorSerializer))]
        public Tensor Weights { get; set; }

        /// <summary>
        /// The Bias of the neuron
        /// </summary>
        public double Bias { get; set; }

        /// <summary>
        /// constructor of a Neuron
        /// </summary>
        /// <param name="activationType"></param>
        /// <param name="index">the out put of the Neuron will be written to the Output[index]</param>
        public Neuron(ActivationFunctionEnum activationType, int index)
        {
            ActivationType = activationType;
            Index = index;
        }

        /// <summary>
        /// Initializes the Neuron. Input and Output must be single dimensional Tensors
        /// </summary>
        /// <param name="input"></param>
        /// <param name="output"></param>
        public void Build(Tensor input, Tensor output)
        {
            _input = input;
            _output = output;

            _activationFunction = Factory.Instance.ResolveNamed<IActivationFunction>(ActivationType);

            if (Weights == null)
            {
                Weights = input.CreateSimilar();
                InitializeRandom();
            }
        }

        private void InitializeRandom()
        {
            var _random = Factory.Instance.Resolve<IInitializer>();

            Bias = _random.GenerateInitialValue();

            for (int x = 0; x < Weights.Width; x++)
                    Weights[x] = _random.GenerateInitialValue();
        }

        private double GetDotProduct()
        {
            double result = 0;
            for (int x = 0; x < _input.Width; x++)
                    result += Weights[x] * _input[x];

            result += Bias;

            if (Double.IsNaN(result))
                throw new ArithmeticException("Neuron output is NaN");

            return result;

        }

        /// <summary>
        /// Fetches the data from input Tensor and using Bias and Weights and Activation Function calculates the output which will be written directly to output Tensor
        /// </summary>
        public void Run()
        {
            _output[Index] = _activationFunction.Calculate(GetDotProduct());
        }

        /// <summary>
        /// Performs a backpropagation on a neuron during training
        /// </summary>
        /// <param name="delta">the gradients</param>
        /// <param name="rate">learning rate</param>
        /// <returns></returns>
        public Tensor Backpropagate(double delta, double rate)
        {
            Tensor neuronDelta = Weights.CreateSimilar();

            double gradient = _activationFunction.CalculateDerivative(GetDotProduct()) * delta;

            for (int index = 0; index < neuronDelta.Width; index++)
            {
                //Calculate delta to be passed to the previous layer
                neuronDelta[index] = gradient * _input[index] * Weights[index];

                //Update weights
                Weights[index] = Weights[index] - (rate * gradient * _input[index]);
            }
            Bias -= rate * gradient;

            return neuronDelta;
        }
    }
}
