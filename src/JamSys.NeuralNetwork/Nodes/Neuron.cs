using JamSys.NeuralNetwork.ActivationFunctions;
using JamSys.NeuralNetwork.Tensors;
using JamSys.NeuralNetwork.Initializers;
using System;
using System.Text.Json.Serialization;

namespace JamSys.NeuralNetwork.Nodes
{
    public class Neuron : INode
    {
        private Tensor _input;
        private Tensor _output;

        private IActivationFunction _activationFunction;


        public ActivationFunctionEnum ActivationType { get; private set; }

        public int Index { get; private set; }

        [JsonConverter(typeof(TensorSerializer))]
        public Tensor Weights { get; set; }

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
        /// Input and Output must be single dimensional Tensors
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

        public double GetDotProduct()
        {
            double result = 0;
            for (int x = 0; x < _input.Width; x++)
                    result += Weights[x] * _input[x];

            result += Bias;

            if (Double.IsNaN(result))
                throw new ArithmeticException("Neuron output is NaN");

            return result;

        }

        public void Run()
        {
            _output[Index] = _activationFunction.Calculate(GetDotProduct());
        }

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
            Bias = Bias - (rate * gradient);

            return neuronDelta;
        }
    }
}
