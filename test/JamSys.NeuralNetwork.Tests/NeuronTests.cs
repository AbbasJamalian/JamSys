using JamSys.NeuralNetwork.DataSet;
using JamSys.NeuralNetwork.Nodes;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;
using Xunit.Abstractions;

namespace JamSys.NeuralNetwork.Tests
{
    public class NeuronTests
    {
        private readonly ITestOutputHelper console;

        public NeuronTests(ITestOutputHelper output)
        {
            this.console = output;
        }

        [Fact]
        public void GeneralTest()
        {
            var input = new Tensor(3);
            var output = new Tensor(1);

            input[0] = 1;
            input[1] = 2;
            input[2] = 3;

            var neuron = new Neuron(ActivationFunctionEnum.Linear, 0);
            neuron.Build(input, output);

            //We just overwrite the weights and bias to 1
            neuron.Weights[0] = 1;
            neuron.Weights[1] = 0.5;
            neuron.Weights[2] = 1;
            neuron.Bias = 1;

            neuron.Run();

            Assert.Equal(6, output[0]);
        }

        [Fact]
        public void TrainNeuron()
        {

            var input = new Tensor(2);
            var output = new Tensor(1);

            IDataSetProvider dataSet = new RawDataSet();
            dataSet.Add(new Tensor(2) { [0] = 1, [1] = 1 }, new Tensor(1) { [0] = 1 });
            dataSet.Add(new Tensor(2) { [0] = 1, [1] = 0 }, new Tensor(1) { [0] = 0 });
            dataSet.Add(new Tensor(2) { [0] = 0, [1] = 1 }, new Tensor(1) { [0] = 0 });
            dataSet.Add(new Tensor(2) { [0] = 0, [1] = 0 }, new Tensor(1) { [0] = 0 });


            var neuron = new Neuron(ActivationFunctionEnum.Sigmoid, 0);
            neuron.Build(input, output);

            int epoches = 50000;
            double learningRate = 0.02;
            double delta = 0;

            for (int epoche = 0; epoche < epoches; epoche++)
            {
                for (int index = 0; index < dataSet.Size; index++)
                {
                    input.Copy(dataSet[index].data);
                    neuron.Run();

                    delta = output[0] - dataSet[index].label[0];
                    neuron.Backpropagate(delta, learningRate);
                }
            }

            //Test
            double totalError = 0;
            for (int index = 0; index < dataSet.Size; index++)
            {
                input.Copy(dataSet[index].data);
                neuron.Run();

                totalError += Math.Pow(output[0] - dataSet[index].label[0], 2);
            }
            totalError /= dataSet.Size;

            this.console.WriteLine($"Total Error = {totalError}");

            Assert.True(totalError < 0.01);
        }
    }
}
