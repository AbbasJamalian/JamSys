using System;
using Xunit;
using Xunit.Abstractions;

namespace JamSys.NeuralNetwork.Tests
{
    public class NetworkTests
    {
        private readonly ITestOutputHelper console;

        public NetworkTests(ITestOutputHelper output)
        {
            this.console = output;
        }

        [Fact]
        public void SaveAndLoadTest()
        {
            var network = Factory.Instance.CreateNetwork()
                .AddInputLayer(2)
                .AddDenseLayer(4, ActivationFunctionEnum.LeakyReLU)
                .AddDenseLayer(2, ActivationFunctionEnum.LeakyReLU)
                .AddSoftmaxLayer(2)
                .Build();

            Tensor input = new Tensor(2);

            input[0] = 1;
            input[1] = 2;

            var output = network.Run(input);

            string saveResult = network.Save();

            var net2 = Factory.Instance.CreateNetwork()
                .Load(saveResult);

            var out2 = net2.Run(input);

            Assert.Equal(0, string.Compare(output.ToString(), out2.ToString()));

        }

        [Fact]
        public void XorGateTraining()
        {

            var network = Factory.Instance.CreateNetwork()
                .AddInputLayer(2)
                .AddDenseLayer(3, ActivationFunctionEnum.Sigmoid)
                .AddDenseLayer(1, ActivationFunctionEnum.Sigmoid)
                .Build();

            //XOR Gate
            var ds = Factory.Instance.CreateDataSet()
                    .Add(new Tensor(2) { [0] = 1, [1] = 1 }, new Tensor(1) { [0] = 0 })
                    .Add(new Tensor(2) { [0] = 1, [1] = 0 }, new Tensor(1) { [0] = 1 })
                    .Add(new Tensor(2) { [0] = 0, [1] = 1 }, new Tensor(1) { [0] = 1 })
                    .Add(new Tensor(2) { [0] = 0, [1] = 0 }, new Tensor(1) { [0] = 0 });

            var trainer = Factory.Instance.CreateTrainer()
                .Configure(c =>
                {
                    c.LossFunction = LossFunctionEnum.MSE;
                    c.LearningRate = 0.2;
                });

            trainer.Train(network, ds, 50000);

            //Validation
            double totalError = trainer.Validate(network, ds);
            console.WriteLine($"Total Error: {totalError}");

            for (int index = 0; index < ds.Size; index++)
            {
                var output = network.Run(ds[index].data);
                console.WriteLine($"{output[0]}");
            }

            Assert.True(totalError < 0.001);
        }
    }
}
