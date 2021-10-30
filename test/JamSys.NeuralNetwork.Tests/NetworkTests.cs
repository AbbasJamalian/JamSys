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
