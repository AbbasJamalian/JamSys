using JamSys.NeuralNetwork;
using JamSys.NeuralNetwork.ActivationFunctions;
using JamSys.NeuralNetwork.Initializers;
using System;
using Xunit;

namespace JamSys.NeuralNetwork.Tests
{
    public class FactoryTests
    {
        [Fact]
        public void CreateNetwork()
        {
            var network = Factory.Instance.CreateNetwork();
            Assert.NotNull(network);
            Assert.True(network is INetwork);
        }

        [Fact]
        public void SigmoidActivationFunctionTest()
        {
            var sigmoid = Factory.Instance.ResolveNamed<IActivationFunction>(ActivationFunctionEnum.Sigmoid);
            Assert.NotNull(sigmoid);
        }

        [Fact]
        public void SingletonTest()
        {
            //IRandomGenerator is registered as singleton
            var service1 = Factory.Instance.Resolve<IInitializer>();
            var service2 = Factory.Instance.Resolve<IInitializer>();
            Assert.True(Object.ReferenceEquals(service1, service2));

            //That should not be the case for network
            var network1 = Factory.Instance.Resolve<INetwork>();
            var network2 = Factory.Instance.Resolve<INetwork>();
            Assert.False(Object.ReferenceEquals(network1, network2));

        }
    }
}
