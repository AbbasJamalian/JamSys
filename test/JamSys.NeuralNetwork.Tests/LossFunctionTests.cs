using JamSys.NeuralNetwork.LossFunctions;
using JamSys.NeuralNetwork.Training;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Xunit;
using Xunit.Abstractions;

namespace JamSys.NeuralNetwork.Tests
{
    public class LossFunctionTests
    {
        private readonly ITestOutputHelper console;

        public LossFunctionTests(ITestOutputHelper output)
        {
            this.console = output;
        }

        [Fact]
        public void CrossEntropyTests()
        {
            var crossEntropy = new CrossEntropyLoss();

            Tensor values = new Tensor(3);
            Tensor labels = new Tensor(3);

            values[0] = 0.2;
            values[1] = 0.6;
            values[2] = 0.2;

            labels[0] = 1;
            labels[1] = 0;
            labels[2] = 0;

            double error = crossEntropy.CalculateTotalLoss(values, labels);

            //Result should be 1.6094
            Assert.InRange<double>(error, 1.6093, 1.6095);
        }

    }
}
