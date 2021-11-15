using JamSys.NeuralNetwork;
using JamSys.NeuralNetwork.Layers;
using System.Text.Json;
using Xunit;
using Xunit.Abstractions;

namespace JamSys.NeuralNetwork.Tests
{
    public class LayerTests
    {
        private readonly ITestOutputHelper console;

        public LayerTests(ITestOutputHelper output)
        {
            this.console = output;
        }

        [Fact]
        public void SoftmaxLayerTests()
        {
            var inputLayer = new InputLayer(3);
            inputLayer.Build(null);

            inputLayer.Input[0] = 1;
            inputLayer.Input[1] = 2;
            inputLayer.Input[2] = 3;

            var softmaxLayer = new SoftmaxLayer(3);

            softmaxLayer.Build(inputLayer);

            softmaxLayer.Run();

            double result = softmaxLayer.Output[0] + softmaxLayer.Output[1] + softmaxLayer.Output[2];
            Assert.Equal(1.0, result);
        }

        [Fact]
        public void InputLayerSerialization()
        {
            ILayer inputLayer = new InputLayer(3);

            JsonSerializerOptions options = new()
            {
                WriteIndented = true,
                //Encoder = System.Text.Encodings.Web.JavaScriptEncoder.Create(UnicodeRanges.BasicLatin),
                Converters = { new LayerSerializer() },
            };

            string result = JsonSerializer.Serialize(inputLayer, options);

            var loadedLayer = JsonSerializer.Deserialize<ILayer>(result, options);

            Assert.NotNull(loadedLayer);
            Assert.True(loadedLayer is InputLayer);

            Assert.Equal(((InputLayer)inputLayer).Width, ((InputLayer)loadedLayer).Width);
            Assert.Equal(((InputLayer)inputLayer).Height, ((InputLayer)loadedLayer).Height);
            Assert.Equal(((InputLayer)inputLayer).Depth, ((InputLayer)loadedLayer).Depth);

        }

        [Fact]
        public void DenseLayerSerialization()
        {
            ILayer inputLayer = new InputLayer(2);
            inputLayer.Build(null);

            ILayer layer = new DenseLayer(2, ActivationFunctionEnum.Sigmoid);
            layer.Build(inputLayer);

            JsonSerializerOptions options = new()
            {
                WriteIndented = true,
                Converters = { new LayerSerializer() },
            };

            string result = JsonSerializer.Serialize(layer, options);

            var loadedLayer = JsonSerializer.Deserialize<ILayer>(result, options);

            string loadedResult = JsonSerializer.Serialize(loadedLayer, options);

            Assert.Equal(0, string.Compare(result, loadedResult));
        }
    }
}
