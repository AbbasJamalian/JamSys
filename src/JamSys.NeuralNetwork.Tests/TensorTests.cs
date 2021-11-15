using JamSys.NeuralNetwork.Tensors;
using System;
using System.Text.Json;
using Xunit;

namespace JamSys.NeuralNetwork.Tests
{
    public class TensorTests
    {
        [Fact]
        public void TensorBasicTest()
        {
            var tensor = new Tensor(24, 4, 53);

            Assert.Equal(24, tensor.Width);
            Assert.Equal(4, tensor.Height);
            Assert.Equal(53, tensor.Depth);

            tensor.Clear();
        }

        [Theory]
        [InlineData(0, 0, 0, 666)]
        [InlineData(23, 3, 52, 1974)]
        [InlineData(10, 1, 15, 2021)]
        public void ReadWriteValues(int x, int y, int z, double value)
        {
            var tensor = new Tensor(24, 4, 53);
            tensor[x, y, z] = value;
            Assert.Equal(value, tensor[x, y, z]);
        }

        [Theory]
        [InlineData(24, 0, 0)]
        [InlineData(0, 4, 0)]
        [InlineData(0, 0, 53)]
        public void OutOfRangeCheck(int x, int y, int z)
        {
            var tensor = new Tensor(24, 4, 53);
            Assert.Throws<IndexOutOfRangeException>(() => tensor[x, y, z] = 666);
        }

        [Theory]
        [InlineData(0, 0, 0)]
        [InlineData(-1, 4, 5)]
        [InlineData(1, 1, 0)]
        [InlineData(1, -1, 1)]
        public void ConstructorCheck(int x, int y, int z)
        {
            Assert.Throws<ArgumentOutOfRangeException>(() => new Tensor(x, y, z));
        }

        [Fact]
        public void TestCopy()
        {
            Random _rand = new Random();
            var tensor = new Tensor(24, 4, 53);

            for (int x = 0; x < tensor.Width; x++)
                for (int y = 0; y < tensor.Height; y++)
                    for (int z = 0; z < tensor.Depth; z++)
                        tensor[x, y, z] = _rand.NextDouble();

            var copy = new Tensor(24, 4, 53);

            copy.Copy(tensor);

            for (int x = 0; x < tensor.Width; x++)
                for (int y = 0; y < tensor.Height; y++)
                    for (int z = 0; z < tensor.Depth; z++)
                        Assert.Equal(tensor[x, y, z], copy[x, y, z]);

            //Changing the copy should not affect the source
            copy[1, 3, 2] = copy[1, 3, 2] + 74;
            Assert.NotEqual(tensor[1, 3, 2], copy[1, 3, 2]);

        }


        [Fact]
        public void TestSerialization()
        {
            var tensor = new Tensor(2, 3, 4);

            for (int x = 0; x < tensor.Width; x++)
                for (int y = 0; y < tensor.Height; y++)
                    for (int z = 0; z < tensor.Depth; z++)
                        tensor[x, y, z] = x+y+z;

            var options = new JsonSerializerOptions();
            options.Converters.Add(new TensorSerializer());

            string jsonString = JsonSerializer.Serialize(tensor, options);

            var result = JsonSerializer.Deserialize<Tensor>(jsonString, options);

            Assert.Equal(tensor.Width, result.Width);
            Assert.Equal(tensor.Height, result.Height);
            Assert.Equal(tensor.Depth, result.Depth);

            for (int x = 0; x < result.Width; x++)
                for (int y = 0; y < result.Height; y++)
                    for (int z = 0; z < result.Depth; z++)
                        Assert.Equal(x + y + z, result[x, y, z]);

        }
    }

}
