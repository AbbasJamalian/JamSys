using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace JamSys.NeuralNetwork.Layers
{
    public class InputLayer : ILayer
    {
        [JsonIgnore]
        public Tensor Input { get; private set; }

        [JsonIgnore]
        public Tensor Output { get; private set; }

        public int Width { get; private set; }
        public int Height { get; private set; }
        public int Depth { get; private set; }

        public InputLayer(int width, int height = 1, int depth = 1)
        {
            Width = width;
            Depth = depth;
            Height = height;
        }

        public void Dispose()
        {
            if (Output != null)
            {
                Output.Dispose();
                Output = null;
            }

            //Because the Input Layer is the owner of input
            if (Input != null)
            {
                Input.Dispose();
                Input = null;
            }
        }

        public void Build(ILayer previousLayer)
        {
            Input = new Tensor(Width, Height, Depth);
            Output = Input;
        }

        public void Clear()
        {
            Input?.Clear();
            Output?.Clear();
        }

        public void Run()
        {
            //TODO: Check if Input matches the initialized dimensions
            Output = Input;
        }

        public Tensor Backpropagate(Tensor delta, double rate)
        {
            return delta;
        }
    }
}
