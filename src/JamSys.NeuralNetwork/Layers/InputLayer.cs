#region License
/*
 * Copyright (c) 2020 - Abbas Jamalian
 * This file is part of JamSys Project and is licensed under the MIT License. 
 * For more details see the License file provided with the software
 */
#endregion License

using System.Text.Json.Serialization;

namespace JamSys.NeuralNetwork.Layers
{
    public class InputLayer : ILayer
    {
        [JsonIgnore]
        public Tensor Input { get;  private set; }

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

        /// <summary>
        /// 
        /// </summary>
        /// <param name="previousLayer">Input Layer ignores this parameter. it can be null.</param>
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
            Output = Input;
        }

        public Tensor Backpropagate(Tensor delta, double rate)
        {
            return delta;
        }
    }
}
