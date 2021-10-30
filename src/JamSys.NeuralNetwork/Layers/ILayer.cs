using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace JamSys.NeuralNetwork.Layers
{
    public interface ILayer : IDisposable
    {
        [JsonIgnore]
        public Tensor Input { get; set; }

        [JsonIgnore]
        public Tensor Output { get; }

        public void Build(ILayer previousLayer);

        public void Clear();

        void Run();

        Tensor Backpropagate(Tensor delta, double rate);

    }
}
