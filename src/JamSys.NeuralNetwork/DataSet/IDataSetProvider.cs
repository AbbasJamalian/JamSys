using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JamSys.NeuralNetwork.DataSet
{
    public interface IDataSetProvider : IDisposable
    {
        public int Size { get;  }

        public (Tensor data, Tensor label) this[int x] { get; }

        void Add(Tensor data, Tensor label);

        void Clear();

        void Shuffle();
    }
}
