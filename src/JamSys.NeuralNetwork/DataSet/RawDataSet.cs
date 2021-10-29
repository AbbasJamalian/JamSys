using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JamSys.NeuralNetwork.DataSet
{
    public class RawDataSet : IDataSetProvider
    {
        protected List<Tensor> Data;
        protected List<Tensor> Labels;

        public (Tensor data, Tensor label) this [int x]
        {
            get { return GetData(x); }
        }

        public int Size => GetSize();

        public RawDataSet()
        {
            Data = new List<Tensor>();
            Labels = new List<Tensor>();
        }

        public void Add(Tensor data, Tensor label)
        {
            Data.Add(data);
            Labels.Add(label);
        }

        public void Clear()
        {
            Data.ForEach(d => d.Clear());
            Labels.ForEach(l => l.Clear());
            Data.Clear();
            Labels.Clear();
        }

        public void Dispose()
        {
            if (Data != null)
            {
                Data.ForEach(d => d.Dispose());
                Labels.ForEach(l => l.Dispose());

                Data.Clear();
                Data = null;
                Labels.Clear();
                Labels = null;
            }
        }

        public virtual (Tensor data, Tensor label) GetData(int index)
        {
            return new(Data[index], Labels[index]);
        }

        public virtual void Shuffle()
        {
            Random random = new Random();
            for (var i = Size - 1; i >= 0; i--)
            {
                var j = random.Next(i);

                var tempData = Data[j];
                var tempLabel = Labels[j];

                Data[j] = Data[i];
                Labels[j] = Labels[i];

                Data[j] = tempData;
                Labels[j] = tempLabel;
            }
        }

        public virtual int GetSize()
        {
            return Data.Count;
        }
    }
}
