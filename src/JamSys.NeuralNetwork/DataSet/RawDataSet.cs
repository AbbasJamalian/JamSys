#region License
/*
 * Copyright (c) 2020 - Abbas Jamalian
 * This file is part of JamSys Project and is licensed under the MIT License. 
 * For more details see the License file provided with the software
 */
#endregion License

using System;
using System.Collections.Generic;

namespace JamSys.NeuralNetwork.DataSet
{

    /// <summary>
    /// The basic implementation of the IDataSetProvider which works with raw Tensors as data and label. 
    /// </summary>
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

        public IDataSetProvider Add(Tensor data, Tensor label)
        {
            Data.Add(data);
            Labels.Add(label);
            return this;
        }

        public void Clear()
        {
            Data.ForEach(d => d.Clear());
            Labels.ForEach(l => l.Clear());
            Data.Clear();
            Labels.Clear();
        }

        public virtual (Tensor data, Tensor label) GetData(int index)
        {
            return new(Data[index], Labels[index]);
        }

        public virtual void Shuffle()
        {
            Random random = new();
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
