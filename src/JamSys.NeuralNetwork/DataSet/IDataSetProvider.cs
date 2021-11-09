#region License
/*
 * Copyright (c) 2020 - Abbas Jamalian
 * This file is part of JamSys Project and is licensed under the MIT License. 
 * For more details see the License file provided with the software
 */
#endregion License

namespace JamSys.NeuralNetwork.DataSet
{
    /// <summary>
    /// Represents a DataSet which is a set of data with their labels. Both data and labels are Tensors
    /// </summary>
    public interface IDataSetProvider
    {
        /// <summary>
        /// the number of items in the dataset
        /// </summary>
        public int Size { get;  }

        public (Tensor data, Tensor label) this[int x] { get; }

        /// <summary>
        /// Addas a new item to Dataset
        /// </summary>
        /// <param name="data">Data must match the input of network</param>
        /// <param name="label">Label must match the output of the network</param>
        /// <returns></returns>
        IDataSetProvider Add(Tensor data, Tensor label);

        /// <summary>
        /// Removes all items from dataset
        /// </summary>
        void Clear();

        /// <summary>
        /// randomly changes the sequence of items in dataset
        /// </summary>
        void Shuffle();
    }
}
