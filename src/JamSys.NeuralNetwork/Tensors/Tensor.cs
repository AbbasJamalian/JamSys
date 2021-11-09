#region License
/*
 * Copyright (c) 2020 - Abbas Jamalian
 * This file is part of JamSys Project and is licensed under the MIT License. 
 * For more details see the License file provided with the software
 */
#endregion License

using System;
using System.Globalization;
using System.Text;
using System.Text.Json.Serialization;

namespace JamSys.NeuralNetwork
{
    /// <summary>
    /// Tensor class representsd 1,2 or 3 dimensional arrays. The values are of type double
    /// </summary>
    public class Tensor
    {
        /// <summary>
        /// Three dimensional array of double values
        /// </summary>
        private double[,,] _values;

        /// <summary>
        /// Width of Tensor (x axis)
        /// </summary>
        public int Width { get; private set; }

        /// <summary>
        /// Depth of Tensor (y axis) - must be one for one-dimensional arrays
        /// </summary>
        public int Height { get; private set; }

        /// <summary>
        /// Depth of Tensor (z axis) - must be one for one or two dimensional arrays
        /// </summary>
        public int Depth { get; private set; }


        /// <summary>
        /// Shows if the internal array of Tensor is null or has values
        /// </summary>
        [JsonIgnore]
        public bool HasValues { get { return _values != null; }  }

        [JsonIgnore]
        public double this [int x,int y,int z]  
        { 
            get { return _values[x, y, z];  } 
            set 
            {
                Init();
                _values[x, y, z] = value; 
            } 
        }

        [JsonIgnore]
        public double this[int x, int y]
        {
            get { return this[x, y, 0]; }
            set { this[x, y, 0] = value; }
        }

        [JsonIgnore]
        public double this[int x]
        {
            get { return this[x, 0, 0]; }
            set { this[x, 0, 0] = value; }
        }

        public Tensor(int width, int height = 1, int depth = 1)
        {
            if (width >= 1 && height >= 1 && depth >= 1)
            {
                Width = width;
                Depth = depth;
                Height = height;
            }
            else
                throw new ArgumentOutOfRangeException("Invalid dimensions");
        }

        /// <summary>
        /// Provides the access to internal array of a Tensor
        /// </summary>
        /// <returns></returns>
        public double[,,] GetRaw()
        {
            return _values;
        }

        /// <summary>
        /// Copies the contents of the source Tensor to the current Tensor. Source Tensor must have the same dimensions
        /// </summary>
        /// <param name="source"></param>
        public void Copy(Tensor source)
        {
            if (Width != source.Width || Height != source.Height || Depth != source.Depth)
                throw new InvalidOperationException("the source Tensor must have the same dimension");

            if (source.HasValues)
            {
                _values = (double[,,])source.GetRaw().Clone();
            }
            else
            {
                _values = null;
            }
        }

        /// <summary>
        /// Adds the values of the operant Tensor to the current tensor
        /// </summary>
        /// <param name="operand"></param>
        public void Add(Tensor operand)
        {
            //TODO: Check for same dimensions
            Init();
            for (int x = 0; x < operand.Width; x++)
                for (int y = 0; y < operand.Height; y++)
                    for (int z = 0; z < operand.Depth; z++)
                        _values[x, y, z] += operand[x, y, z];

        }

        /// <summary>
        /// Creates and returns a Tensor with the same dimensions
        /// </summary>
        /// <returns></returns>
        public Tensor CreateSimilar()
        {
            return new Tensor(Width, Height, Depth);
        }

        private void Init()
        {
            if (_values == null)
                _values = new double[Width, Height, Depth];
        }

        /// <summary>
        /// Clears the internal array
        /// </summary>
        public void Clear()
        {
            if (_values != null)
                _values = null;
        }

        public override string ToString()
        {
            StringBuilder builder = new();
            for (int z = 0; z < this.Depth; z++)
            {
                if (z > 0) builder.Append(',');
                builder.Append('[');
                for (int y = 0; y < this.Height; y++)
                {
                    if(y > 0) builder.Append(',');
                    builder.Append('[');
                    for (int x = 0; x < this.Width; x++)
                    {
                        if(x > 0) builder.Append(',');
                        builder.Append(_values[x, y, z].ToString("F4", new CultureInfo("en-US")));
                    }
                    builder.Append(']');
                }
                builder.Append(']');
            }

            return builder.ToString();
        }
    }
}