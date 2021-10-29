using System;
using System.Globalization;
using System.Text;
using System.Text.Json.Serialization;

namespace JamSys.NeuralNetwork
{
    public class Tensor : IDisposable
    {
        [JsonIgnore]
        public double[,,] Values { get; private set; }

        public int Width { get; private set; }
        public int Depth { get; private set; }
        public int Height { get; private set; }

        [JsonIgnore]
        public bool HasValues { get { return Values != null; }  }

        [JsonIgnore]
        public double this [int x,int y,int z]  
        { 
            get { return Values[x, y, z];  } 
            set 
            {
                Init();
                Values[x, y, z] = value; 
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
            if (width < 1 || height < 1 || depth < 1)
                throw new ArgumentOutOfRangeException("width, height and depth must be bigger than one");

            Width = width;
            Depth = depth;
            Height = height;
        }

        public double[,,] GetRaw()
        {
            return Values;
        }

        public void Copy(Tensor source)
        {
            if (Width != source.Width || Height != source.Height || Depth != source.Depth)
                throw new InvalidOperationException("the source Tensor must have the same dimension");

            if (source.HasValues)
            {
                Values = (double[,,])source.GetRaw().Clone();
            }
            else
            {
                Values = null;
            }
        }

        public void Add(Tensor operand)
        {
            //TODO: Check for same dimensions
            Init();
            for (int x = 0; x < operand.Width; x++)
                for (int y = 0; y < operand.Height; y++)
                    for (int z = 0; z < operand.Depth; z++)
                        Values[x, y, z] += operand[x, y, z];

        }

        public Tensor CreateSimilar()
        {
            return new Tensor(Width, Height, Depth);
        }

        private void Init()
        {
            if (Values == null)
                Values = new double[Width, Height, Depth];
        }

        public void Clear()
        {
            if (Values != null)
                Values = null;
        }

        public void Dispose()
        {
            Clear();
        }

        public override string ToString()
        {
            StringBuilder builder = new StringBuilder();
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
                        builder.Append(Values[x, y, z].ToString("F4", new CultureInfo("en-US")));
                    }
                    builder.Append(']');
                }
                builder.Append(']');
            }

            return builder.ToString();
        }
    }
}