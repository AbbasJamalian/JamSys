using System;
using System.Collections.Generic;
using System.Text;

namespace JamSys.NeuralNetwork.Initializers
{
    public interface IInitializer
    {
        double GenerateInitialValue();
    }
}
