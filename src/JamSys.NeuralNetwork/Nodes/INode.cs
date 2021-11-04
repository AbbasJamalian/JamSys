using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace JamSys.NeuralNetwork.Nodes
{
    public interface INode
    {
        void Run();

        Tensor Backpropagate(double delta, double rate);
    }
}
