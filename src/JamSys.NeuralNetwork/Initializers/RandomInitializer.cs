using System;
using System.Collections.Generic;
using System.Text;

namespace JamSys.NeuralNetwork.Initializers
{
    class RandomInitializer : IInitializer
    {
        private Random random;
        public RandomInitializer()
        {
            random = new Random();
        }
        /// <summary>
        /// return a random value between -0.01 and +0.01
        /// </summary>
        /// <returns></returns>
        public double GenerateInitialValue()
        {
            int value = random.Next(-10000, 10000);
            double result = ((double)value) / 1000000;
            return result;
        }
    }
}
