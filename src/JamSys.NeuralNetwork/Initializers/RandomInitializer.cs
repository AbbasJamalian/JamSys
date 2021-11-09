#region License
/*
 * Copyright (c) 2020 - Abbas Jamalian
 * This file is part of JamSys Project and is licensed under the MIT License. 
 * For more details see the License file provided with the software
 */
#endregion License

using System;

namespace JamSys.NeuralNetwork.Initializers
{
    /// <summary>
    /// generates a random value between -0.01 and +0.01
    /// </summary>
    class RandomInitializer : IInitializer
    {
        private readonly Random random;
        public RandomInitializer()
        {
            random = new Random();
        }

        public double GenerateInitialValue()
        {
            int value = random.Next(-10000, 10000);
            double result = ((double)value) / 1000000;
            return result;
        }
    }
}
