#region License
/*
 * Copyright (c) 2020 - Abbas Jamalian
 * This file is part of JamSys Project and is licensed under the MIT License. 
 * For more details see the License file provided with the software
 */
#endregion License

namespace JamSys.NeuralNetwork.Initializers
{
    /// <summary>
    /// The Initializer is used to initialize the parameters like Bias and Weights randomly or based on a certain implementation
    /// </summary>
    public interface IInitializer
    {
        double GenerateInitialValue();
    }
}
