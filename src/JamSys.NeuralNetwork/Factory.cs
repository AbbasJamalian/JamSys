#region License
/*
 * Copyright (c) 2020 - Abbas Jamalian
 * This file is part of JamSys Project and is licensed under the MIT License. 
 * For more details see the License file provided with the software
 */
#endregion License

using JamSys.NeuralNetwork.ActivationFunctions;
using JamSys.NeuralNetwork.DataSet;
using JamSys.NeuralNetwork.Initializers;
using JamSys.NeuralNetwork.LossFunctions;
using JamSys.NeuralNetwork.Training;
using System;
using System.Collections.Generic;
using System.Linq;

namespace JamSys.NeuralNetwork
{
    /// <summary>
    /// Factory is a singleton class which can register services and resolve these services by name or type
    /// </summary>
    public class Factory
    {
        private static Factory _instance = null;

        class FactoryEntry
        {
            public Type Service { get; set; }
            public Type Implementation { get; set; }
            public string Name { get; set; }
            public bool Singleton { get; set; }
            public object Instance { get; set; }

            public FactoryEntry(Type service, Type implementation, string name, bool singleton)
            {
                Service = service;
                Implementation = implementation;
                Name = name;
                Singleton = singleton;
            }
        }

        private readonly List<FactoryEntry> _registeredTypes = new();

        public static Factory Instance 
        { 
            get 
            {
                if (_instance == null)
                    _instance = new Factory();
                return _instance;
            } 
        }

        private Factory()
        {
            Register<INetwork>(typeof(Network.Network));
            Register<IDataSetProvider>(typeof(RawDataSet));
            Register<IInitializer>(typeof(RandomInitializer), true);


            RegisterNamed<IActivationFunction>(typeof(LinearActivation), Enum.GetName(ActivationFunctionEnum.Linear));
            RegisterNamed<IActivationFunction>(typeof(SigmoidActivation), Enum.GetName(ActivationFunctionEnum.Sigmoid));
            RegisterNamed<IActivationFunction>(typeof(ReLUActivation), Enum.GetName(ActivationFunctionEnum.ReLU));
            RegisterNamed<IActivationFunction>(typeof(LeakyReLUActivation), Enum.GetName(ActivationFunctionEnum.LeakyReLU));
            RegisterNamed<IActivationFunction>(typeof(TanhActivation), Enum.GetName(ActivationFunctionEnum.Tanh));

            RegisterNamed<ILossFunction>(typeof(MseLoss), Enum.GetName(LossFunctionEnum.MSE));
            RegisterNamed<ILossFunction>(typeof(CrossEntropyLoss), Enum.GetName(LossFunctionEnum.CrossEntropy));

            RegisterNamed<ITrainer>(typeof(SGDTrainer), Enum.GetName(TrainerEnum.SGDTrainer));

        }

        /// <summary>
        /// Resolves the INetwork interface
        /// </summary>
        /// <returns>implementation of the INetwork</returns>
        public INetwork CreateNetwork()
        {
            return Resolve<INetwork>();
        }

        /// <summary>
        /// Resolves ITrainer interface. Currently only SGDTrainer is supported
        /// </summary>
        /// <returns></returns>
        public ITrainer CreateTrainer()
        {
            return ResolveNamed<ITrainer>(TrainerEnum.SGDTrainer);
        }

        /// <summary>
        /// Creates a generic dataset provider
        /// </summary>
        /// <returns></returns>
        public IDataSetProvider CreateDataSet()
        {
            return Resolve<IDataSetProvider>();
        }

        /// <summary>
        /// registers a service using default implementation type
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="implementingType"></param>
        /// <param name="singleton"></param>
        public void Register<T>(Type implementingType, bool singleton = false)
        {
            RegisterNamed<T>(implementingType, typeof(T).Name, singleton);
        }

        /// <summary>
        /// Registers a service based on multiple implementation types
        /// </summary>
        /// <typeparam name="T">Service Type</typeparam>
        /// <param name="implementingType">Implementation type</param>
        /// <param name="name">named instance</param>
        /// <param name="singleton">true if the service is singleton</param>
        public void RegisterNamed<T>(Type implementingType, string name, bool singleton = false)
        {
            _registeredTypes.Add(new FactoryEntry(typeof(T), implementingType, name, singleton));
        }

        /// <summary>
        /// Resolves the service by default type
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <returns></returns>
        public T Resolve<T>()
        {
            return ResolveNamed<T>("");
        }

        /// <summary>
        /// uses the enumeration to generate a name to resolve the service
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="enumeration"></param>
        /// <returns></returns>
        public T ResolveNamed<T>(Enum enumeration)
        {
            return ResolveNamed<T>(Enum.GetName(enumeration.GetType(), enumeration));
        }

        /// <summary>
        /// Resolves a service based on the implementation type name
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="name"></param>
        /// <returns></returns>
        public T ResolveNamed<T>(string name)
        {
            if (string.IsNullOrEmpty(name))
                name = typeof(T).Name;
            var query = from t in _registeredTypes
                        where string.Compare(t.Service.Name, typeof(T).Name, true) == 0 && string.Compare(name, t.Name, true) == 0
                        select t;

            var entry = query.FirstOrDefault();

            if (entry?.Implementation != null)
            {
                T instance;
                if (entry.Singleton)
                {
                    if(entry.Instance == null)
                        entry.Instance = Activator.CreateInstance(entry.Implementation);

                    instance = (T)entry.Instance;
                }
                else
                {
                    instance = (T)Activator.CreateInstance(entry.Implementation);
                }

                return instance;
            }
            else
                throw new ArgumentException("Unable to find the type or the named type");
        }
    }
}
