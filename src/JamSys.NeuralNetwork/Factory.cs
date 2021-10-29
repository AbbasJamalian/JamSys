using JamSys.NeuralNetwork.ActivationFunctions;
using JamSys.NeuralNetwork.Initializers;
using JamSys.NeuralNetwork.LossFunctions;
using JamSys.NeuralNetwork.Training;
using System;
using System.Collections.Generic;
using System.Linq;

namespace JamSys.NeuralNetwork
{
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

        private List<FactoryEntry> _registeredTypes = new List<FactoryEntry>();

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

        public INetwork CreateNetwork()
        {
            return Resolve<INetwork>();
        }

        public ITrainer CreateTrainer(TrainerEnum trainer)
        {
            return ResolveNamed<ITrainer>(Enum.GetName(trainer));
        }

        public void Register<T>(Type implementingType, bool singleton = false)
        {
            RegisterNamed<T>(implementingType, typeof(T).Name, singleton);
        }

        public void Unregister<T>(Type serviceType)
        {
            _registeredTypes.Where(t => t.Service.GetType().Equals(serviceType)).Select(t => _registeredTypes.Remove(t));
        }

        public void RegisterNamed<T>(Type implementingType, string name, bool singleton = false)
        {
            _registeredTypes.Add(new FactoryEntry(typeof(T), implementingType, name, singleton));
        }

        public T Resolve<T>()
        {
            return ResolveNamed<T>("");
        }

        public T ResolveNamed<T>(Enum enumeration)
        {
            return ResolveNamed<T>(Enum.GetName(enumeration.GetType(), enumeration));
        }

        public T ResolveNamed<T>(string name)
        {
            if (string.IsNullOrEmpty(name))
                name = typeof(T).Name;
            var query = from t in _registeredTypes
                        where string.Compare(t.Service.Name, typeof(T).Name, true) == 0 && string.Compare(name, t.Name, true) == 0
                        select t;

            var entry = query.FirstOrDefault();

            if (entry.Implementation != null)
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
