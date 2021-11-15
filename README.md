# JamSys Neural Network

The Vision of the JamSys is to provide a cloud based Neural Network framework to be used by business or IoT solutions. At its core, JamSys offers a Neural Network library which can be used stand-alone. The implementation is currently at early stages.

## Tensors
Tensor objects are used s input our output of the network as well as data and label of the datasets. A Tensor is a one, two or three dimensional array. 

## Usage
Here is a code sample how to create the network, create a datase, train network with a dataset to solve the XOR Problem and use the network

### Create Network
    var network = Factory.Instance.CreateNetwork()
                    .AddInputLayer(2)
                    .AddDenseLayer(3, ActivationFunctionEnum.Sigmoid)
                    .AddDenseLayer(1, ActivationFunctionEnum.Sigmoid)
                    .Build();

### Create DataSet and train the network
    	          //XOR Gate
                var dataset = Factory.Instance.CreateDataSet()
                        .Add(new Tensor(2) { [0] = 1, [1] = 1 }, new Tensor(1) { [0] = 0 })
                        .Add(new Tensor(2) { [0] = 1, [1] = 0 }, new Tensor(1) { [0] = 1 })
                        .Add(new Tensor(2) { [0] = 0, [1] = 1 }, new Tensor(1) { [0] = 1 })
                        .Add(new Tensor(2) { [0] = 0, [1] = 0 }, new Tensor(1) { [0] = 0 });
    
                var trainer = Factory.Instance.CreateTrainer()
                    .Configure(c =>
                    {
                        c.LossFunction = LossFunctionEnum.MSE;
                        c.LearningRate = 0.2;
                    });
    
                trainer.Train(network, dataset, 50000);
    
                //Validation
                double totalError = trainer.Validate(network, dataset);
                console.WriteLine($"Total Error: {totalError}");

### Use the network to predict values
                for (int index = 0; index < dataset.Size; index++)
                {
                    var output = network.Run(dataset[index].data);
                    console.WriteLine($"{output[0]}");
                }

## Load and Save the network
The network structure and parameters can be saved to a JSON string and can be loaded from JSON string. 

    	//Save the network 1
    	string saveResult = net1.Save();
    
    	//load it to a new network
            var net2 = Factory.Instance.CreateNetwork()
                        .Load(saveResult);

