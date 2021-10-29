using JamSys.NeuralNetwork.DataSet;
using JamSys.NeuralNetwork.LossFunctions;

namespace JamSys.NeuralNetwork.Training
{
    public class SGDTrainer : ITrainer
    {
        private int _batchSize;
        private double _learningRate;
        private ILossFunction _lossFunction;

        public SGDTrainer()
        {
        }


        public ITrainer Initialize(int batchSize, double learningRate, LossFunctionEnum lossFunction)
        {
            _batchSize = batchSize;
            _learningRate = learningRate;
            _lossFunction = Factory.Instance.ResolveNamed<ILossFunction>(lossFunction);

            return this;
        }

        public void Dispose()
        {
        }

        public void Train(INetwork network, IDataSetProvider dataSet, int epoches)
        {
            //TODO: Check Input and Output sizes with the dataset data and label sizes
            for (int epoche = 0; epoche < epoches; epoche++)
            {
                dataSet.Shuffle();
                for (int index = 0; index < dataSet.Size; index += _batchSize)
                {
                    var delta = network.Output.CreateSimilar();
                    for (int x = 0; x < delta.Width; x++)
                        delta[x] = 0;


                    for (int offset = 0; offset < _batchSize && offset + index < dataSet.Size; offset++)
                    {
                        var output = network.Run(dataSet[index + offset].data);

                            delta.Add(_lossFunction.CalculateGradients(output, dataSet[index + offset].label));
                    }

                    for (int x = 0; x < delta.Width; x++)
                        delta[x] = (delta[x] / _batchSize);

                    network.Backpropagate(delta, _learningRate);
                    delta.Dispose();
                }
            }
        }

        public double Validate(INetwork network, IDataSetProvider dataSet)
        {
            double totalError = 0;

            for (int index = 0; index < dataSet.Size; index++)
            {
                var output = network.Run(dataSet[index].data);

                totalError += _lossFunction.CalculateTotalLoss(output, dataSet[index].label);
            }

            totalError /= dataSet.Size;

            return totalError;
        }
    }
}
