using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace JamSys.NeuralNetwork.Layers
{
    public class LayerSerializer : JsonConverter<ILayer>
    {
        public override ILayer Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            ILayer layer = null;
            Type layerType = typeof(ILayer);

            while (reader.Read())
            {
                if (reader.TokenType == JsonTokenType.EndObject)
                {
                    break;
                }

                if (reader.TokenType == JsonTokenType.PropertyName)
                {
                    var propName = reader.GetString().ToLower();
                    if (string.Compare(propName, "type", true) == 0)
                    {
                        reader.Read();
                        var typeName = reader.GetString().ToLower();
                        switch (typeName)
                        {
                            case "inputlayer":
                                layerType = typeof(InputLayer);
                                break;
                            case "denselayer":
                                layerType = typeof(DenseLayer);
                                break;
                            case "softmaxlayer":
                                layerType = typeof(SoftmaxLayer);
                                break;
                        }
                    }
                    else if (string.Compare(propName, "layer", true) == 0)
                    {
                        reader.Read();
                        layer = JsonSerializer.Deserialize(ref reader, layerType) as ILayer;
                    }
                }
            }
            return layer;
        }

        public override void Write(Utf8JsonWriter writer, ILayer value, JsonSerializerOptions options)
        {
            writer.WriteStartObject();
            writer.WriteString("Type", value.GetType().Name);
            writer.WritePropertyName("Layer");

            switch (value)
            {
                case InputLayer layer:
                    JsonSerializer.Serialize<InputLayer>(writer, layer, options);
                    break;
                case DenseLayer layer:
                    JsonSerializer.Serialize<DenseLayer>(writer, layer, options);
                    break;
                case SoftmaxLayer layer:
                    JsonSerializer.Serialize<SoftmaxLayer>(writer, layer, options);
                    break;
            }
            writer.WriteEndObject();
        }
    }
}
