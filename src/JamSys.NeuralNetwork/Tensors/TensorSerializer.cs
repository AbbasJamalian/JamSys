#region License
/*
 * Copyright (c) 2020 - Abbas Jamalian
 * This file is part of JamSys Project and is licensed under the MIT License. 
 * For more details see the License file provided with the software
 */
#endregion License

using System;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace JamSys.NeuralNetwork.Tensors
{
    /// <summary>
    /// Provides custome JSON serializer and deserializer for Tensors
    /// </summary>
    public class TensorSerializer : JsonConverter<Tensor>
    {
        public override bool CanConvert(Type typeToConvert)
        {
            return true;
        }

        public override Tensor Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            int width = 0;
            int height = 0;
            int depth = 0;

            Tensor value = null;
            while (reader.Read())
            {
                if (reader.TokenType == JsonTokenType.EndObject)
                {
                    break;
                }

                if (reader.TokenType == JsonTokenType.PropertyName)
                {
                    var propName = reader.GetString().ToLower();

                    switch (propName)
                    {
                        case "width":
                            reader.Read();
                            width = reader.GetInt32();
                            break;
                        case "depth":
                            reader.Read();
                            depth = reader.GetInt32();
                            break;
                        case "height":
                            reader.Read();
                            height = reader.GetInt32();
                            break;
                        case "values":
                            value = new Tensor(width, height, depth);
                            ReadArray(reader, value);
                            break;
                        default:
                            break;
                    }
                }
            }
            return value;
        }

        private void ReadArray(Utf8JsonReader reader, Tensor value)
        {

            for (int z = 0; z < value.Depth; z++)
            {
                for (int y = 0; y < value.Height; y++)
                {
                    for (int x = 0; x < value.Width; x++)
                    {
                        while (reader.Read())
                        {
                            if (reader.TokenType == JsonTokenType.Number)
                            {
                                value[x, y, z] = reader.GetDouble();
                                break;
                            }
                        }
                    }
                }
            }
        }

        public override void Write(Utf8JsonWriter writer, Tensor value, JsonSerializerOptions options)
        {
            writer.WriteStartObject();

            writer.WriteNumber("Width", value.Width);
            writer.WriteNumber("Depth", value.Depth);
            writer.WriteNumber("Height", value.Height);

            writer.WritePropertyName("Values");
            writer.WriteStartArray();

            for (int z = 0; z < value.Depth; z++)
            {
                for (int y = 0; y < value.Height; y++)
                {
                    for (int x = 0; x < value.Width; x++)
                    {
                        writer.WriteNumberValue(value[x, y, z]);
                    }
                }
            }
            writer.WriteEndArray();

            writer.WriteEndObject();

        }
    }
}
