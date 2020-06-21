import argparse

from ctranslate2 import converters, specs
import json


def get_bert_model_spec(config):
    return specs.bert_spec.BertSpec(
        num_layers=config["num_hidden_layers"],
        num_heads=config["num_attention_heads"])


def main():
    parser = argparse.ArgumentParser(
        description="Release an OpenNMT-py model for inference")
    parser.add_argument("--model", "-m",
                        help="The model file", required=True)
    parser.add_argument("--config", "-c",
                        help="The model config file", required=True)
    parser.add_argument("--output", "-o",
                        help="The output path", required=True)
    parser.add_argument("--quantization", "-q",
                        choices=["int8", "int16"],
                        default=None,
                        help="Quantization type for Bert model.")
    opt = parser.parse_args()

    def _load_config(config):
        with open(config) as json_file:
            return json.load(json_file)

    config = _load_config(opt.config)
    model_spec = get_bert_model_spec(config)

    converters.TransformersConverter(opt.model).convert(opt.output, model_spec,
                                                        force=True,
                                                        quantization=opt.quantization)


if __name__ == "__main__":
    main()
