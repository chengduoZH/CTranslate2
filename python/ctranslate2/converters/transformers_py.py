from ctranslate2.converters import utils
from ctranslate2.converters.converter import Converter
from ctranslate2.specs import common_spec
from ctranslate2.specs import bert_spec
from ctranslate2.specs.model_spec import ModelSpec
import os
import shutil


class TransformersConverter(Converter):
    """Converts models generated by Transformers."""

    def __init__(self, model_path):
        self._model_path = model_path

    def _save_vocabulary(self, vocab, output_path):
        with open(output_path, "wb") as output_file:
            for word in vocab.itos:
                word = word.encode("utf-8")
                output_file.write(word)
                output_file.write(b"\n")

    def _load(self, model_spec):
        import torch
        variables = torch.load(self._model_path, map_location="cpu")
        variables["num_heads"] = model_spec.num_heads
        if isinstance(model_spec, bert_spec.BertSpec):
            set_transformer_spec(model_spec, variables)
        else:
            raise NotImplementedError()
        return None, None

    def convert(self, output_dir, model_spec, vmap=None, quantization=None,
                force=False):
        if os.path.exists(output_dir) and not force:
            raise RuntimeError(
                "output directory %s already exists, use --force to override" % output_dir)
        if not isinstance(model_spec, ModelSpec):
            raise TypeError(
                "model_spec should extend ctranslate2.specs.ModelSpec")
        try:
            self._load(model_spec)
        except NotImplementedError:
            raise NotImplementedError(
                "This converter does not support the model %s" % model_spec)
        model_spec.validate()
        model_spec.optimize(quantization=quantization)

        # Create model directory.
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)
        model_spec.serialize(os.path.join(output_dir, "model.bin"))
        return output_dir


def set_transformer_spec(spec, variables):
    set_transformer_encoder(spec.encoder, variables)


def set_transformer_encoder(spec, variables, relative=False):
    set_input_layers(spec.bert_embedding_layer, variables, "encoder",
                     relative=relative)
    set_layer_norm(spec.bert_embedding_layer.layer_norm, variables,
                   "embeddings.LayerNorm")
    for i, layer in enumerate(spec.layer):
        set_transformer_encoder_layer(
            layer, variables, "encoder.layer.%d" % i, relative=relative)


def set_input_layers(spec, variables, scope, relative=False):
    set_position_encodings(
        spec.position_encodings, variables,
        "embeddings.position_embeddings.weight")
    set_embeddings(
        spec.embeddings,
        variables,
        "embeddings.word_embeddings.weight",
        multiply_by_sqrt_depth=False)
    set_token_type_encodings(spec.token_type_encodings, variables,
                             "embeddings.token_type_embeddings.weight")


def set_transformer_encoder_layer(spec, variables, scope, relative=False):
    set_multi_head_attention(
        spec.self_attention,
        variables,
        "%s.attention" % scope,
        self_attention=True,
        relative=relative)
    set_layer_norm(spec.self_attention.layer_norm, variables,
                   "%s.attention.output.LayerNorm" % scope)
    set_Intermediate(spec.ffn, variables, scope)


def set_Intermediate(spec, variables, scope):
    set_linear(spec.linear_0, variables, "%s.intermediate.dense" % scope)
    set_linear(spec.linear_1, variables, "%s.output.dense" % scope)
    set_layer_norm(spec.layer_norm, variables, "%s.output.LayerNorm" % scope)


def set_multi_head_attention(spec, variables, scope, self_attention=False,
                             relative=False):
    split_layers = [common_spec.LinearSpec() for _ in range(3)]
    set_linear(split_layers[0], variables, "%s.self.query" % scope)
    set_linear(split_layers[1], variables, "%s.self.key" % scope)
    set_linear(split_layers[2], variables, "%s.self.value" % scope)
    utils.fuse_linear(spec.linear[0], split_layers)
    set_linear(spec.linear[-1], variables, "%s.output.dense" % scope)


def set_layer_norm(spec, variables, scope):
    spec.gamma = _get_variable(variables, "%s.weight" % scope)
    spec.beta = _get_variable(variables, "%s.bias" % scope)


def set_linear(spec, variables, scope):
    spec.weight = _get_variable(variables, "%s.weight" % scope)
    spec.bias = _get_variable(variables, "%s.bias" % scope)


def set_embeddings(spec, variables, scope, multiply_by_sqrt_depth=True):
    spec.weight = _get_variable(variables, scope)
    spec.multiply_by_sqrt_depth = multiply_by_sqrt_depth


def set_position_encodings(spec, variables, scope):
    spec.encodings = _get_variable(variables, scope).squeeze()


def set_token_type_encodings(spec, variables, scope):
    spec.weight = _get_variable(variables, scope)


def _get_variable(variables, name):
    return variables[name].numpy()