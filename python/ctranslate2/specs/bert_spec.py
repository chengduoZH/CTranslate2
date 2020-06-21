"""Declares specification of the Bert model."""

import numpy as np

from ctranslate2.specs import common_spec
from ctranslate2.specs import model_spec
from ctranslate2.specs import transformer_spec


class TokenTypeEncoderSpec(model_spec.LayerSpec):
    def __init__(self):
        self.encodings = model_spec.OPTIONAL


class BertSelfOutputLayerSpec(model_spec.LayerSpec):
    def __init__(self):
        self.bert_self_output = common_spec.LinearSpec()
        self.layer_norm = common_spec.LinearSpec()


class BertSelfAttentionLayerSpec(model_spec.LayerSpec):
    def __init__(self):
        self.layer_norm = common_spec.LayerNormSpec()
        num_projections = 2
        self.linear = [common_spec.LinearSpec() for _ in range(num_projections)]


class BertAttentionLayerSpace(model_spec.LayerSpec):
    def __init__(self):
        self.bert_attention = BertSelfAttentionLayerSpec()


class BertOutput(model_spec.LayerSpec):
    def __init__(self):
        self.linear = common_spec.LinearSpec()
        self.layer_norm = common_spec.LayerNormSpec()


class BertIntermediate(model_spec.LayerSpec):
    def __init__(self):
        self.linear = common_spec.LinearSpec()


class BertEncoderLayerSpec(model_spec.LayerSpec):
    def __init__(self):
        self.self_attention = BertSelfAttentionLayerSpec()
        self.ffn = transformer_spec.FeedForwardSpec()


class BertEmbeddingsLayerSpace(model_spec.LayerSpec):
    def __init__(self):
        self.embeddings = common_spec.EmbeddingsSpec()
        self.position_encodings = transformer_spec.PositionEncoderSpec()
        self.token_type_encodings = TokenTypeEncoderSpec()
        self.layer_norm = common_spec.LayerNormSpec()


class BertEncoderSpec(model_spec.LayerSpec):
    def __init__(self, num_layers):
        self.bert_embedding_layer = BertEmbeddingsLayerSpace()
        self.layer = [BertEncoderLayerSpec() for _ in range(num_layers)]


class BertSpec(model_spec.ModelSpec):
    def __init__(self, num_layers, num_heads):
        self.num_heads = np.dtype("int8").type(num_heads)
        self.encoder = BertEncoderSpec(num_layers)

    @property
    def name(self):
        return "BertSpec"

    @property
    def revision(self):
        return 3

    @property
    def source_vocabulary_size(self):
        return self.encoder.embeddings.weight.shape[0]
