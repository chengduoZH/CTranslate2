#include "bert/model/bert.h"

#include <cmath>

#include "../device_dispatch.h"

namespace bert {
namespace models {

static bool replace(std::string &str, const std::string &from, const std::string &to) {
  size_t start_pos = str.find(from);
  if (start_pos == std::string::npos)
    return false;
  str.replace(start_pos, from.length(), to);
  return true;
}

static std::string map_v1_variable_name(std::string name) {
  // V1 variable names were simply the names defined by OpenNMT-tf.
  replace(name, "transformer/", "");
  replace(name, ":0", "");
  replace(name, "w_embs", "embeddings/weight");
  replace(name, "kernel", "weight");
  replace(name, "LayerNorm", "layer_norm");
  replace(name, "dense", "projection");
  replace(name, "conv1d_", "linear_");
  replace(name, "conv1d", "linear_0");
  if (name.find("encoder") != std::string::npos) {
    replace(name, "multi_head", "self_attention");
  } else {
    replace(name, "masked_multi_head", "self_attention");
    replace(name, "multi_head", "attention");
  }
  return name;
}

BertModel::BertModel(const std::string &path,
                     size_t spec_revision,
                     size_t num_heads)
        : ctranslate2::models::Model(path, spec_revision), _num_heads(num_heads) {
}

size_t BertModel::num_heads() const {
  return _num_heads;
}


size_t BertModel::current_spec_revision() const {
  return 3;
}

bool BertModel::is_quantizable(const std::string &variable_name) const {
  return ends_with(variable_name, "weight");
}

bool BertModel::is_linear_weight(const std::string &variable_name) const {
  // Linear weights are all variables that are quantizable and not under the "embeddings" scope.
  return is_quantizable(variable_name) && !(variable_name.find("embeddings") != std::string::npos || variable_name.find("token_type_encodings") != std::string::npos);
}

bool BertModel::is_packable(const std::string &variable_name) const {
  // Disallow packing for the last linear layer which can be dynamically masked.
  return (is_linear_weight(variable_name)
          && variable_name.find("projection") == std::string::npos);
}

void BertModel::register_variable(const std::string &name, StorageView &variable) {
  std::string var_name = name;
  if (_spec_revision == 1)
    var_name = map_v1_variable_name(name);
  Model::register_variable(var_name, variable);
}

void BertModel::finalize() {
  Model::finalize();
  if (_spec_revision >= 3)
    _num_heads = get_variable("num_heads").as_scalar<int8_t>();
}

std::unique_ptr<layers::Encoder> BertModel::make_encoder() const {
  return std::unique_ptr<layers::Encoder>(new BertEncoder(*this, "encoder"));
}

PositionEncoder::PositionEncoder(const BertModel &model, const std::string &scope)
        : ctranslate2::models::PositionEncoder(){
  _model_encoding = model.get_variable_if_exists(scope + "/encodings");
}

TokenTypeEncoder::TokenTypeEncoder(const BertModel &model, const std::string &scope)
        : _embeddings(model.get_variable(scope + "/weight")),
          _qscale(model.get_variable_if_exists(scope + "/weight_scale")) {
}

void TokenTypeEncoder::operator()(const StorageView &ids, StorageView &output) const {
  PROFILE("Embeddings");
  if (_embeddings.dtype() == DataType::INT16 || _embeddings.dtype() == DataType::INT8) {
    const auto device = ids.device();
    StorageView gathered(_embeddings.dtype(), device);
    _gather_op(_embeddings, ids, gathered);
    if (_qscale->is_scalar())
      ops::Dequantize()(gathered, *_qscale, output);
    else {
      StorageView scale(_qscale->dtype(), device);
      _gather_op(*_qscale, ids, scale);
      ops::Dequantize()(gathered, scale, output);
    }
  } else {
    _gather_op(_embeddings, ids, output);
  }
}

BertFeedForward::BertFeedForward(const BertModel &model,
                                 const std::string &scope)
        : _layer_norm(model, scope + "/layer_norm"), _ff1(model, scope + "/linear_0"),
          _ff2(model, scope + "/linear_1") {
}

void BertFeedForward::operator()(const StorageView &input, StorageView &output) const {
  StorageView inner(input.device());
  _ff1(input, inner);
  ops::GELU()(inner, inner);
  _ff2(inner, output);
  ops::Add()(input, output, output);
  _layer_norm(output, output);
}


BertEncoderLayer::BertEncoderLayer(const BertModel &model,
                                   const std::string &scope)
        : _self_attention(model,
                          scope + "/self_attention",
                          model.num_heads(),
        /*self_attention=*/true), _ff(model, scope + "/ffn") {
}

void BertEncoderLayer::operator()(const StorageView &input,
                                  const StorageView &lengths,
                                  StorageView &output) const {
  PROFILE("BertEncoderLayer");
  StorageView context(input.device());
  _self_attention(input, nullptr, &lengths, context);
  _ff(context, output);
}


BertEncoder::BertEncoder(const BertModel &model, const std::string &scope)
        : _embeddings(model, scope + "/bert_embedding_layer/embeddings"),
          _position_encoder(new PositionEncoder(model, scope +
                                                       "/bert_embedding_layer/position_encodings")),
          _token_type_encoder(new TokenTypeEncoder(model, scope +
                                                          "/bert_embedding_layer/token_type_encodings")),
          _output_norm(model, scope + "/bert_embedding_layer/layer_norm") {
  for (size_t l = 0;; ++l) {
    try {
      _layers.emplace_back(new BertEncoderLayer(model,
                                                scope + "/layer_" + std::to_string(l)));
    } catch (std::exception &) {
      if (l == 0)
        throw;
      else
        break;
    }
  }
}

void BertEncoder::operator()(const StorageView &ids,
                             const StorageView &lengths,
                             StorageView &output) {
  PROFILE("BertEncoder");
  StorageView input(output.device());
  StorageView token_type_embedding(output.device());
  // split word ids and token ids.

  StorageView word_ids(DataType::INT32);
  StorageView token_ids(DataType::INT32);

  ops::Split split(0, true);
  split(ids, word_ids, token_ids);

  _embeddings(word_ids, input);
  (*_position_encoder)(input);
  (*_token_type_encoder)(token_ids, token_type_embedding);
  ctranslate2::ops::Add add;
  add(input, token_type_embedding, input);

  _output_norm(input, input);

  for (size_t l = 0; l < _layers.size(); ++l) {
    (*_layers[l])(input, lengths, output);
    if (l + 1 < _layers.size())
      swap(input, output);
  }
}

}
}
