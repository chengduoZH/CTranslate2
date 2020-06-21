#pragma once

// This file defines the execution engine for a BertSpec model.

#include "ctranslate2/models/sequence_to_sequence.h"
#include "ctranslate2/layers/encoder.h"
#include "ctranslate2/layers/layers.h"
#include "ctranslate2/models/transformer.h"

#include "ctranslate2/models/model.h"

namespace bert {
namespace models {
using namespace ctranslate2;


class BertModel : public ctranslate2::models::Model {
public:
  BertModel(const std::string &path, size_t spec_revision, size_t num_heads = 0);

  size_t num_heads() const;

  size_t current_spec_revision() const override;

  std::unique_ptr<layers::Encoder> make_encoder() const ;

protected:
  bool is_quantizable(const std::string &variable_name) const override;

  bool is_linear_weight(const std::string &variable_name) const override;

  bool is_packable(const std::string &variable_name) const override;

  void register_variable(const std::string &name, StorageView &variable) override;

  void finalize() override;

  size_t _num_heads;
};

class PositionEncoder: public ctranslate2::models::PositionEncoder {
public:
  PositionEncoder(const BertModel &model, const std::string &scope);
};

using namespace layers;
class TokenTypeEncoder{
public:
  TokenTypeEncoder(const BertModel &model, const std::string &scope);

  void operator()(const StorageView& ids, StorageView &input) const;

private:
  const ops::Gather _gather_op;
  const StorageView& _embeddings;
  const StorageView* _qscale;
};

class BertFeedForward {
public:
  BertFeedForward(const BertModel &model, const std::string &scope);

  void operator()(const StorageView &input, StorageView &output) const;

private:
  const layers::LayerNorm _layer_norm;
  const layers::Dense _ff1;
  const layers::Dense _ff2;
};

class BertEncoderLayer {
public:
  BertEncoderLayer(const BertModel &model, const std::string &scope);

  void operator()(const StorageView &input,
                  const StorageView &lengths,
                  StorageView &output) const;

private:
  const layers::BertMultiHeadAttention _self_attention;
  const BertFeedForward _ff;
};

class BertEncoder : public layers::Encoder {
public:
  BertEncoder(const BertModel &model, const std::string &scope);

  // ids contains word ids and token type ids.
  void operator()(const StorageView &ids,
                  const StorageView &lengths,
                  StorageView &output) override;

private:
  const layers::Embeddings _embeddings;
  const std::unique_ptr<PositionEncoder> _position_encoder;
  const std::unique_ptr<TokenTypeEncoder> _token_type_encoder;
  const layers::LayerNorm _output_norm;
  std::vector<std::unique_ptr<const BertEncoderLayer>> _layers;
};

}
}
