#pragma once

#include <string>
#include <vector>

#include "ctranslate2/ops/ops.h"
#include "ctranslate2/models/model.h"

namespace ctranslate2 {
class Embeddings {
public:
  Embeddings(const models::Model &model, const std::string &scope);

  void operator()(const StorageView &ids, StorageView &output) const;

private:
  const ops::Gather _gather_op;
  const StorageView &_embeddings;
  const StorageView *_qscale;
  const std::unique_ptr<const StorageView> _scale;
};

class LayerNorm {
public:
  LayerNorm(const models::Model &model, const std::string &scope);

  void operator()(const StorageView &input, StorageView &output) const;

private:
  const ops::LayerNorm norm_op_;
  const StorageView &beta_;
  const StorageView &gamma_;
};


class EncoderLayer {

};

class BartEncoder {

};

class SelfAttention {

};

class BartModel {

};

class BartClassificationHead {

};
}
