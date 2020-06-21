#pragma once

#include <string>
#include <vector>

#include "ctranslate2/ops/ops.h"
#include "ctranslate2/models/model.h"
#include "ctranslate2/layers/attention.h"
#include "ctranslate2/models/transformer.h"

namespace bert {

using namespace ctranslate2;

class Bert {
public:
  Bert(const std::string &model_dir,
       Device device = Device::CPU,
       int device_index = 0,
       ComputeType compute_type = ComputeType::DEFAULT);

  std::vector<std::vector<std::vector<float>>>  operator()(const std::vector<std::vector<size_t >> &input,
                                             const std::vector<std::vector<size_t>> &token_type_ids) const;

  void set_model(const std::shared_ptr<const ctranslate2::models::Model> &model);

private:
  std::shared_ptr<const ctranslate2::models::Model> _model;
  std::unique_ptr<layers::Encoder> _encoder;
};


}
