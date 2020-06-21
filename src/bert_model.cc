#include "bert/bert_model.h"
#include "bert/model/bert.h"
#include <algorithm>
#include <numeric>


namespace bert {


Bert::Bert(const std::string &model_dir,
           Device device,
           int device_index,
           ComputeType compute_type) {
  set_model(ctranslate2::models::Model::load(model_dir, device, device_index, compute_type));
}

void Bert::set_model(const std::shared_ptr<const ctranslate2::models::Model> &model) {
  const auto *bert_model = dynamic_cast<const models::BertModel *>(model.get());
  if (!model)
    throw std::invalid_argument("Translator expects a model of type SequenceToSequenceModel");
  _model = model;
  _encoder = bert_model->make_encoder();
}

static std::pair<StorageView, StorageView>
make_inputs(const std::vector<std::vector<size_t>> &ids, const std::vector<std::vector<size_t>> &token_ids,
            Device device) {
  const dim_t batch_size = ids.size();

  // Record lengths and maximum length.
  dim_t max_length = 0;
  StorageView lengths({batch_size}, DataType::INT32);
  for (dim_t i = 0; i < batch_size; ++i) {
    const dim_t length = ids[i].size();
    lengths.at<int32_t>(i) = length;
    max_length = std::max(max_length, length);
  }

  // Make 2D input.
  StorageView input({batch_size * 2, max_length}, int32_t(0));
  for (dim_t i = 0; i < batch_size; ++i) {
    const dim_t length = ids[i].size();
    for (dim_t t = 0; t < length; ++t)
      input.at<int32_t>({i, t}) = ids[i][t];
  }

  for (dim_t i = 0; i < batch_size; ++i) {
    const dim_t length = token_ids[i].size();
    for (dim_t t = 0; t < length; ++t)
      input.at<int32_t>({i + batch_size, t}) = token_ids[i][t];
  }

  return std::make_pair(input.to(device), lengths.to(device));
}

std::vector<std::vector<std::vector<float>>> Bert::operator()(const std::vector<std::vector<size_t>> &source_ids,
                                                              const std::vector<std::vector<size_t >> &token_type_ids) const {
  // TODO(zcd): alignment source ids.
  std::vector<std::vector<size_t>> token_type_ids_tmp;
  if (token_type_ids.empty()) {
    token_type_ids_tmp.reserve(source_ids.size());
    for (auto &ids : source_ids) {
      std::vector<size_t> item(ids.size(), 0);
      token_type_ids_tmp.emplace_back(item);
    }
  }
  const Device device = _model->device();
  std::pair<StorageView, StorageView> inputs = make_inputs(source_ids, token_type_ids_tmp, device);
  StorageView &ids = inputs.first;
  StorageView &lengths = inputs.second;

  // Encode sequence.
  StorageView encoded(device);
  (*_encoder)(ids, lengths, encoded);
//  PrintStorage<float>(encoded, "encoded");

  auto vec = encoded.to_vector<float>();

  int64_t feat_len = encoded.dim(-1);
  std::vector<std::vector<std::vector<float>>> result;
  result.reserve(encoded.dim(0));
  for (int64_t i = 0; i < encoded.dim(0); ++i) {
    result.emplace_back();
    auto &sub_result = result.back();
    sub_result.reserve(encoded.dim(1));
    for (int64_t j = 0; j < encoded.dim(1); ++j) {
      sub_result.emplace_back();
      auto &sub_vec = sub_result.back();
      sub_vec.reserve(feat_len);
      std::copy(vec.begin() + (i * encoded.dim(1) + j) * feat_len,
                vec.begin() + (i * encoded.dim(1) + j) * feat_len + feat_len,
                std::back_inserter(sub_vec));
    }
  }
  return result;
}


}
