#include "ctranslate2/decoding.h"

#include <cmath>
#include <limits>
#include <map>

#include "ctranslate2/ops/ops.h"
#include "./device_dispatch.h"

namespace ctranslate2 {

  static const ops::Gather gather;

  static void split_batch_beam(StorageView& input, dim_t beam_size) {
    Shape shape = input.shape();
    shape.insert(shape.begin() + 1, beam_size);
    shape[0] /= beam_size;
    input.reshape(shape);
  }

  static void merge_batch_beam(StorageView& input) {
    Shape shape = input.shape();
    shape[0] *= shape[1];
    shape.erase(shape.begin() + 1);
    input.reshape(shape);
  }

  static void gather_batch(StorageView& data, const StorageView& indices, dim_t beam_size) {
    split_batch_beam(data, beam_size);
    gather(data, indices);
    merge_batch_beam(data);
  }

  static void tile(StorageView& input, const StorageView& repeats) {
    static const ops::Tile tile_op{};
    tile_op(input, repeats);
  }

  static void expand_to_beam_size(StorageView& input, dim_t beam_size) {
    Shape original_shape(input.shape());
    Shape tile_shape(input.shape());
    tile_shape.insert(std::next(tile_shape.begin()), 1);
    input.reshape(tile_shape);
    StorageView repeats({input.rank()}, static_cast<int32_t>(1));
    repeats.at<int32_t>(1) = beam_size;
    tile(input, repeats);
    original_shape[0] *= beam_size;
    input.reshape(original_shape);
  }

  static void expand_to_beam_size(layers::DecoderState& state, dim_t beam_size) {
    for (auto& pair : state) {
      if (!pair.second.empty())
        expand_to_beam_size(pair.second, beam_size);
    }
  }

  static void penalize_token(StorageView& log_probs, dim_t token) {
    DEVICE_DISPATCH(log_probs.device(),
                    primitives<D>::strided_fill(log_probs.data<float>() + token,
                                                static_cast<float>(-1e10),
                                                log_probs.dim(-1),
                                                log_probs.dim(0)));
  }


  BeamSearch::BeamSearch(const dim_t beam_size, const float length_penalty, const float coverage_penalty)
    : _beam_size(beam_size)
    , _length_penalty(length_penalty)
    , _coverage_penalty(coverage_penalty) {
  }

  void
  BeamSearch::search(layers::Decoder& decoder,
                     layers::DecoderState& state,
                     const Sampler& sampler,
                     const std::vector<size_t>& start_ids,
                     const dim_t start_step,
                     const dim_t end_id,
                     const dim_t max_length,
                     const dim_t min_length,
                     const std::vector<size_t>* output_ids_map,
                     std::vector<std::vector<std::vector<size_t>>>& sampled_ids,
                     std::vector<std::vector<float>>* scores,
                     std::vector<std::vector<std::vector<std::vector<float>>>>* attention,
                     const size_t num_hypotheses) const {
    PROFILE("beam_search");
    const dim_t max_step = start_step + max_length;
    const Device device = decoder.device();
    const dim_t batch_size = start_ids.size();
    dim_t cur_batch_size = batch_size;
    const ops::TopK topk_op(_beam_size);
    StorageView alive_seq({batch_size, 1},
                          std::vector<int32_t>(start_ids.begin(), start_ids.end()));

    expand_to_beam_size(state, _beam_size);
    expand_to_beam_size(alive_seq, _beam_size);

    StorageView gather_indices(DataType::INT32);
    StorageView topk_ids(alive_seq);
    StorageView topk_scores;
    StorageView topk_log_probs({_beam_size}, std::numeric_limits<float>::lowest());
    topk_log_probs.at<float>(0) = 0;
    tile(topk_log_probs, StorageView({1}, static_cast<int32_t>(batch_size)));

    using Result = std::pair<std::vector<size_t>, std::vector<std::vector<float>>>;
    std::vector<std::map<float, Result>> hypotheses;
    hypotheses.resize(batch_size);
    sampled_ids.clear();
    sampled_ids.resize(batch_size);
    if (scores) {
      scores->clear();
      scores->resize(batch_size);
    }
    if (attention) {
      attention->clear();
      attention->resize(batch_size);
    }

    std::vector<bool> top_beam_finished(batch_size, false);
    std::vector<dim_t> batch_offset(batch_size);
    for (dim_t i = 0; i < batch_size; ++i) {
      batch_offset[i] = i;
      sampled_ids[i].reserve(num_hypotheses);
      if (scores)
        (*scores)[i].reserve(num_hypotheses);
      if (attention)
        (*attention)[i].reserve(num_hypotheses);
    }

    StorageView logits(device);
    StorageView log_probs(device);
    StorageView alive_attention;
    StorageView attention_step;
    StorageView attention_step_device(device);

    StorageView coverage;

    for (dim_t step = start_step; step < max_step; ++step) {
      // Compute log probs for the current step.
      decoder(step,
              topk_ids.to(device),
              state,
              &logits,
              (attention || _coverage_penalty != 0) ? &attention_step_device : nullptr);
      ops::LogSoftMax()(logits, log_probs);
      const dim_t vocabulary_size = log_probs.dim(-1);

      // Multiply by the current beam log probs.
      DEVICE_DISPATCH(log_probs.device(),
                      primitives<D>::add_depth_broadcast(topk_log_probs.to(device).data<float>(),
                                                         log_probs.data<float>(),
                                                         topk_log_probs.size(),
                                                         log_probs.size()));

      // Penalize by the length, if enabled.
      float length_penalty_weight = 1.0;
      if (_length_penalty != 0) {
        length_penalty_weight = std::pow((5.0 + static_cast<float>(step + 1)) / 6.0, _length_penalty);
        ops::Mul()(log_probs, StorageView(1.f / length_penalty_weight, log_probs.device()), log_probs);
      }

      // Penalize end_id, if configured.
      if (step < min_length)
        penalize_token(log_probs, end_id);

      // Flatten the probs into a list of candidates.
      log_probs.reshape({cur_batch_size, _beam_size * vocabulary_size});

      // TopK candidates.
      sampler(log_probs, topk_ids, topk_scores, _beam_size);
      if (attention || _coverage_penalty != 0)
        attention_step.copy_from(attention_step_device);

      topk_log_probs = topk_scores;
      // Recover the true log probs if length penalty was applied.
      if (_length_penalty != 0)
        ops::Mul()(topk_log_probs, StorageView(length_penalty_weight, topk_log_probs.device()), topk_log_probs);

      // Unflatten the ids.
      gather_indices.resize({cur_batch_size * _beam_size});
      for (dim_t i = 0; i < topk_ids.size(); ++i) {
        auto flat_id = topk_ids.at<int32_t>(i);
        auto beam_id = flat_id / vocabulary_size;
        auto word_id = flat_id % vocabulary_size;
        auto batch_id = i / _beam_size;
        if (output_ids_map)
          word_id = output_ids_map->at(word_id);
        topk_ids.at<int32_t>(i) = word_id;
        gather_indices.at<int32_t>(i) = beam_id + batch_id * _beam_size;
      }

      // Append last prediction.
      gather(alive_seq, gather_indices);
      alive_seq.reshape({cur_batch_size, _beam_size, alive_seq.dim(-1)});
      topk_ids.reshape({cur_batch_size, _beam_size, 1});
      StorageView cur_alive_seq(std::move(alive_seq));
      ops::Concat(-1)({&cur_alive_seq, &topk_ids}, alive_seq);
      topk_log_probs.reshape({cur_batch_size, _beam_size});
      topk_scores.reshape({cur_batch_size, _beam_size});
      topk_ids.reshape({cur_batch_size, _beam_size});

      if(_coverage_penalty != 0){
        if(step == 0){
          coverage = attention_step;
        }else{
          gather(coverage, gather_indices);
          ops::Add()(attention_step, coverage, coverage);
        }
        auto penalty = StorageView({cur_batch_size * _beam_size, 1}, coverage.dtype(), coverage.device());
        auto tmp = StorageView(coverage.shape(), coverage.dtype(), coverage.device());
        ops::Min()(coverage, 1.0f, tmp);
        ops::Log()(tmp, tmp);
        int row = std::accumulate(tmp.shape().begin(), tmp.shape().end()-1,	1,std::multiplies<int>());
        int col = tmp.shape().back();
        tmp.reshape({row, col});
        ops::MatMul()(tmp, StorageView({col, 1}, 1.0f, coverage.device()), penalty);
        ops::Mul()(penalty, StorageView(_coverage_penalty, penalty.device()), penalty);
        ops::Add()(penalty, topk_scores, topk_scores);
      }

      if (attention) {
        if (alive_attention.empty()){
          alive_attention = attention_step;
        }
        else {
          gather(alive_attention, gather_indices);
          StorageView cur_alive_attention(std::move(alive_attention));
          ops::Concat(1)({&cur_alive_attention, &attention_step}, alive_attention);
        }
        alive_attention.reshape({cur_batch_size,
                                 _beam_size,
                                 alive_attention.dim(1),
                                 alive_attention.dim(2)});
      }

      // Check if some hypotheses are finished.
      std::vector<bool> finished(cur_batch_size, false);
      dim_t finished_count = 0;
      for (dim_t i = 0; i < cur_batch_size; ++i) {
        const dim_t batch_id = batch_offset[i];
        for (dim_t k = 0; k < _beam_size; ++k) {
          if (topk_ids.at<int32_t>({i, k}) == static_cast<int32_t>(end_id)
              || step + 1 == max_step) {
            if (k == 0)
              top_beam_finished[i] = true;
            float score = topk_scores.at<float>({i, k});
            // Prevent this beam from advancing in the next step.
            topk_log_probs.at<float>({i, k}) = -1e10;
            // Save the finished hypothesis only if it is still a candidate.
            if (hypotheses[batch_id].size() < num_hypotheses
                || -score < hypotheses[batch_id].rbegin()->first) {
              std::vector<size_t> hypothesis;
              std::vector<std::vector<float>> attn;
              const dim_t max_time = alive_seq.dim(-1);
              hypothesis.reserve(max_time);
              if (attention)
                attn.reserve(max_time);
              for (dim_t t = 1; t < max_time; ++t) {
                const int32_t id = alive_seq.at<int32_t>({i, k, t});
                if (id == static_cast<int32_t>(end_id))
                  break;
                hypothesis.push_back(id);
                if (attention) {
                  const auto* attn_vec = alive_attention.index<float>({i, k, t - 1});
                  attn.emplace_back(attn_vec, attn_vec + alive_attention.dim(-1));
                }
              }

              // Use -score as the key to iterate the map from best to worst.
              hypotheses[batch_id].emplace(std::piecewise_construct,
                                           std::forward_as_tuple(-score),
                                           std::forward_as_tuple(std::move(hypothesis),
                                                                 std::move(attn)));
            }
          }
        }

        if (top_beam_finished[i] && hypotheses[batch_id].size() >= num_hypotheses) {
          ++finished_count;
          finished[i] = true;

          // Return the "num_hypotheses" best hypotheses.
          for (auto& pair : hypotheses[batch_id]) {
            if (sampled_ids[batch_id].size() >= num_hypotheses)
              break;
            sampled_ids[batch_id].emplace_back(std::move(pair.second.first));
            if (scores) {
              (*scores)[batch_id].push_back(-pair.first);
            }
            if (attention) {
              (*attention)[batch_id].emplace_back(std::move(pair.second.second));
            }
          }
          hypotheses[batch_id].clear();
        }
      }

      // If all remaining sentences are finished, no need to go further.
      if (finished_count == cur_batch_size)
        break;

      // If some sentences finished on this step, ignore them for the next step.
      if (finished_count > 0) {
        auto old_batch_size = cur_batch_size;
        cur_batch_size -= finished_count;
        StorageView keep_batches({cur_batch_size}, DataType::INT32);
        size_t write_index = 0;
        size_t read_index = 0;
        for (; read_index < finished.size(); ++read_index) {
          if (!finished[read_index]) {
            keep_batches.at<int32_t>(write_index) = read_index;
            top_beam_finished[write_index] = top_beam_finished[read_index];
            batch_offset[write_index] = batch_offset[read_index];
            ++write_index;
          }
        }
        gather(topk_ids, keep_batches);
        gather(topk_log_probs, keep_batches);
        gather(alive_seq, keep_batches);
       
        alive_seq.reshape({cur_batch_size * _beam_size, alive_seq.dim(-1)});
        if (attention)
          gather(alive_attention, keep_batches);

        // On CPU, we reorder first and then remove finished batches. Otherwise, we remove
        // finished batches from the reorder indices and then reorder. The motivation for this
        // difference is to enable the fast in place gather on CPU for state elements that should
        // not be reordered (see Decoder::gather_state and Gather::operator()).

        if (device == Device::CPU) {
          decoder.gather_state(state, gather_indices);
          for (auto& pair : state)
            gather_batch(pair.second, keep_batches, _beam_size);
        } else {
          gather_batch(gather_indices, keep_batches, _beam_size);
          decoder.gather_state(state, gather_indices.to(device));
        }

        if(_coverage_penalty != 0){
          coverage.reshape({old_batch_size, _beam_size, coverage.dim(1), coverage.dim(2)});
          gather(coverage, keep_batches);
          coverage.reshape({cur_batch_size * _beam_size, coverage.dim(2), coverage.dim(3)});
        }

      } else {
        decoder.gather_state(state, gather_indices.to(device));
      }

      topk_ids.reshape({cur_batch_size * _beam_size, 1});
      topk_log_probs.reshape({cur_batch_size * _beam_size});
      alive_seq.reshape({cur_batch_size * _beam_size, alive_seq.dim(-1)});
      if (attention)
        alive_attention.reshape({cur_batch_size * _beam_size,
                                 alive_attention.dim(2),
                                 alive_attention.dim(3)});
    }
  }

  void
  GreedySearch::search(layers::Decoder& decoder,
                       layers::DecoderState& state,
                       const Sampler& sampler,
                       const std::vector<size_t>& start_ids,
                       const dim_t start_step,
                       const dim_t end_id,
                       const dim_t max_length,
                       const dim_t min_length,
                       const std::vector<size_t>* output_ids_map,
                       std::vector<std::vector<std::vector<size_t>>>& sampled_ids,
                       std::vector<std::vector<float>>* scores,
                       std::vector<std::vector<std::vector<std::vector<float>>>>* attention,
                       const size_t) const {
    PROFILE("greedy_search");
    const dim_t max_step = start_step + max_length;
    const Device device = decoder.device();
    const dim_t batch_size = start_ids.size();
    StorageView sample_from({batch_size, 1},
                            std::vector<int32_t>(start_ids.begin(), start_ids.end()));

    sampled_ids.clear();
    sampled_ids.resize(batch_size);
    if (scores) {
      scores->clear();
      scores->resize(batch_size);
    }
    if (attention) {
      attention->clear();
      attention->resize(batch_size);
    }

    StorageView logits(device);
    StorageView log_probs(device);
    StorageView alive({batch_size}, DataType::INT32);
    std::vector<bool> finished(batch_size, false);
    std::vector<dim_t> batch_offset(batch_size);
    for (dim_t i = 0; i < batch_size; ++i) {
      batch_offset[i] = i;
      sampled_ids[i].resize(1);
      if (scores)
        (*scores)[i].resize(1);
      if (attention)
        (*attention)[i].resize(1);
    }

    StorageView best_ids( DataType::INT32);
    StorageView best_probs;
    StorageView attention_step;
    StorageView attention_step_device(device);

    for (dim_t step = start_step; step < max_step; ++step) {
      decoder(step,
              sample_from.to(device),
              state,
              &logits,
              attention ? &attention_step_device : nullptr);

      // Compute log probs only if scores should be returned.
      if (scores) {
        ops::LogSoftMax()(logits, log_probs);
      } else {
        log_probs.shallow_copy(logits);
      }

      // Penalize end_id, if configured.
      if (step < min_length)
        penalize_token(log_probs, end_id);

      sampler(log_probs, best_ids, best_probs);
      if (attention)
        attention_step.copy_from(attention_step_device);

      std::vector<bool> finished_batch(log_probs.dim(0), false);
      bool one_finished = false;
      dim_t count_alive = 0;
      for (dim_t i = 0; i < log_probs.dim(0); ++i) {
        int32_t true_id = best_ids.scalar_at<int32_t>({i});
        if (output_ids_map)
          true_id = output_ids_map->at(true_id);
        dim_t batch_id = batch_offset[i];
        if (true_id == static_cast<int32_t>(end_id)) {
          finished[batch_id] = true;
          finished_batch[i] = true;
          one_finished = true;
        } else {
          sample_from.at<int32_t>(i) = true_id;
          sampled_ids[batch_id][0].push_back(true_id);
          ++count_alive;
          if (scores) {
            (*scores)[batch_id][0] += best_probs.scalar_at<float>({i});
          }
          if (attention) {
            const auto* attn = attention_step.index<float>({i});
            (*attention)[batch_id][0].emplace_back(attn, attn + attention_step.dim(-1));
          }
        }
      }

      // No more sentences are alive, stop here.
      if (count_alive == 0)
        break;

      // Remove finished sentences from the execution.
      if (one_finished) {
        alive.resize({count_alive});
        size_t write_index = 0;
        size_t read_index = 0;
        for (; read_index < finished_batch.size(); ++read_index) {
          if (!finished_batch[read_index]) {
            batch_offset[write_index] = batch_offset[read_index];
            alive.at<int32_t>(write_index) = read_index;
            ++write_index;
          }
        }
        gather(sample_from, alive);
        auto alive_device = alive.to(device);
        decoder.gather_state(state, alive_device);
      }
    }
  }

  void initialize_decoder_with_prefix(layers::Decoder& decoder,
                                      layers::DecoderState& state,
                                      const std::vector<size_t>& start_ids,
                                      const std::vector<size_t>& prefix_ids,
                                      std::vector<std::vector<float>>* prefix_attention) {
    const Device device = decoder.device();
    const size_t prefix_size = prefix_ids.size();

    StorageView input({1, 1}, std::vector<int32_t>(start_ids.begin(), start_ids.end()));
    StorageView attention(device);
    if (prefix_attention)
      prefix_attention->reserve(prefix_size);

    for (size_t i = 0; i < prefix_size; ++i) {
      decoder(i,
              input.to(device),
              state,
              /*logits=*/nullptr,
              prefix_attention ? &attention : nullptr);
      if (prefix_attention)
        prefix_attention->emplace_back(attention.to_vector<float>());
      input.at<int32_t>(0) = prefix_ids[i];
    }
  }

  template <typename T>
  static std::vector<std::vector<T>> batch_to_hypotheses(std::vector<std::vector<T>>& array) {
    if (array.empty())
      return array;
    std::vector<std::vector<T>> new_array;
    new_array.emplace_back();
    new_array.front().reserve(array.size());
    for (auto& vector : array) {
      new_array.front().emplace_back(std::move(vector[0]));
    }
    return new_array;
  }

  std::vector<GenerationResult<size_t>>
  decode(layers::Decoder& decoder,
         layers::DecoderState& state,
         const SearchStrategy& search_strategy,
         const Sampler& sampler,
         std::vector<size_t> start_ids,
         const std::vector<std::vector<size_t>>* prefix_ids,
         const std::vector<size_t>* output_ids_map,
         const dim_t end_id,
         const dim_t max_length,
         const dim_t min_length,
         const size_t num_hypotheses,
         const bool return_alternatives,
         const bool return_scores,
         const bool return_attention) {
    dim_t start_step = 0;

    // Forward target prefix, if set (only batch_size = 1 for now).
    std::vector<std::vector<std::vector<float>>> prefix_attention;
    if (prefix_ids) {
      if (prefix_ids->size() > 1)
        throw std::invalid_argument("Batch decoding with a prefix is not supported");
      if (return_attention)
        prefix_attention.resize(1);
      initialize_decoder_with_prefix(decoder,
                                     state,
                                     start_ids,
                                     prefix_ids->front(),
                                     return_attention ? &prefix_attention[0] : nullptr);
      start_ids[0] = prefix_ids->front().back();
      start_step += prefix_ids->front().size();
    }

    std::vector<std::vector<std::vector<size_t>>> expanded_ids;
    std::vector<std::vector<float>> expanded_scores;
    std::vector<std::vector<std::vector<std::vector<float>>>> expanded_attention;
    if (return_alternatives) {
      // In this translation mode, we first expand the next "num_hypotheses" candidate words
      // before running the full decoding on each prefix. This is to ensure that we get unique
      // alternatives at this decoding position.
      BeamSearch(num_hypotheses).search(decoder,
                                        state,
                                        BestSampler(),
                                        start_ids,
                                        start_step,
                                        end_id,
                                        /*max_length=*/1,
                                        /*min_length=*/1,
                                        output_ids_map,
                                        expanded_ids,
                                        return_scores ? &expanded_scores : nullptr,
                                        return_attention ? &expanded_attention : nullptr,
                                        num_hypotheses);

      // The next input is the words we just expanded.
      start_ids.resize(num_hypotheses);
      for (size_t i = 0; i < num_hypotheses; ++i)
        start_ids[i] = expanded_ids[0][i].back();
      start_step += 1;
    }

    std::vector<std::vector<std::vector<size_t>>> sampled_ids;
    std::vector<std::vector<float>> scores;
    std::vector<std::vector<std::vector<std::vector<float>>>> attention;
    search_strategy.search(decoder,
                           state,
                           sampler,
                           start_ids,
                           start_step,
                           end_id,
                           max_length,
                           min_length,
                           output_ids_map,
                           sampled_ids,
                           return_scores ? &scores : nullptr,
                           return_attention ? &attention : nullptr,
                           return_alternatives ? 1 : num_hypotheses);

    if (return_alternatives) {
      // We convert outputs from shape num_hypotheses x 1 to 1 x num_hypotheses.
      sampled_ids = batch_to_hypotheses(sampled_ids);
      scores = batch_to_hypotheses(scores);
      attention = batch_to_hypotheses(attention);
    }

    // Build results.
    const size_t batch_size = sampled_ids.size();
    std::vector<GenerationResult<size_t>> results;
    results.reserve(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {

      // Aggregate result from the optional prefix and expansion step.
      if (prefix_ids || return_alternatives) {

        for (size_t h = 0; h < num_hypotheses; ++h) {
          // Finalize the generated ids.
          std::vector<size_t>& ids = sampled_ids[i][h];
          if (!expanded_ids.empty())
            ids.insert(ids.begin(), expanded_ids[i][h][0]);
          if (prefix_ids)
            ids.insert(ids.begin(), prefix_ids->at(i).begin(), prefix_ids->at(i).end());

          // Finalize the score.
          if (return_scores && !expanded_scores.empty())
            scores[i][h] += expanded_scores[i][h];

          // Finalize the attention.
          if (return_attention) {
            std::vector<std::vector<float>>& attn = attention[i][h];
            if (!expanded_attention.empty())
              attn.insert(attn.begin(), expanded_attention[i][h][0]);
            if (!prefix_attention.empty())
              attn.insert(attn.begin(),
                          prefix_attention[i].begin(),
                          prefix_attention[i].end());
          }
        }
      }

      GenerationResult<size_t> result(std::move(sampled_ids[i]));
      if (return_scores)
        result.set_scores(std::move(scores[i]));
      if (return_attention)
        result.set_attention(std::move(attention[i]));
      results.emplace_back(std::move(result));
    }

    return results;
  }

}
