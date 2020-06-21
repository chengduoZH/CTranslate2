#include "bert/model/bert.h"
#include "bert/bert_model.h"
#include "test_utils.h"

extern std::string g_data_dir;

TEST(BertTest,  BertModel) {
  bert::Bert bert("../../python/tests/bert_model_int8/");
  auto result = bert({{1, 2},
                      {3, 4}}, {});
  std::cout << result[0].size() << std::endl;
  for (auto &vec: result) {
    for (auto &items : vec) {
      for (auto &item : items) {
        std::cout << item << " ";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}
