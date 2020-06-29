import torch
from transformers.modeling_bert import BertModel, BertConfig
import numpy as np
import ctranslate2
import time

cfg = BertConfig()
torch_model = BertModel.from_pretrained('./bert_model')
torch_model.eval()

input = [[1, 2], [3, 4]]
input_ids = torch.from_numpy(np.array(input, np.long))
torch_model(input_ids)[0]
torch_model(input_ids)[0]
torch_model(input_ids)[0]
print(torch_model(input_ids)[0])
#
# ct2_bert_model = ctranslate2.translator.Bert("./bert_model_fp32/")
# ct2_bert_model.run_batch(input)
# ct2_bert_model.run_batch(input)
# ct2_bert_model.run_batch(input)
# print(np.array(ct2_bert_model.run_batch(input)))

ct2_bert_int8_model = ctranslate2.translator.Bert("./bert_model_int8/")
ct2_bert_int8_model.run_batch(input)
ct2_bert_int8_model.run_batch(input)
ct2_bert_int8_model.run_batch(input)
print(np.array(ct2_bert_int8_model.run_batch(input)))

iterations = 100
start = time.time()
for _ in range(iterations):
    input = np.random.randint(1000, size=(1, 64))
    input_ids = torch.from_numpy(input)
    torch_model(input_ids)
end = time.time()
print(f"time consume: {(end - start) / iterations}")

# start = time.time()
# for _ in range(iterations):
#     input = np.random.randint(1000, ct2_bert_modelsize=(1, 64)).tolist()
#     ct2_bert_model.run_batch(input)
# end = time.time()
# print(f"time consume: {(end - start) / iterations}")

start = time.time()
for _ in range(iterations):
    input = np.random.randint(1000, size=(1, 64)).tolist()
    ct2_bert_int8_model.run_batch(input)
end = time.time()
print(f"time consume: {(end - start) / iterations}")