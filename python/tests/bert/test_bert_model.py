
import torch
from transformers.modeling_bert import BertModel, BertConfig
import numpy as np

cfg = BertConfig()
torch_model = BertModel.from_pretrained('./bert_model')
torch_model.eval()
a = np.array([[1, 2], [3, 4]], np.long)
input_ids = torch.from_numpy(a)
print(torch_model(input_ids)[0])
