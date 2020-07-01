import torch
from transformers.modeling_bert import BertModel, BertConfig
import numpy as np
import ctranslate2
import time


def get_run_time(func, iterations=5):
    begin = time.time()
    for _ in range(iterations):
        func()
    return (time.time() - begin) / iterations


def get_random_data(range=1000, shape=(1, 64)):
    return np.random.randint(range, size=shape)


if __name__ == "__main__":
    # load ber model
    cfg = BertConfig()
    torch_model = BertModel.from_pretrained('./bert_model')
    torch_model.eval()

    input = [[1, 2], [3, 4]]
    input_ids = torch.from_numpy(np.array(input, np.long))
    get_run_time(lambda: torch_model(input_ids))
    print(torch_model(input_ids)[0])

    ct2_bert_model = ctranslate2.translator.Bert("./bert_model_fp32/") #, intra_threads=1)
    get_run_time(lambda: ct2_bert_model(input))
    print(np.array(ct2_bert_model(input)))

    ct2_bert_int8_model = ctranslate2.translator.Bert("./bert_model_int8/") #, intra_threads=1)
    get_run_time(lambda: ct2_bert_int8_model(input))
    print(np.array(ct2_bert_int8_model(input)))

    def _run_torch_bert():
        input_ids = torch.from_numpy(get_random_data())
        torch_model(input_ids)
    with torch.autograd.profiler.profile() as prof:
        print(
            f"torch_bert time consume: {get_run_time(_run_torch_bert, iterations=100)}")
    print(prof.key_averages().table(sort_by="self_cpu_time_total"))

    ctranslate2.translator.init_profiling("cpu", 1)
    def _run_ct2_bert():
        ct2_bert_model(get_random_data().tolist())
    print(f"ct2_bert time consume: {get_run_time(_run_ct2_bert, iterations=100)}")

    ctranslate2.translator.dump_profiling("./profile.prof")
    #
    # def _run_ct2_int8_bert():
    #     ct2_bert_int8_model(get_random_data().tolist())
    # print(f"ct2_int8_bert time consume: {get_run_time(_run_ct2_int8_bert, iterations=100)}")
