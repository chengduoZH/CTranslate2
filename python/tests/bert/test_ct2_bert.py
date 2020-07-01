import torch
from transformers.modeling_bert import BertModel, BertConfig
import numpy as np
import ctranslate2
import time

with_profile = False


def get_run_time(func, iterations=5):
    begin = time.time()
    for _ in range(iterations):
        func()
    return (time.time() - begin) / iterations


def get_random_data(range=1000, shape=(1, 128)):
    return np.random.randint(range, size=shape)


def test_torch_bert(with_quantize=False):
    input = [[1, 2], [3, 4]]
    input_ids = torch.from_numpy(np.array(input, np.long))
    torch_model = BertModel.from_pretrained('./bert_model')
    torch_model.eval()
    info = "not with quantize."
    if with_quantize:
        torch.quantization.quantize_dynamic(torch_model, inplace=True)
        info = "with quantize."

    get_run_time(lambda: torch_model(input_ids))

    # print(torch_model(input_ids)[0])

    def run_torch_bert():
        torch_model(torch.from_numpy(get_random_data()))

    if with_profile:
        with torch.autograd.profiler.profile() as prof:
            print(
                f"torch_bert({info}) time consume: {get_run_time(run_torch_bert, iterations=100)}")
        print(prof.key_averages().table(sort_by="self_cpu_time_total"))
    else:
        print(
            f"torch_bert({info}) time consume: {get_run_time(run_torch_bert, iterations=100)}")


def test_ct2_bert(model_path):
    ct2_bert_model = ctranslate2.translator.Bert(
        model_path)
    input = [[1, 2], [3, 4]]
    get_run_time(lambda: ct2_bert_model(input))
    # print(np.array(ct2_bert_model(input)))

    def run_ct2_bert():
        ct2_bert_model(get_random_data().tolist())

    if with_profile:
        ctranslate2.translator.init_profiling("cpu", 1)
    print(
        f"ct2_bert({model_path}) time consume: {get_run_time(run_ct2_bert, iterations=100)}")
    if with_profile:
        ctranslate2.translator.dump_profiling("./profile.prof")


if __name__ == "__main__":
    test_torch_bert()
    test_torch_bert(with_quantize=True)
    test_ct2_bert("./bert_model_fp32/")
    test_ct2_bert("./bert_model_int8/")
