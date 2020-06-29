#!/usr/bin/env bash
/opt/miniconda3/bin/transformers-converter  --model ./bert_model/pytorch_model.bin \
                       --config ./bert_model/config.json   \
                       --output ./bert_model_int8 \
                       --quantization int8
