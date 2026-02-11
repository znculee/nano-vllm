#!/bin/bash

HF_HUB_ENABLE_HF_TRANSFER=1 \
hf download Qwen/Qwen3-0.6B \
  --local-dir ./huggingface/Qwen3-0.6B
