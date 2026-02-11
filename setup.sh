#!/bin/bash

# uv venv -p 3.12
# source .venv/bin/activate

# https://github.com/Dao-AILab/flash-attention?tab=readme-ov-file#installation-and-features
uv pip install packaging psutil ninja
uv pip install torch>=2.4.0
uv pip install triton>=3.0.0 transformers>=4.51.0 xxhash
MAX_JOBS=8 uv pip install flash-attn --no-build-isolation
