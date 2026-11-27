#!/usr/bin/env bash
source "/workspace/SemanticAnon/.venv/bin/activate"
export LD_LIBRARY_PATH="/workspace/SemanticAnon/.venv/lib/python3.11/site-packages/nvidia/cublas/lib:/workspace/SemanticAnon/.venv/lib/python3.11/site-packages/nvidia/cudnn/lib:/workspace/SemanticAnon/.venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PATH="/root/.local/bin:$PATH"