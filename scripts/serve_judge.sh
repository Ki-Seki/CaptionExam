#!/usr/bin/env bash
# Serve the default judge, Phi-4, on an OpenAI-compatible endpoint with vLLM.
#
#   GPU=0 PORT=8001 ./scripts/serve_judge.sh
#   captionexam judge --judge-base-url http://localhost:8001/v1
set -euo pipefail

MODEL="${MODEL:-microsoft/phi-4}"
PORT="${PORT:-8001}"
GPU="${GPU:-0}"
# The longest judge prompts we have seen are about 3k tokens; 16k leaves room for long captions.
MAX_LEN="${MAX_LEN:-16384}"

# The FlashInfer sampler has failed under guided (JSON schema) decoding; the judge relies on it.
export VLLM_USE_FLASHINFER_SAMPLER=0
export CUDA_VISIBLE_DEVICES="$GPU"
exec vllm serve "$MODEL" --port "$PORT" --max-model-len "$MAX_LEN" --gpu-memory-utilization 0.85
