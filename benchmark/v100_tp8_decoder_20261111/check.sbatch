#!/bin/bash
#SBATCH --job-name=v100-tp8-decoder
#SBATCH --partition=turso
#SBATCH --qos=wide
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=80
#SBATCH --gres=gpu:V100:8
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --no-requeue
#SBATCH --time=01:00:00
set -euo pipefail
cd "$HOME/.scratch/sglang-v100-tp8-decoder-20261111"
old="$HOME/.scratch/sglang-v100-tp-20261111"
export TMPDIR="$PWD/tmp"
mkdir -p "$TMPDIR"
export APPTAINER_BIND="$PWD:$PWD,$old:$old,$old/cache:/root/sglang-v100-jit"
for rel in python/sglang/jit_kernel/sm70_nvfp4_moe_decode.py python/sglang/jit_kernel/csrc/sm70_nvfp4_moe_decode.cu python/sglang/srt/layers/moe/moe_runner/marlin.py; do
 export APPTAINER_BIND="$APPTAINER_BIND,$PWD/$rel:/opt/sglang/$rel:ro"
done
for rel in python/sglang/srt/layers/quantization/marlin_utils.py python/sglang/srt/layers/quantization/modelopt_quant.py; do
 export APPTAINER_BIND="$APPTAINER_BIND,$old/$rel:/opt/sglang/$rel:ro"
done
export APPTAINER_BIND="$APPTAINER_BIND,$old/image-qsa-tp.py:/opt/sglang/python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py:ro"
export APPTAINERENV_TMPDIR="$TMPDIR"
export APPTAINERENV_PYTHONPATH="$PWD/test-deps:/opt/sglang/python"
export APPTAINERENV_TORCH_CUDA_ARCH_LIST=7.0
export APPTAINERENV_SGLANG_V100_NVFP4_MOE_BUILD_DIR="$PWD/cache/nvfp4_moe"
export APPTAINERENV_TORCH_EXTENSIONS_DIR="$old/cache/torch-extensions"
export APPTAINERENV_TRITON_CACHE_DIR="$old/cache/triton"
export APPTAINERENV_FLASHINFER_WORKSPACE_BASE="$old/cache/flashinfer"
export APPTAINERENV_FLASHINFER_DISABLE_VERSION_CHECK=1
export APPTAINERENV_SGLANG_SM70_BATCHED_MOE=0
hostname
nvidia-smi
sha256sum "$HOME/llmturso-work/sifs/sglang-v100.sif" > sif.sha256
apptainer exec --nv --cleanenv "$HOME/llmturso-work/sifs/sglang-v100.sif" /opt/venv/bin/python -u compare_original.py
apptainer exec --nv --cleanenv "$HOME/llmturso-work/sifs/sglang-v100.sif" /opt/venv/bin/python -m pytest -q test/manual/kernels/moe/test_sm70_nvfp4_moe_decode.py
apptainer exec --nv --cleanenv "$HOME/llmturso-work/sifs/sglang-v100.sif" /opt/venv/bin/python -u test/manual/kernels/moe/benchmark_sm70_nvfp4_decode.py
python3 serve.py
