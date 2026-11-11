# TP4/TP8 weight preparation, QSA routing, and small-batch MoE decode on v5.1.
# Keep v5.1's image-only FP16 KV and MTP ring fixes when updating QSA.
FROM geesegeesegeese/sglang-v100@sha256:80dccc86c3383cc81b0188e67cdd74961e9c4038a3986e1d0fd87e73733261e7

COPY python/sglang/srt/layers/quantization/marlin_utils.py \
     /opt/sglang/python/sglang/srt/layers/quantization/marlin_utils.py
COPY python/sglang/srt/layers/quantization/modelopt_quant.py \
     /opt/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py
COPY patches/v100-qwen38-v5.1-tp8-qsa.patch /opt/sglang/patches/v100-qwen38-v5.1-tp8-qsa.patch
COPY python/sglang/jit_kernel/csrc/sm70_nvfp4_moe_decode.cu \
     /opt/sglang/python/sglang/jit_kernel/csrc/sm70_nvfp4_moe_decode.cu
COPY patches/v100-qwen38-v5.1-tp8-decoder.patch /opt/sglang/patches/v100-qwen38-v5.1-tp8-decoder.patch
RUN cd /opt/sglang \
    && git apply --check patches/v100-qwen38-v5.1-tp8-qsa.patch \
    && git apply patches/v100-qwen38-v5.1-tp8-qsa.patch \
    && git apply --check patches/v100-qwen38-v5.1-tp8-decoder.patch \
    && git apply patches/v100-qwen38-v5.1-tp8-decoder.patch \
    && python -m py_compile \
       python/sglang/srt/layers/quantization/marlin_utils.py \
       python/sglang/srt/layers/quantization/modelopt_quant.py \
       python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py \
       python/sglang/jit_kernel/sm70_nvfp4_moe_decode.py \
       python/sglang/srt/layers/moe/moe_runner/marlin.py
