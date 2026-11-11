# TP4/TP8 weight preparation and QSA routing on the published v5.1 runtime.
# Keep v5.1's image-only FP16 KV and MTP ring fixes when updating QSA.
FROM geesegeesegeese/sglang-v100@sha256:80dccc86c3383cc81b0188e67cdd74961e9c4038a3986e1d0fd87e73733261e7

COPY python/sglang/srt/layers/quantization/marlin_utils.py \
     /opt/sglang/python/sglang/srt/layers/quantization/marlin_utils.py
COPY python/sglang/srt/layers/quantization/modelopt_quant.py \
     /opt/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py
COPY patches/v100-qwen38-v5.1-tp8-qsa.patch /opt/sglang/patches/v100-qwen38-v5.1-tp8-qsa.patch
RUN cd /opt/sglang \
    && git apply --check patches/v100-qwen38-v5.1-tp8-qsa.patch \
    && git apply patches/v100-qwen38-v5.1-tp8-qsa.patch \
    && python -m py_compile \
       python/sglang/srt/layers/quantization/marlin_utils.py \
       python/sglang/srt/layers/quantization/modelopt_quant.py \
       python/sglang/srt/layers/attention/qwen_sparse_attn_backend.py
