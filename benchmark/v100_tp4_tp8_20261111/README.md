# Qwen3.8-Flash-Next on DGX-1: TP4 and TP8

Checkpoint: `RadixArk/Qwen3.8-Flash-Next-NVFP4`, revision
`7b719225242aacd3dbd3f9407468c2ee9a9d2594`.
Runtime baseline: published v5.1 image, digest
`sha256:80dccc86c3383cc81b0188e67cdd74961e9c4038a3986e1d0fd87e73733261e7`.

This records the original TP8 generic-Marlin baseline. See the follow-up
[TP8 decoder report](../v100_tp8_decoder_20261111/README.md) for the specialized
small-batch implementation and its validation.

## Fixes

1. TP8 shards the 640-wide routed expert intermediate to 80. Fused gate/up
   therefore has N=160, which fails Marlin's
   [N%64 repack requirement](https://github.com/zhinianqin/marlin_v100/blob/6d72a49939701d26b15b617a4cd2423174adb2d1/csrc/quantization/marlin/gptq_marlin_repack.cu#L405). After
   checkpoint sharding, pad each gate/up half to 96 and down K to 96, including
   their block scales. The padded codes and scales are zero. Marlin now sees
   gate/up N=192 and down K=96. Its existing generic kernels handle these
   shapes; no extension rebuild or TP4 tuning-table change is needed.
2. TP8 has three local QSA query heads, versus six at TP4. The native sparse
   prefill/decode/verification route only accepted six. Pad the query with
   three zero heads, run the existing six-head kernel against the same KV
   head, and discard those extra outputs. Query heads are independent, so
   this preserves the attention function. Materialize dense output rows before
   the fused output gate, which uses flat row addressing. The six-head TP4
   route stays intact.

The TP4 gated expert width (160) already satisfies the repacker and is returned
without allocation or parameter copies. In the run recorded here, TP8 used
generic Marlin MoE; the follow-up implementation adds TP8 specialization.

## Published-image integration

The v5.1 image contains FP16-cache and MTP ring fixes that are absent from the
checkout's QSA backend. Replacing that entire image file with the checkout
file would lose those fixes. The image-specific patch applies only the TP8
head adapter to the image's existing QSA backend. The other two changed Python
files were verified byte-for-byte identical between the unmodified checkout
and the published image.

Build a thin runtime layer from the repository root:

```bash
docker build -f docker/v100-qwen38-tp.Dockerfile -t sglang-v100:tp .
```

Validation used read-only source binds into the digest-pinned SIF: the two
quantization files above and the image's QSA backend with the patch applied.
The existing SIF and llmturso image pin were left unchanged. The Docker layer
was prepared and its patch application checked; an image build was not run.

For a source install, the Python changes are used directly. The existing
launcher defaults to TP4; append `--tensor-parallel-size 8` to select TP8.
Use the checkpoint's 262144-token context limit. A configured maximum context
does not by itself demonstrate that all concurrent requests fit in the KV pool.

## Validation

On `dgx1-02`, eight V100-SXM2 32 GB GPUs:

- CPU padding regression passed for gated and ungated expert computations.
- TP4 and TP8 MoE numerical tests passed with 512 experts at batches 1, 4,
  17, and 2048, in eager execution and CUDA graphs. Maximum absolute error
  against the dequantized PyTorch reference was below 4e-6.
- All 16 attention cases passed: three/six query heads, E5M2/FP16 KV,
  decode/target verification/draft extension with CUDA graphs, and sparse
  prefill. Dense output rows and the downstream fused output gate were also
  checked against PyTorch. The image's existing FP16-cache implementation was
  retained.
- Full TP4 and TP8 serving passed in job `3038851`, which completed in
  **48m44s with exit code 0**. Both used MTP-3/4, FP16 KV, eight request slots,
  and returned four correct basic responses. The pools were 277280 tokens
  for TP4 and 554560 for TP8.
- TP4 served one 262016-input-token request; TP8 served two distinct requests
  submitted together. Each generated exactly 64 output tokens, with zero
  cached input tokens and zero retractions. The inputs repeat token IDs 42
  and 43 to test capacity; they are not a model-quality benchmark. These
  checks establish admission and completion, without proving simultaneous GPU
  batching.
- [Numerical-test output](numerics.log.gz) was saved before full model startup.

Evidence: [TP4 results](tp4-results.json), [TP8 results](tp8-results.json),
[TP4 server log](tp4-server.log.gz), [TP8 server log](tp8-server.log.gz),
[Slurm test output](serving.log.gz), [tested source hashes](sources.json), and
[exact launch plans](launch-plans.json). Server termination messages at the log
tails come from the harness's controlled cleanup after the checks.

Tests, run in the serving environment:

```bash
python test/registered/unit/layers/quantization/test_sm70_nvfp4_padding.py
python test/manual/quant/test_sm70_nvfp4_tp.py
python test/manual/attention/test_sm70_qsa_tp.py
```

Remote working files and raw logs are under
`~/.scratch/sglang-v100-tp-20261111/` on Turso. Focused Ruff checks and
`git diff --check` passed; existing ModelOpt F841/F541 findings were excluded.

## Parallelism choice

For independent users, prefer two TP4 replicas. One TP4 server was faster than
TP8 in this short-prompt comparison despite using half the GPUs:

| Single-server configuration | GPUs | 1 client, output tok/s | 4 clients, aggregate output tok/s |
| --- | ---: | ---: | ---: |
| TP4 | 4 | 97.8 | 230.4 |
| TP8 | 8 | 77.2 | 205.6 |

Each timed request asked the same short question, used greedy decoding with
thinking disabled, and generated 256 tokens with EOS ignored. One cohort per
concurrency was measured, after the basic responses. Rates include prefill and
HTTP overhead. The recommendation follows from these single-server results
and topology; routed two-replica throughput was not measured here.

The DGX-1 has two fully connected four-GPU NVLink groups, and the fork's fastest
small-batch expert kernels in this baseline were specialized to TP4. This is ordinary replica
data parallelism, with a separate model and KV pool per replica. It is distinct
from DP attention or expert parallelism; see the
[SGLang DP guide](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/dp_dpa_smg_guide.mdx).

TP8 remains useful for one larger shared KV pool. Its 554560-token pool can
admit request-length combinations that would fragment two separate
277280-token pools. Both layouts retain the model's 262144-token context limit.
