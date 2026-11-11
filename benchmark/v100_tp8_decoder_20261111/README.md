# TP8 small-batch NVFP4 MoE decode on V100

The Qwen3.8 Flash Next TP8 routed expert has 80 logical intermediate channels,
padded to 96 by the NVFP4 weight loader. The specialized SM70 decoder now
instantiates both the existing TP4 I=160 kernel and a TP8 I=96 kernel. It reads
Marlin-packed weights directly; no additional serving-time repacking is needed.
The gate/up projection retains 40 splits. The down projection uses three splits
instead of five, preserving 32-element FP16 partials and FP32 final reductions.

TP8 dispatch accepts 1–32 FP16 token rows, including MTP draft/verification batches,
with 512 experts and top-k 10. Larger batches and unsupported configurations
retain generic Marlin. The source checkout's TP4 dispatch remains limited to
four rows. The published-image integration preserves v5.1's existing optional
wider TP4 dispatch and its environment controls.

## Reproduce

Build the derived runtime from the repository root:

```bash
docker build -f docker/v100-qwen38-tp.Dockerfile -t sglang-v100:tp .
```

The Dockerfile applies a targeted Python patch and replaces the decoder's CUDA
source. It preserves the published image's additional FP16 KV/MTP fixes and
batched-MoE controls. The extension is JIT compiled on first use; use a writable,
disk-backed `SGLANG_V100_NVFP4_MOE_BUILD_DIR` outside the container for its cache.
No new environment flag is needed to enable TP8 specialization.

On a V100 in the serving environment:

```bash
python -m pytest -q test/registered/unit/layers/moe/test_sm70_nvfp4_moe_decode.py
python test/manual/kernels/moe/benchmark_sm70_nvfp4_decode.py
```

The numerical regression covers both widths at batches 1, 4, 8, and 32, varying
FP4 codes, per-block and per-expert scales, masked routes, unaligned metadata,
and CUDA graph capture/replay with updated inputs. It compares against an
independent dequantized FP32 PyTorch implementation. The manual benchmark uses
512 experts, checks real runner dispatch and agreement with generic Marlin,
and times CUDA graph replay with four distinct route sets to avoid repeatedly
reading just one batch-one route from L2.

## Kernel results

Slurm job `3048522`, `dgx1-02`, V100-SXM2 32 GB, CUDA 12.8, Torch 2.9.1:

- All 25 GPU tests passed. The numerical tests require relative L2 error and
  maximum error normalized to the reference's largest magnitude below 0.003.
  The additional TP4 M=8 case exposed a cancellation near zero outside the
  previous absolute per-element tolerance; both normalized checks passed.
- The TP4 specialization produced **bitwise-identical** results to the original
  image extension at M=1, 4, 8, and 32 in a separate regression comparison.
- The 512-expert benchmark checked generic-Marlin agreement and actual runner
  dispatch for every TP8 batch below, with CUDA graph replay. Latencies are
  medians of five samples, each timing 80 layer calls, after warmup. Four route
  sets share the same weight tensors; routes include expert 511.

| TP8 rows | Generic Marlin, µs | Specialized decoder, µs | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 114.7 | 21.1 | 5.43× |
| 2 | 125.7 | 29.5 | 4.26× |
| 4 | 154.3 | 45.3 | 3.40× |
| 8 | 189.6 | 75.6 | 2.51× |
| 16 | 386.5 | 144.4 | 2.68× |
| 32 | 599.4 | 278.0 | 2.16× |

These are isolated MoE layer timings, excluding attention, routing, TP
collectives, and request scheduling. They do not imply the same end-to-end
speedup. Full results: [kernel-benchmark.json](kernel-benchmark.json).

## Full-model serving results

A/B run in Slurm job `3049330` on the same eight-GPU DGX-1 allocation,
completed in 13m24s with exit code 0. Both
variants use the same checkpoint (`RadixArk/Qwen3.8-Flash-Next-NVFP4`, revision
`7b719225242aacd3dbd3f9407468c2ee9a9d2594`), runtime, quantization/QSA overlays,
FP16 KV, MTP-3/4, eight request slots, and a 554560-token KV pool. The baseline
binds the image's original TP4-only runner; the specialized variant binds the
patched runner. Before each server starts, the harness checks the SHA256 of
the runner file **inside the container** against the expected source, avoiding
ambiguous duplicate bind destinations. Both use the same new extension's
unchanged routing kernels.

| TP8 configuration | 1 client, output tok/s | 4 clients, aggregate output tok/s |
| --- | ---: | ---: |
| Generic Marlin | 77.1 | 201.5 |
| Specialized MoE decoder | 86.1 | 243.3 |
| Improvement | 11.7% | 20.7% |

Each request generates 256 tokens, with greedy decoding, thinking disabled,
and EOS ignored. The prompt asks how a computer executes a Python program.
One cohort per concurrency is measured after four basic warmup requests;
rates include prefill and HTTP overhead. These are workload-specific samples,
not a concurrency sweep or a statistical confidence interval. Raw responses
and timings are retained in [baseline-throughput.json](baseline-throughput.json)
and [specialized-throughput.json](specialized-throughput.json).

The specialized server also:

- Returned exactly `4` to each of four concurrent basic arithmetic requests.
- Completed two distinct 262016-input-token requests submitted together, each
  producing 64 output tokens, with zero cached input tokens and zero retractions.
  Both completed in 143.5 seconds wall time. These repeated-token inputs probe
  pool capacity and completion, not quality or simultaneous GPU batching.

Evidence: [numerical and kernel output](numerics.log.gz),
[serving harness output](serving.log.gz), [launch plan](launch-plan.json),
[tested source hashes](sources.json), [baseline server log](baseline-server.log.gz),
[specialized server log](specialized-server.log.gz),
[basic responses](specialized-responses.json),
[long-context results](specialized-capacity.json), and
[server configuration/state](specialized-server-info.json).
SIGTERM/SIGQUIT messages at the log tails come from controlled harness cleanup.

Validation used read-only source binds into the derived v5.1 SIF identified in
[sif.sha256](sif.sha256); the shared SIF was not modified. The image patch was
checked against the captured original files and reproduces the tested Python
sources. The Docker layer was not built or published.

The TP8 kernel improves both tested serving workloads. Replica data parallelism
remains a separate throughput/memory tradeoff: this run does not measure two
routed TP4 replicas. The [earlier report](../v100_tp4_tp8_20261111/README.md)
records TP4's single-server baseline; its TP8 recommendation predates this
specialization. TP8 retains one larger shared KV pool.
