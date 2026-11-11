"""V100 TP4/TP8 decoder versus Marlin; run directly on one idle V100.

Reports CUDA graph latency with four independent route sets per graph
so the batch-one benchmark does not repeatedly read a single L2-resident route.
"""

import json
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.jit_kernel.sm70_nvfp4_moe_decode import sm70_nvfp4_moe_decode
from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
from sglang.srt.layers.moe.moe_runner.marlin import (
    MarlinMoeQuantInfo,
    fused_experts_none_to_marlin,
)
from sglang.srt.layers.quantization.marlin_utils import marlin_make_workspace


def latency(functions):
    for _ in range(3):
        for fn in functions:
            fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for fn in functions:
            fn()
    for _ in range(5):
        graph.replay()
    samples = []
    for _ in range(5):
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
        start.record()
        for _ in range(20):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / (20 * len(functions)))
    return sorted(samples)[len(samples) // 2]


def main():
    torch.manual_seed(101)
    workspace = marlin_make_workspace("cuda", max_blocks_per_sm=4)
    results = []
    for intermediate in (160, 96):
        w13 = torch.empty(
            (512, 160, 4 * intermediate), dtype=torch.int32, device="cuda"
        ).random_()
        w2 = torch.empty(
            (512, intermediate // 16, 5120), dtype=torch.int32, device="cuda"
        ).random_()
        s13 = torch.full(
            (512, 160, 2 * intermediate), 0x70, dtype=torch.uint8, device="cuda"
        ).view(torch.float8_e4m3fn)
        s2 = torch.full(
            (512, intermediate // 16, 2560), 0x70, dtype=torch.uint8, device="cuda"
        ).view(torch.float8_e4m3fn)
        g13 = torch.full((512,), 100.0, device="cuda")
        g2 = torch.full((512,), 100.0, device="cuda")
        quant = MarlinMoeQuantInfo(
            w13, w2, s13, s2, None, None, 4, w13_global_scale=g13, w2_global_scale=g2
        )
        config = SimpleNamespace(
            activation="silu",
            routed_scaling_factor=None,
            num_experts=512,
            num_local_experts=512,
            swiglu_limit=None,
            gate_up_input_scale=1.0,
            wide_output_scale=1.0,
        )
        for batch in (1, 2, 4, 8, 16, 32):
            specialized, generic = [], []
            for _ in range(4):
                x = torch.randn(batch, 2560, device="cuda", dtype=torch.float16)
                ids = torch.randint(
                    0, 512, (batch, 10), device="cuda", dtype=torch.int32
                )
                ids[0, 0] = 511
                weights = torch.softmax(torch.randn(batch, 10, device="cuda"), dim=-1)
                logits = torch.zeros(batch, 512, device="cuda", dtype=torch.float16)

                def decode(x=x, ids=ids, weights=weights):
                    return sm70_nvfp4_moe_decode(
                        x, w13, w2, s13, s2, g13, g2, ids, weights
                    )

                def marlin(x=x, ids=ids, weights=weights, logits=logits):
                    return fused_marlin_moe(
                        x,
                        w13,
                        w2,
                        s13,
                        s2,
                        logits,
                        weights,
                        ids,
                        w1_global_scale=g13,
                        w2_global_scale=g2,
                        workspace=workspace,
                        num_bits=4,
                    )

                out, ref = decode(), marlin()
                error = float((out.float() - ref.float()).norm() / ref.float().norm())
                assert error < 0.005, (intermediate, batch, error)
                if intermediate == 96 or batch <= 4:
                    dispatch = SimpleNamespace(
                        hidden_states=x,
                        topk_output=SimpleNamespace(
                            topk_ids=ids, topk_weights=weights, router_logits=logits
                        ),
                    )
                    with patch(
                        "sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe.fused_marlin_moe",
                        side_effect=AssertionError("specialization was bypassed"),
                    ):
                        routed = fused_experts_none_to_marlin(
                            dispatch, quant, config
                        ).hidden_states
                    torch.testing.assert_close(routed, out, rtol=0, atol=0)
                specialized.append(decode)
                generic.append(marlin)
            decode_us, marlin_us = latency(specialized), latency(generic)
            result = dict(
                intermediate=intermediate,
                batch=batch,
                decode_us=decode_us,
                marlin_us=marlin_us,
                speedup=marlin_us / decode_us,
                relative_l2=error,
            )
            results.append(result)
            print(json.dumps(result), flush=True)
    print(
        "PASS numerical agreement, real runner dispatch, and graph replay", flush=True
    )
    with open("decoder-benchmark.json", "w") as stream:
        json.dump(results, stream, indent=2)


if __name__ == "__main__":
    main()
