"""V100 numerical check for Qwen3.8 TP4/TP8 Marlin shapes (no checkpoint).

Run in the V100 serving environment: python test/manual/quant/test_sm70_nvfp4_tp.py
"""

import torch

from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
from sglang.srt.layers.quantization.gptq import gptq_marlin_moe_repack
from sglang.srt.layers.quantization.marlin_utils import (
    sm70_nvfp4_marlin_pad_moe_weights,
    sm70_nvfp4_marlin_process_global_scale,
    sm70_nvfp4_marlin_process_scales,
)


def dequant(weight, scale, global_scale):
    codes = torch.stack((weight & 15, weight >> 4), -1).flatten(-2).long()
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=weight.device,
    )
    return (
        lut[codes]
        * scale.float().repeat_interleave(16, -1)
        * global_scale[:, None, None]
    ).half()


def check(tp):
    torch.manual_seed(23)
    experts, topk, hidden, intermediate = 512, 10, 2560, 640 // tp
    w13 = torch.randint(
        256, (experts, 2 * intermediate, hidden // 2), dtype=torch.uint8, device="cuda"
    )
    w2 = torch.randint(
        256, (experts, hidden, intermediate // 2), dtype=torch.uint8, device="cuda"
    )
    s13 = (
        torch.rand(experts, 2 * intermediate, hidden // 16, device="cuda") * 0.5 + 0.25
    ).to(torch.float8_e4m3fn)
    s2 = (
        torch.rand(experts, hidden, intermediate // 16, device="cuda") * 0.5 + 0.25
    ).to(torch.float8_e4m3fn)
    g13 = torch.full((experts,), 0.01, device="cuda")
    g2 = torch.full((experts,), 0.02, device="cuda")
    ref13 = dequant(w13[:topk], s13[:topk], g13[:topk])
    ref2 = dequant(w2[:topk], s2[:topk], g2[:topk])
    w13, w2, s13, s2 = sm70_nvfp4_marlin_pad_moe_weights(w13, w2, s13, s2, True)

    def repack(w):
        e, n, pk = w.shape
        return gptq_marlin_moe_repack(
            w.view(torch.int32).transpose(1, 2).contiguous(),
            torch.empty((e, 0), dtype=torch.int32, device="cuda"),
            pk * 2,
            n,
            4,
        )

    w13, w2 = repack(w13), repack(w2)
    s13, f13 = sm70_nvfp4_marlin_process_scales(
        s13.transpose(1, 2).contiguous(), torch.float16
    )
    s2, f2 = sm70_nvfp4_marlin_process_scales(
        s2.transpose(1, 2).contiguous(), torch.float16
    )
    g13 = sm70_nvfp4_marlin_process_global_scale(g13, torch.float16) / f13
    g2 = sm70_nvfp4_marlin_process_global_scale(g2, torch.float16) / f2
    for batch in (1, 4, 17, 2048):
        x = torch.randn(batch, hidden, dtype=torch.float16, device="cuda") * 0.1
        ids = torch.arange(topk, dtype=torch.int32, device="cuda").repeat(batch, 1)
        weights = torch.softmax(torch.randn(batch, topk, device="cuda"), -1)
        expected = torch.zeros_like(x, dtype=torch.float32)
        for e in range(topk):
            gate, up = torch.nn.functional.linear(x, ref13[e]).chunk(2, -1)
            act = (torch.nn.functional.silu(gate) * up).half()
            expected += (
                torch.nn.functional.linear(act, ref2[e]).float() * weights[:, e, None]
            )

        def run():
            return fused_marlin_moe(
                x,
                w13,
                w2,
                s13,
                s2,
                torch.zeros(batch, experts, device="cuda"),
                weights,
                ids,
                num_bits=4,
                w1_global_scale=g13,
                w2_global_scale=g2,
            )

        actual = run()
        torch.testing.assert_close(actual.float(), expected, rtol=0.02, atol=2e-4)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, actual, rtol=0.005, atol=2e-5)
        print(
            f"PASS TP{tp} batch={batch} eager+graph max_error={(actual.float() - expected).abs().max().item():.6g}",
            flush=True,
        )


if __name__ == "__main__":
    for tp in (4, 8):
        check(tp)
