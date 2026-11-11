import unittest

import torch

from sglang.srt.layers.quantization.marlin_utils import (
    sm70_nvfp4_marlin_pad_moe_weights,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


class TestSm70Nvfp4Padding(CustomTestCase):
    def test_gate_up_padding_preserves_expert_function(self):
        for intermediate in (80, 160):
            for gated in (True, False):
                with self.subTest(intermediate=intermediate, gated=gated):
                    torch.manual_seed(71)
                    shards = 2 if gated else 1
                    w13 = torch.randint(
                        256, (2, shards * intermediate, 32), dtype=torch.uint8
                    )
                    w2 = torch.randint(
                        256, (2, 64, intermediate // 2), dtype=torch.uint8
                    )
                    s13 = torch.ones((2, shards * intermediate, 4)).to(
                        torch.float8_e4m3fn
                    )
                    s2 = torch.ones((2, 64, intermediate // 16)).to(torch.float8_e4m3fn)
                    original = (w13, w2, s13, s2)
                    padded = sm70_nvfp4_marlin_pad_moe_weights(*original, gated)
                    if shards * intermediate % 64 == 0:
                        for before, after in zip(original, padded):
                            self.assertIs(before, after)
                    self.assertEqual(padded[0].shape[1] % 64, 0)
                    self.assertEqual(padded[1].shape[-1] * 2 % 16, 0)

                    def dequant(weight, scale):
                        codes = (
                            torch.stack((weight & 15, weight >> 4), -1)
                            .flatten(-2)
                            .long()
                        )
                        lut = torch.tensor(
                            [
                                0,
                                0.5,
                                1,
                                1.5,
                                2,
                                3,
                                4,
                                6,
                                0,
                                -0.5,
                                -1,
                                -1.5,
                                -2,
                                -3,
                                -4,
                                -6,
                            ]
                        )
                        return lut[codes] * scale.float().repeat_interleave(16, -1)

                    def forward(weights):
                        a, b, sa, sb = weights
                        projected = x @ dequant(a, sa).transpose(1, 2)
                        if gated:
                            gate, up = projected.chunk(2, -1)
                            projected = torch.nn.functional.silu(gate) * up
                        else:
                            projected = torch.nn.functional.silu(projected)
                        return projected @ dequant(b, sb).transpose(1, 2)

                    x = torch.randn(2, 3, 64) * 0.01
                    torch.testing.assert_close(
                        forward(padded), forward(original), rtol=1e-5, atol=1e-5
                    )


if __name__ == "__main__":
    unittest.main()
