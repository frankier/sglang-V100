import importlib.util
from pathlib import Path

import torch

path = (
    Path.home()
    / ".scratch/sglang-v100-tp-20261111/cache/nvfp4_moe/sglang_sm70_nvfp4_moe_v100.so"
)
spec = importlib.util.spec_from_file_location("sglang_sm70_nvfp4_moe_v100", path)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
from sglang.jit_kernel.sm70_nvfp4_moe_decode import sm70_nvfp4_moe_decode  # noqa: E402

torch.manual_seed(111)
w13 = torch.empty((12, 160, 640), device="cuda", dtype=torch.int32).random_()
w2 = torch.empty((12, 10, 5120), device="cuda", dtype=torch.int32).random_()
s13 = torch.full((12, 160, 320), 0x70, dtype=torch.uint8, device="cuda")
s2 = torch.full((12, 10, 2560), 0x70, dtype=torch.uint8, device="cuda")
g13 = torch.full((12,), 100.0, device="cuda")
g2 = torch.full((12,), 100.0, device="cuda")
for batch in (1, 4, 8, 32):
    x = torch.randn(batch, 2560, device="cuda", dtype=torch.float16)
    ids = torch.randint(0, 12, (batch, 10), device="cuda", dtype=torch.int32)
    ids[:, -1] = -1
    weights = torch.softmax(torch.randn(batch, 10, device="cuda"), dim=-1)
    new = sm70_nvfp4_moe_decode(x, w13, w2, s13, s2, g13, g2, ids, weights)
    routes = batch * 10
    gate = torch.empty((40, routes, 320), device="cuda")
    act = torch.empty((routes, 160), device="cuda", dtype=torch.float16)
    down = torch.empty((routes, 5, 2560), device="cuda")
    old = torch.empty_like(x)
    original.decode(x, w13, w2, s13, s2, g13, g2, ids, weights, gate, act, down, old)
    torch.testing.assert_close(new, old, rtol=0, atol=0)
    print(f"PASS original TP4 decoder bitwise identical M={batch}", flush=True)
