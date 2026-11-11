"""Check TP4/TP8 native QSA routing against PyTorch on a V100."""

from types import SimpleNamespace

import torch

from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.layers.attention.tilelang_fa_v100 import _decode_cuda
from sglang.srt.layers.elementwise.sigmoid_mul import fused_sigmoid_mul
from sglang.srt.model_executor.forward_batch_info import ForwardMode


def check(heads, dtype):
    torch.manual_seed(79)
    rows, context = 4, 64
    q = torch.randn(rows, heads, 256, device="cuda", dtype=torch.float16) * 0.1
    k = (torch.randn(256, 1, 256, device="cuda") * 0.1).to(dtype)
    v = (torch.randn(256, 1, 256, device="cuda") * 0.1).to(dtype)
    table = torch.randperm(256, device="cuda", dtype=torch.int32).reshape(rows, context)
    reqs = torch.arange(rows, dtype=torch.int32, device="cuda")
    lengths = torch.tensor([47, 61, 53, 63], dtype=torch.int32, device="cuda")
    indices = torch.arange(64, dtype=torch.int32, device="cuda").repeat(rows, 1)
    indices[:, 9] = -1
    metadata = SimpleNamespace(sequence_lengths=lengths, row_req_pool_indices=reqs)
    backend = QwenSparseAttnBackend.__new__(QwenSparseAttnBackend)
    backend.token_to_kv_pool = SimpleNamespace(
        get_key_buffer=lambda _: k, get_value_buffer=lambda _: v
    )
    backend.req_to_token_pool = SimpleNamespace(req_to_token=table)
    backend.max_context_len = 262144
    backend._resolve_metadata = lambda _: metadata
    backend._sm70_sparse_decode_logged = False
    layer = SimpleNamespace(layer_id=0, scaling=1 / 16)
    expected = []
    for row in range(rows):
        logical = indices[row]
        logical = logical[(logical >= 0) & (logical < lengths[row])]
        slots = table[row, logical.long()].long()
        scores = q[row].float() @ k[slots, 0].float().T / 16
        expected.append(torch.softmax(scores, -1) @ v[slots, 0].float())
    expected = torch.stack(expected)
    for mode in (
        ForwardMode.DECODE,
        ForwardMode.TARGET_VERIFY,
        ForwardMode.DRAFT_EXTEND_V2,
    ):
        batch = SimpleNamespace(forward_mode=mode, req_pool_indices=reqs)
        assert backend._can_use_sm70_sparse_decode(q, k, v, batch, metadata, indices)

        def run():
            output = backend._forward_paged_attention(q, layer, batch, indices)
            assert output.is_contiguous(), "The fused output gate requires dense rows"
            return output.reshape_as(q)

        actual = run()
        torch.testing.assert_close(actual.float(), expected, rtol=0.01, atol=5e-5)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(captured, actual, rtol=0.002, atol=1e-5)
        # Qwen's gate supports strided gate rows, but assumes dense attention
        # rows. A three-head slice of a six-head output violates that contract.
        gate = torch.randn(rows, heads, 512, dtype=q.dtype, device=q.device)[..., 256:]
        gated = fused_sigmoid_mul(actual.reshape(rows, -1), gate, inplace=True)
        torch.testing.assert_close(
            gated.reshape_as(q).float(),
            expected * torch.sigmoid(gate.float()),
            rtol=0.01,
            atol=5e-5,
        )
        print(
            f"PASS QSA heads={heads} {dtype} {mode.name} eager+graph+gate", flush=True
        )

    # Sparse chunk-prefill uses the same six-head kernel ABI, but a causal
    # selection bound for each row. The first three heads must remain exact.
    batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND, req_pool_indices=reqs[:1], seq_lens=lengths[:1]
    )
    assert backend._can_use_sm70_sparse_prefill(q, k, v, batch, indices)
    native_q = torch.nn.functional.pad(q, (0, 0, 0, 3)) if heads == 3 else q
    native_output = _decode_cuda.sm70_cuda_qsa_prefill(
        native_q, k, v, table, reqs[:1], indices, lengths[:1], 1 / 16
    )[:, :heads]
    actual = backend._pad_extend_output(native_output, rows)
    assert actual.is_contiguous(), "Prefill must also return dense attention rows"
    actual = actual.reshape_as(q)
    expected = []
    for row in range(rows):
        count = min(indices.shape[1], int(lengths[0]) - rows + row + 1)
        logical = indices[row, :count]
        logical = logical[(logical >= 0) & (logical < int(lengths[0]) - rows + row + 1)]
        slots = table[0, logical.long()].long()
        scores = q[row].float() @ k[slots, 0].float().T / 16
        expected.append(torch.softmax(scores, -1) @ v[slots, 0].float())
    torch.testing.assert_close(
        actual.float(), torch.stack(expected), rtol=0.01, atol=5e-5
    )
    print(f"PASS QSA heads={heads} {dtype} sparse prefill", flush=True)


if __name__ == "__main__":
    dtypes = [torch.float8_e5m2]
    # The published v5.1 image also supplies native FP16-cache kernels.
    if hasattr(_decode_cuda, "_qsa_cache_view"):
        dtypes.append(torch.float16)
    for dtype in dtypes:
        for heads in (6, 3):
            check(heads, dtype)
