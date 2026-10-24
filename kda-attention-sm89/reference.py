"""Reference wrapper for the kda-attention-sm89 task.

Exposes the ground-truth oracle (KDACore) and the parallel reference to port
(chunkwise_dplr_forward) at the task's fixed shapes, plus input generation.
Multi-head is handled by folding (batch, num_heads) into one leading batch
dimension B_eff = batch * num_heads, since KDA heads are independent of each
other (no cross-head mixing in the recurrence) -- this matches how the
reference repo itself expects to be called (it only has a batch dim).
"""
import sys
sys.path.insert(0, "/home/popina/kimi-delta-attention")

import torch
import torch.nn.functional as F

from kda.core import KDACore
from kda.chunkwise import chunkwise_dplr_forward

HEAD_DIM = 128
VALUE_DIM = 128
CHUNK_SIZE = 64
BATCH = 2
NUM_HEADS = 4
B_EFF = BATCH * NUM_HEADS


def make_inputs(seq_len: int, seed: int, device: str = "cuda", dtype=torch.float32):
    g = torch.Generator(device="cpu").manual_seed(seed)
    q = F.normalize(torch.randn(B_EFF, seq_len, HEAD_DIM, generator=g), dim=-1)
    k = F.normalize(torch.randn(B_EFF, seq_len, HEAD_DIM, generator=g), dim=-1)
    v = torch.randn(B_EFF, seq_len, VALUE_DIM, generator=g)
    alpha = torch.sigmoid(torch.randn(B_EFF, seq_len, HEAD_DIM, generator=g))
    beta = torch.sigmoid(torch.randn(B_EFF, seq_len, 1, generator=g))
    return tuple(t.to(device=device, dtype=dtype) for t in (q, k, v, alpha, beta))


def reference_naive(q, k, v, alpha, beta):
    """Ground-truth oracle: naive O(L) sequential recurrence (KDACore)."""
    core = KDACore(HEAD_DIM, VALUE_DIM)
    return core(q, k, v, alpha, beta)


def reference_chunkwise(q, k, v, alpha, beta, chunk_size: int = CHUNK_SIZE):
    """Parallel reference to port: chunked WY/DPLR form. Algebraically
    identical to reference_naive (confirmed by the repo's own 23-test suite)."""
    return chunkwise_dplr_forward(q, k, v, alpha, beta, chunk_size=chunk_size)


if __name__ == "__main__":
    # Smoke test: confirm the two references agree on this machine/GPU, at a
    # small size, before anything else in this task trusts them as oracles.
    q, k, v, alpha, beta = make_inputs(seq_len=256, seed=0)
    out_naive, state_naive = reference_naive(q, k, v, alpha, beta)
    out_chunk, state_chunk = reference_chunkwise(q, k, v, alpha, beta)
    out_err = (out_naive - out_chunk).abs().max().item()
    state_err = (state_naive - state_chunk).abs().max().item()
    print(f"max abs output error (naive vs chunkwise): {out_err:.3e}")
    print(f"max abs state error (naive vs chunkwise): {state_err:.3e}")
    assert out_err < 1e-3 and state_err < 1e-3, "references disagree on this GPU -- stop and investigate"
    print("PASS: references agree on this GPU")
