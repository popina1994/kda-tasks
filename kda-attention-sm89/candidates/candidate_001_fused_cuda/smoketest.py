import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.environ.get("KDA_ORACLE_DIR", os.path.expanduser("~/kimi-delta-attention")))

import torch
from reference import make_inputs, reference_naive, reference_chunkwise
from kernel import kda_candidate_001

torch.manual_seed(0)
q, k, v, alpha, beta = make_inputs(seq_len=128, seed=0)  # 2 chunks of 64

out_ref, state_ref = reference_chunkwise(q, k, v, alpha, beta)
out_cu, state_cu = kda_candidate_001(q, k, v, alpha, beta)

out_err = (out_ref - out_cu).abs().max().item()
state_err = (state_ref - state_cu).abs().max().item()
print(f"output shape: ref={tuple(out_ref.shape)} cuda={tuple(out_cu.shape)}")
print(f"state shape: ref={tuple(state_ref.shape)} cuda={tuple(state_cu.shape)}")
print(f"max abs output error: {out_err:.4e}")
print(f"max abs state error: {state_err:.4e}")
print(f"output rel-ish check: ref mean|.|={out_ref.abs().mean().item():.4e} cuda mean|.|={out_cu.abs().mean().item():.4e}")

if out_err < 2e-2 and state_err < 2e-2:
    print("PASS")
else:
    print("FAIL")
    # Help localize: compare first chunk only (no inter-chunk state carry issues)
    out_err_c0 = (out_ref[:, :64] - out_cu[:, :64]).abs().max().item()
    print(f"  chunk-0-only max abs output error: {out_err_c0:.4e}")
