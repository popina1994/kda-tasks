"""Generic correctness-validation harness for the kda-attention-sm89 task.

Usage: python validate.py [candidate_name]
Default candidate: candidate_001_fused_cuda. Add a new candidate by dropping
a candidates/<name>/ directory with a kernel.py that exposes forward(q, k, v,
alpha, beta, chunk_size=64) -- this script never needs to change.
"""
import importlib
import os
import sys

TASK_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_ROOT)
sys.path.insert(0, "/home/popina/kimi-delta-attention")

import torch
from reference import make_inputs, reference_chunkwise

DEFAULT_CANDIDATE = "candidate_001_fused_cuda"
SEQ_LENS = [512, 2048, 8192]
SEEDS = [0, 1, 2, 3, 4]
RTOL, ATOL = 2e-2, 2e-2


def load_candidate(name: str):
    sys.path.insert(0, os.path.join(TASK_ROOT, "candidates", name))
    module = importlib.import_module("kernel")
    return module.forward


def main():
    candidate_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CANDIDATE
    forward = load_candidate(candidate_name)
    print(f"candidate: {candidate_name}")

    all_pass = True
    for seq_len in SEQ_LENS:
        for seed in SEEDS:
            q, k, v, alpha, beta = make_inputs(seq_len=seq_len, seed=seed)
            out_ref, state_ref = reference_chunkwise(q, k, v, alpha, beta)
            out_cu, state_cu = forward(q, k, v, alpha, beta)

            out_ok = torch.allclose(out_ref, out_cu, rtol=RTOL, atol=ATOL)
            state_ok = torch.allclose(state_ref, state_cu, rtol=RTOL, atol=ATOL)
            out_err = (out_ref - out_cu).abs().max().item()
            state_err = (state_ref - state_cu).abs().max().item()
            status = "PASS" if (out_ok and state_ok) else "FAIL"
            if status == "FAIL":
                all_pass = False
            print(f"seq_len={seq_len:5d} seed={seed} {status}  out_err={out_err:.3e} state_err={state_err:.3e}")

    print()
    print("ALL PASS" if all_pass else "SOME FAILED")
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
