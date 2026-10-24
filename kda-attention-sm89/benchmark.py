"""Generic benchmarking harness for the kda-attention-sm89 task.

Usage: python benchmark.py [candidate_name]
Default candidate: candidate_001_fused_cuda. Appends one row per candidate
run to benchmark.csv rather than overwriting it, so results across
candidates accumulate in one place for comparison.
"""
import importlib
import os
import sys
import time
from datetime import datetime, timezone

TASK_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, TASK_ROOT)
sys.path.insert(0, os.environ.get("KDA_ORACLE_DIR", os.path.expanduser("~/kimi-delta-attention")))

import torch
from reference import make_inputs, reference_chunkwise

DEFAULT_CANDIDATE = "candidate_001_fused_cuda"
SEQ_LENS = [512, 2048, 8192]
WARMUP = 3
ITERS = 10
CSV_PATH = os.path.join(TASK_ROOT, "benchmark.csv")


def load_candidate(name: str):
    sys.path.insert(0, os.path.join(TASK_ROOT, "candidates", name))
    module = importlib.import_module("kernel")
    return module.forward


def timed(fn, *args, warmup=WARMUP, iters=ITERS):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1e6  # microseconds


def main():
    candidate_name = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CANDIDATE
    forward = load_candidate(candidate_name)
    print(f"candidate: {candidate_name}")

    write_header = not os.path.exists(CSV_PATH)
    meta = (
        torch.cuda.get_device_name(0),
        torch.__version__,
        datetime.now(timezone.utc).strftime("%Y-%m-%d"),
    )
    rows = []
    print("mode,seq_len,exec_us,speedup_vs_pytorch")
    for seq_len in SEQ_LENS:
        q, k, v, alpha, beta = make_inputs(seq_len=seq_len, seed=0)

        pt_us = timed(reference_chunkwise, q, k, v, alpha, beta)
        rows.append(("pytorch_eager", seq_len, round(pt_us, 2), 1.0, *meta))
        print(f"pytorch_eager,{seq_len},{pt_us:.2f},1.00")

        cu_us = timed(forward, q, k, v, alpha, beta)
        speedup = pt_us / cu_us
        rows.append((candidate_name, seq_len, round(cu_us, 2), round(speedup, 2), *meta))
        print(f"{candidate_name},{seq_len},{cu_us:.2f},{speedup:.2f}")

    with open(CSV_PATH, "a") as f:
        if write_header:
            f.write("mode,seq_len,exec_us,speedup_vs_pytorch,gpu,torch,date\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


if __name__ == "__main__":
    main()
