# kda-tasks

Task workspaces from replicating NVIDIA's Kernel Design Agents (KDA) workflow
on a consumer GPU: RTX 4070 SUPER (sm_89, Ada), WSL2 Ubuntu-22.04, CUDA 12.8,
torch 2.14.1+cu130.

KDA blog post: https://nvlabs.github.io/kda/blog/2026-09-27-kda-for-kda/
KDA methodology repo (prompts/skills only): https://github.com/NVlabs/kda
(cloned separately at `~/kda-repo`, deliberately not part of this repo).

The blog's own benchmark (KDAttn/FlashKDA on a B300) needs Hopper/Blackwell
features (TMA, tcgen05) that Ada doesn't have, so this does the same
contract -> draft -> plan -> implement -> validate -> promote loop on tasks that
fit sm_89.

## Layout

| Directory | Task | Status |
|---|---|---|
| [gemm-sm89/](gemm-sm89/) | Rehearsal: CuTe-DSL Ampere dense GEMM | candidate_001 promoted |
| [kda-attention-sm89/](kda-attention-sm89/) | Fused Kimi Delta Attention forward (CUDA C++) | candidate_001 promoted |

Each task has `docs/` (task contract, draft, plan), `candidates.jsonl` (one
record per candidate with status and evidence), and `benchmark.csv` (results
accumulate across runs). `runs/` and `profile/` are placeholders for logs and
Nsight output.

### gemm-sm89

NVIDIA's unmodified `tensorop_gemm.py` from CUTLASS
(`~/cutlass-ref/examples/python/CuTeDSL/cute/ampere/kernel/dense_gemm/`),
tuned only by `atom_layout_mnk`. Correct on all 4 target shapes against
`torch.bmm`. At 4096^3 fp16, `atom_layout_mnk=2,2,1` gets 73.43 TFLOPs, 107.1%
of cuBLAS (68.54 TFLOPs). The naive `1,1,1` layout got 9% of cuBLAS. See
`gemm-sm89/benchmark.csv`.

```bash
cd ~/cutlass-ref/examples/python/CuTeDSL
python3 cute/ampere/kernel/dense_gemm/tensorop_gemm.py --mnkl 4096,4096,4096,1 \
  --atom_layout_mnk 2,2,1 --ab_dtype Float16 --c_dtype Float16 --acc_dtype Float32 \
  --a_major m --b_major n --c_major n --tolerance 1e-2   # expect PASS
```

### kda-attention-sm89

A fused CUDA C++ kernel for the chunkwise-DPLR forward pass of Kimi Delta
Attention (arXiv:2510.26692), checked against the oracle in
`~/kimi-delta-attention` (github.com/hwilner/kimi-delta-attention), which
`reference.py` and `validate.py` import by absolute path.

- `candidates/candidate_001_fused_cuda/`: one kernel launch per 64-row chunk,
  one CTA per (batch*head), 128 threads, about 80KB dynamic shared memory. No
  tensor cores.
- Validation matrix: seq_len in {512, 2048, 8192} x 5 seeds = 15 cases, max
  error about 3e-7 against a 2e-2 tolerance.
- Speedup over PyTorch-eager `chunkwise_dplr_forward` was roughly 8-20x across
  the runs in `kda-attention-sm89/benchmark.csv`, varying by run and seq_len.
  The eager baseline is slow mainly because its triangular inversion issues one
  CUDA op per row.
- Key algebraic step: the reference's `(C, C, d_h)` ratio tensor is
  multiplicatively separable, so the coupling and intra-chunk terms reduce to
  small GEMMs after pre-multiplying the decay into k and q. Details are in
  `docs/draft.md`.

```bash
cd kda-attention-sm89
python3 validate.py [candidate_name]   # default candidate_001_fused_cuda; expect ALL PASS, 15 cases
python3 benchmark.py [candidate_name]  # appends to benchmark.csv
```

To add a candidate, create `candidates/candidate_002_<name>/` with a
`kernel.py` that exposes `forward`, then run the commands above with its name.
No harness changes are needed. `libkda.so` is a build product and is not
committed; rebuild it with `candidates/candidate_001_fused_cuda/compile.sh`
(`nvcc -arch=sm_89 -O3 --shared -Xcompiler -fPIC`) before running.

## Provenance

This work started in a Windows-side Claude Code session, driving WSL through
`wsl.exe`. Its project notes were `kda_gemm_sm89_progress.md` in that session's
memory directory
(`C:\Users\djord\.claude\projects\c--Users-djord-Documents-Programs-Kernels\memory\`).
That session:

1. Set up the environment, including the torch 2.14.1 upgrade that
   `nvidia-cutlass-dsl` 4.8.0 requires.
2. Did the GEMM rehearsal.
3. Wrote and promoted the KDA attention kernel.
4. Reorganized `kda-attention-sm89` into the multi-candidate layout above.

This repo is the continuation, run natively inside WSL. The `~/kda-repo`
checkout was used for an initial re-verification attempt, but the work lives
here, so this directory is the repo of record. The re-verification checklist
for a fresh session:

1. `nvidia-smi`, `nvcc --version`, and
   `python3 -c "import torch; print(torch.__version__, torch.cuda.is_available())"`
   (expect `2.14.1+cu130 True`).
2. `validate.py` and `benchmark.py` in `kda-attention-sm89`.
3. The `tensorop_gemm.py` command above.

## Known limitations and next steps

- candidate_001 launches only 8 CTAs (batch 2 x 4 heads), far below the 36 SMs
  of a 4070 SUPER, and uses scalar FMA rather than `mma.sync`. Both are
  hypotheses, not profiled facts.
- Nsight Compute (`ncu`) fails under WSL with `ERR_NVGPUCTRPERM`. GPU
  performance counters have to be enabled on the Windows host (NVIDIA Control
  Panel, or the registry setting) before candidate_002 work can be
  evidence-driven.
