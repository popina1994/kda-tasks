# Draft Plan: fp16-gemm-sm89

## Baseline
- Reference: torch.matmul on fp16 tensors (cuBLAS under the hood), run on the RTX 4070 SUPER.
- Validated against a numpy fp64 matmul of the same random inputs (ground truth), rtol/atol per contract.
- No existing implementation yet; this is a greenfield kernel.

## Risks and unknowns
- CuTe-DSL's documented optimization targets are Ampere/Hopper/Blackwell; Ada (sm_89) is supported by underlying CUTLASS but less battle-tested in CuTe-DSL examples. Example code in CuTe-DSL docs may assume TMA (Hopper+) and need rewriting to the Ampere mma.sync/cp.async style for sm_89.
- WSL2 + Nsight Compute: GPU performance counters can be blocked by the NVIDIA driver's permission restriction (NVreg_RestrictProfilingToAdminUsers) even when nvidia-smi works. Needs a smoke test before relying on ncu for evidence.
- 12GB VRAM is not a binding constraint at these shapes (max working set at 4096 cubed fp16 is roughly 100MB for A+B+C combined), so no risk there.
- The 50 percent-of-cuBLAS target is a placeholder sized for a single hand/agent-written tile kernel versus a vendor-tuned library, not a multi-iteration KDA-style result; may revise after seeing first numbers.

## Candidate directions (ranked)
1. Straightforward CuTe-DSL tiled GEMM: 128x128x32 tile, 2-stage shared-memory pipeline, mma.sync fp16-to-fp32 tensor-core MMA, no TMA. Expected: correct, moderate perf (illustrative baseline for the loop).
2. If (1) underperforms badly: add software pipelining depth (3-4 stages) and swizzled shared memory layout to cut bank conflicts.
3. Stretch: vectorized epilogue and split-K for better SM occupancy at smaller shapes.

Start with (1) only; (2) and (3) are contingent on profiling evidence from ncu-report-skill, not planned upfront.

## First concrete steps
1. Smoke-test Nsight Compute permissions on this machine (ncu --version, then a trivial kernel capture).
2. Write reference.py (numpy fp64 ground truth plus torch.matmul baseline plus TFLOPs calc).
3. Write validate.py implementing the correctness check in the contract.
4. Write kernel.py, candidate 1 CuTe-DSL GEMM.
5. Write benchmark.py, timing harness, TFLOPs vs cuBLAS ratio.
6. Run validate.py then benchmark.py, record in benchmark.csv and candidates.jsonl.
7. Profile with ncu if target not met; use ncu-report-skill's analyze_reports.py to find the bottleneck; revise.

## Validation and evaluation commands
- python validate.py
- python benchmark.py

## Evidence required to promote, revise, or reject
- validate.py exit 0 for all shapes and seeds.
- benchmark.csv row showing TFLOPs and percent-of-cuBLAS at each shape.
- If target metric not met on first candidate: an ncu report plus analyze_reports.py summary identifying the limiting factor (memory-bound vs compute-bound vs occupancy), to justify candidate 2 or 3.
