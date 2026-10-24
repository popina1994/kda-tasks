# Executable Plan: fp16-gemm-sm89

## Candidate 1
- Source: NVIDIA/cutlass examples/python/CuTeDSL/cute/ampere/kernel/dense_gemm/tensorop_gemm.py (unmodified), run via its own CLI.
- Why reuse instead of writing from scratch: this is the official, maintained Ampere-targeted (mma.sync/cp.async, no TMA) tensor-core GEMM reference for CuTe-DSL. Writing a parallel hand-rolled version would just reproduce the same tiling/pipelining/swizzle logic with higher risk of bugs. The KDA loop's job here is choosing it, validating it on this specific GPU, tuning its exposed knob (atom_layout_mnk), and producing evidence -- not re-deriving tensor-core GEMM from zero.
- Blocking issue found: nvidia-cutlass-dsl 4.8.0's torch dtype table references torch.float4_e2m1fn_x2 (FP4, Blackwell-era), which does not exist in the currently installed torch 2.2.1+cu121. Fix in progress: upgrade torch via pip.
- Steps once unblocked:
  1. Run with --skip_ref_check --benchmark none at mnkl=512,512,512,1 to confirm it compiles and launches on sm_89 at all.
  2. Run with correctness check enabled (drop --skip_ref_check) at each of M=N=K in {512,1024,2048,4096}, 5 seeds each -> validate.py wraps this.
  3. Run with --benchmark default --iterations 100 at each shape -> benchmark.py wraps this, also runs torch.matmul alone for the cuBLAS comparison point, records TFLOPs and %-of-cuBLAS to benchmark.csv.
  4. Sweep atom_layout_mnk over {(1,1,1), (2,1,1), (1,2,1), (2,2,1), (4,2,1), (2,4,1)} (subject to the divisibility asserts in the file) at the 4096 shape, record all in benchmark.csv, pick the best as the promoted candidate_001 configuration.
  5. If best result is still <50% of cuBLAS TFLOPs: capture an ncu report for the best config (pending the Nsight Compute permission fix) and consult ncu-report-skill reference docs (adapted for Ada metric names, not B200) to identify whether it is latency-, occupancy-, or bandwidth-bound, then decide whether a candidate_002 (manual tiling/pipeline change) is warranted.

## Validation command
python validate.py

## Evaluation command
python benchmark.py

## Promotion criteria (unchanged from contract)
Passes validate.py for all shapes/seeds AND benchmark.py reports >=50% of cuBLAS TFLOPs at 4096^3, with profiling evidence attached if that required a second candidate.
