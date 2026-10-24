# Task Contract

- Task name: fp16-gemm-sm89
- Objective: Implement a correct, reasonably optimized FP16 tiled GEMM (C = A @ B, row-major, M=N=K in {512,1024,2048,4096}) in CuTe-DSL targeting compute capability 8.9 (NVIDIA RTX 4070 SUPER, Ada, no TMA/tcgen05), as a stand-in exercise to validate the KDA agent-flow loop end-to-end before attempting the real Kimi Delta Attention port.
- Correctness requirements: output matches torch.matmul (fp16 inputs, fp32 accumulation) within rtol=1e-2, atol=1e-2 on random inputs, for all target shapes, across 5 random seeds.
- Performance or quality target: achieve at least 50% of cuBLAS (torch.matmul) achieved TFLOPs on this GPU at M=N=K=4096, fp16 in/out.
- Allowed implementation approaches: Python + CuTe-DSL (nvidia-cutlass-dsl), compiled for sm_89 only. No TMA, no tcgen05 MMA, no Hopper/Blackwell-only intrinsics. PyTorch allowed only for reference/validation, not for the implementation itself.
- Validation command: python validate.py
- Evaluation command: python benchmark.py
- Promotion criteria: candidate passes validate.py for all shapes/seeds AND benchmark.py reports >=50% of cuBLAS TFLOPs at 4096^3, with a profile/ncu report attached as evidence.
