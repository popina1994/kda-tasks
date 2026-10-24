import os
import sys
import time

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark.csv")
sys.path.insert(0, os.environ.get("CUTLASS_CUTE_DSL_DIR", os.path.expanduser("~/cutlass-ref/examples/python/CuTeDSL")))

import torch
import cutlass
from cute.ampere.kernel.dense_gemm.tensorop_gemm import run as gemm_run

SHAPES = [(512, 512, 512, 1), (1024, 1024, 1024, 1), (2048, 2048, 2048, 1), (4096, 4096, 4096, 1)]
ATOM_LAYOUTS = [(1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1), (4, 2, 1), (2, 4, 1)]


def cublas_tflops(m, n, k, iterations=50, warmup=5):
    a = torch.randn((m, k), dtype=torch.float16, device="cuda")
    b = torch.randn((k, n), dtype=torch.float16, device="cuda")
    for _ in range(warmup):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    us_per_iter = (elapsed / iterations) * 1e6
    flops = 2 * m * n * k
    tflops = flops / (us_per_iter * 1e-6) / 1e12
    return tflops, us_per_iter


def main():
    print("mode,M,N,K,atom_layout,exec_us,TFLOPs,pct_of_cublas")
    rows = []

    for (m, n, k, l) in SHAPES:
        cublas_tf, cublas_us = cublas_tflops(m, n, k)
        rows.append(("cublas_ref", m, n, k, "-", round(cublas_us, 2), round(cublas_tf, 2), 100.0))
        print(f"cublas_ref,{m},{n},{k},-,{cublas_us:.2f},{cublas_tf:.2f},100.00")

        layouts_to_try = ATOM_LAYOUTS if (m, n, k) == (4096, 4096, 4096) else [(2, 2, 1)]
        for atom in layouts_to_try:
            try:
                exec_us = gemm_run(
                    (m, n, k, l),
                    cutlass.Float16, cutlass.Float16, cutlass.Float32,
                    "m", "n", "n",
                    atom,
                    tolerance=1e-2,
                    warmup_iterations=5,
                    iterations=50,
                    skip_ref_check=True,
                    benchmark=True,
                )
                flops = 2 * m * n * k
                tflops = flops / (exec_us * 1e-6) / 1e12
                pct = tflops / cublas_tf * 100
                rows.append(("cutlass_dsl", m, n, k, str(atom), round(exec_us, 2), round(tflops, 2), round(pct, 1)))
                print(f"cutlass_dsl,{m},{n},{k},{atom},{exec_us:.2f},{tflops:.2f},{pct:.1f}")
            except Exception as e:
                print(f"cutlass_dsl,{m},{n},{k},{atom},FAILED,{type(e).__name__}: {e}")

    with open(CSV_PATH, "w") as f:
        f.write("mode,M,N,K,atom_layout,exec_us,TFLOPs,pct_of_cublas\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")


if __name__ == "__main__":
    main()
