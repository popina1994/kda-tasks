# Draft Plan: kda-attention-sm89

## Baseline
- Ground-truth oracle: `kda.core.KDACore` (naive O(L) sequential recurrence). Confirmed correct per its own docstring/tests.
- Parallel reference to port: `kda.chunkwise.chunkwise_dplr_forward`. Confirmed algebraically identical to KDACore by the repo's own test suite (23/23 passing on this machine, tests/test_kda_core.py + tests/test_chunkwise.py), before any kernel work started.
- No existing fused kernel baseline on this GPU. The only baseline is pure-PyTorch-eager `chunkwise_dplr_forward` itself, run on the RTX 4070 SUPER.

## Key algebraic simplification found while reading the reference (important for the implementation, not present explicitly in the reference's own comments)
The reference materializes a `ratio[t,s,h] = Gamma_t[h]/Gamma_s[h]` tensor of shape (C,C,d_h) explicitly for clarity, then reduces it with einsums. This is multiplicatively separable: `ratio[t,s,h] = Gamma_t[h] * (1/Gamma_s[h])`. That means every einsum that contracts `ratio` with k/q over the head dim is actually a plain small GEMM once the decay is pre-multiplied into k/q:
- `K_bar = k * Gamma`, `K_tilde = k / Gamma`, `Q_tilde = q * Gamma` (elementwise, (C, d_h) each; the reference already defines K_bar and Q_tilde by these names, just doesn't name K_tilde).
- `coupling[t,s] = beta_t * (K_bar @ K_tilde^T)[t,s]` (strictly lower-triangular part used).
- `intra[t,s] = (Q_tilde @ K_tilde^T)[t,s]` (lower-triangular part used, s<=t).
This avoids ever materializing a (C,C,d_h) tensor on the GPU and turns the two einsums into two small (C x d_h) @ (d_h x C) GEMMs (C=64, d_h=128 in the target config) -- this is the actual shape a real kernel should compute, and is the reason a fused kernel can beat PyTorch eager even without tensor cores: PyTorch eager pays for the full (C,C,d_h) intermediate plus per-op launch overhead that the factored GEMM form never needs.

## Per-chunk algorithm the kernel must implement (one chunk, one (batch,head) pair, state S0 carried in from previous chunk)
1. `log_gamma = cumsum(log(alpha), axis=seq)` -- (C, d_h).
2. `Gamma = exp(log_gamma)`; `K_bar = k*Gamma`; `K_tilde = k*exp(-log_gamma)`; `Q_tilde = q*Gamma`.
3. `coupling = beta[:,None] * (K_bar @ K_tilde^T)`, strictly-lower-triangular masked (C,C).
4. `rhs = beta[:,None] * (v - K_bar @ S0)` (C, d_v).
5. Forward substitution (sequential over the C=64 rows, data-parallel over d_v=128 columns): `U[t] = rhs[t] - sum_{s<t} coupling[t,s] * U[s]`. This is the one inherently-sequential part per chunk; everything else is matmul-parallel. C=64 steps is small -- the whole chunk's forward-sub should be cheap relative to the GEMMs if done with a block-wide reduction per step rather than one thread per step.
6. `intra = tril(Q_tilde @ K_tilde^T, s<=t)` (C,C).
7. `output = Q_tilde @ S0 + intra @ U` (C, d_v) -- this chunk's output rows.
8. `S_end = Gamma[-1,:,None] * S0 + (k * (Gamma[-1]/Gamma))^T @ U` -- state handed to the next chunk.
Inter-chunk: step 8's S_end becomes next chunk's S0; this loop over chunks is sequential (num_chunks iterations) for a given (batch,head), but independent across (batch,head) pairs, which is where the kernel should parallelize across CTAs.

## Risks and unknowns
- This is substantially more complex than the GEMM rehearsal: a correctness bug anywhere in the 8-step per-chunk math is easy to introduce and may not be obvious from a single aggregate tolerance check. Mitigation: validate.py should check intermediate correctness chunk-by-chunk against the PyTorch reference's own intermediates where feasible, not just final output, at least for the first debugging pass (can be relaxed once candidate_001 passes end-to-end).
- Forward substitution (step 5) is sequential over C=64 within a chunk. A naive one-thread-does-everything implementation would be slow; needs at least a block-wide parallel reduction per row. Getting this efficient is the main novel kernel-design risk, distinct from anything in the GEMM task.
- No fixed performance target exists yet (unlike the GEMM task, which had the 50%-of-cuBLAS number from the contract). First candidate's number becomes the number to beat for subsequent iterations.
- float32 vs float16 precision: the reference uses float32 throughout (no explicit fp16 path). Given linear-attention recurrences are more numerically sensitive than a single GEMM (errors compound across chunks via the carried state), candidate_001 should also use float32 rather than jumping to fp16, and the contract's tolerance (2e-2) already reflects a more error-tolerant comparison than the GEMM task's 1e-2 for this reason.
- Nsight Compute profiling is still blocked on this machine (ERR_NVGPUCTRPERM, needs the user to toggle a Windows-side setting) -- may become necessary here, unlike the GEMM task, if candidate_001's forward-substitution step is a bottleneck and needs real profiling evidence to diagnose rather than guesswork.

## Candidate directions (ranked)
1. **candidate_001**: Plain CUDA C++ (no tensor cores, no CuTe-DSL) kernel, one CTA per (batch*num_heads) pair, looping sequentially over chunks internally (keeping S0 resident in shared memory/registers across the loop, avoiding a host-side per-chunk launch). Small GEMMs (step 3,4,6,7,8) done as straightforward shared-memory tiled fp32 matmuls -- these are tiny (64x128 class) so simplicity matters more than tensor-core throughput here. Forward substitution via block-wide parallel reduction per row. Goal: get it *correct* first, and establish whatever speedup (if any) over PyTorch eager that correctness + kernel fusion alone buys.
2. candidate_002 (contingent on candidate_001's numbers): if the GEMM-shaped steps (3,4,6,7,8) dominate runtime (profiling needed to know), port those specific steps to CuTe-DSL tensor-core MMA (mma.sync, same sm_89-safe pattern validated in the GEMM rehearsal) while keeping the forward-substitution step in plain CUDA.
3. candidate_003 (stretch, only if evidence supports it): process multiple (batch,head) pairs per CTA or widen chunk_size, if occupancy profiling shows the GPU is under-utilized with one CTA per (batch,head) at the target batch=2,heads=4 (only 8 CTAs total -- likely far below what saturates an RTX 4070 SUPER's 36 SMs, so this is a real risk flagged upfront, not hypothetical).

Start with candidate_001 only.

## First concrete steps
1. Write `reference.py`: thin wrapper exposing `kda.core.KDACore` and `kda.chunkwise.chunkwise_dplr_forward` with the task's fixed shapes (head_dim=128, value_dim=128, chunk_size=64, batch=2, num_heads=4), plus a function to generate valid random inputs (q,k L2-normalized; alpha in (0,1] via sigmoid; beta in [0,1] via sigmoid) for a given seq_len and seed.
2. Write `validate.py`: for seq_len in {512, 2048, 8192} x 5 seeds, run candidate kernel, compare to `chunkwise_dplr_forward` output (and spot-check against `KDACore` directly for at least one small case) at rtol/atol=2e-2.
3. Write `kernel.cu` (or `.py` CuTe-DSL/ctypes wrapper, TBD once first draft of the CUDA is underway) implementing candidate_001 as scoped above.
4. Write `benchmark.py`: time PyTorch-eager `chunkwise_dplr_forward` vs the candidate kernel at seq_len=8192 (the contract's target), record to `benchmark.csv`.
5. Run validate.py -> benchmark.py -> record results in candidates.jsonl, same pattern as the GEMM task.
6. If validate.py fails: fall back to chunk-by-chunk intermediate comparison (per the risk note above) to localize the bug to one of the 8 steps before re-reading the whole kernel.

## Validation and evaluation commands
- `python validate.py`
- `python benchmark.py`

## Evidence required to promote, revise, or reject
- validate.py passing for all seq_len/seed combinations in the matrix.
- benchmark.csv showing the candidate's latency vs the PyTorch-eager baseline at seq_len=8192, whatever the resulting ratio is (no pre-set target, per the contract).
- If the first candidate is slower than the PyTorch-eager baseline: this itself is valid evidence, not a failure of the loop -- record it, and the promotion decision becomes "revise" (try to understand why) rather than "promote", following the contract's promotion rule (satisfies correctness AND improves-or-preserves the target metric; a regression is an explicit reason to revise, not silently discard).
