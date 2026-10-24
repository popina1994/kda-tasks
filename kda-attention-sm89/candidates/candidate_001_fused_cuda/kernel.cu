// Fused CUDA kernel for one chunk of the KDA chunkwise-DPLR forward pass.
// Fixed shapes for this task: C (chunk_size) = 64, Dh (head_dim) = 128, Dv (value_dim) = 128.
// One CTA per (batch*head) row; blockDim = 128 threads. S0 is read and overwritten in place
// (becomes the state handed to the next chunk). See docs/draft.md for the algebraic derivation
// (K_bar/K_tilde/Q_tilde factoring of the ratio[t,s,h] = Gamma_t[h]/Gamma_s[h] tensor into plain
// matmuls) and the per-step shared-memory/sync-point reasoning.

#include <cuda_runtime.h>
#include <math.h>

#define C 64
#define DH 128
#define DV 128
#define THREADS 128

extern "C" __global__ void kda_chunk_kernel(
    const float* __restrict__ q,     // (B, C, DH)
    const float* __restrict__ k,     // (B, C, DH)
    const float* __restrict__ v,     // (B, C, DV)
    const float* __restrict__ alpha, // (B, C, DH)
    const float* __restrict__ beta,  // (B, C)
    float* __restrict__ S0,          // (B, DH, DV) read+write in place
    float* __restrict__ out          // (B, C, DV)
) {
    extern __shared__ float smem[];
    float* log_gamma = smem;                  // C*DH floats
    float* cxc = log_gamma + C * DH;           // C*C floats (coupling, then reused as intra)
    float* U = cxc + C * C;                    // C*DV floats (rhs, then overwritten to U)

    const int b = blockIdx.x;
    const int tid = threadIdx.x;

    const float* qb = q + (size_t)b * C * DH;
    const float* kb = k + (size_t)b * C * DH;
    const float* vb = v + (size_t)b * C * DV;
    const float* ab = alpha + (size_t)b * C * DH;
    const float* betab = beta + (size_t)b * C;
    float* S0b = S0 + (size_t)b * DH * DV;
    float* outb = out + (size_t)b * C * DV;

    // --- Step A: log_gamma[t,h] = cumsum_t(log(alpha[t,h])), one channel h = tid per thread ---
    {
        const int h = tid; // THREADS == DH == 128
        float acc = 0.0f;
        for (int t = 0; t < C; ++t) {
            acc += logf(ab[t * DH + h]);
            log_gamma[t * DH + h] = acc;
        }
    }
    __syncthreads();

    // --- Step C: coupling[t,s] = beta_t * dot_h(K_bar[t,h], K_tilde[s,h]) for s<t, else 0 ---
    for (int p = tid; p < C * C; p += THREADS) {
        const int t = p / C;
        const int s = p % C;
        float val = 0.0f;
        if (s < t) {
            float acc = 0.0f;
            for (int h = 0; h < DH; ++h) {
                float g_t = log_gamma[t * DH + h];
                float g_s = log_gamma[s * DH + h];
                float kbar_t = kb[t * DH + h] * expf(g_t);
                float ktil_s = kb[s * DH + h] * expf(-g_s);
                acc += kbar_t * ktil_s;
            }
            val = betab[t] * acc;
        }
        cxc[t * C + s] = val;
    }

    // --- Step D: rhs[t,v] = beta_t * (v[t,v] - dot_h(K_bar[t,h], S0[h,v])) ---
    for (int p = tid; p < C * DV; p += THREADS) {
        const int t = p / DV;
        const int vv = p % DV;
        float acc = 0.0f;
        for (int h = 0; h < DH; ++h) {
            float kbar_t = kb[t * DH + h] * expf(log_gamma[t * DH + h]);
            acc += kbar_t * S0b[h * DV + vv];
        }
        U[t * DV + vv] = betab[t] * (vb[t * DV + vv] - acc);
    }
    __syncthreads();

    // --- Step E: forward substitution, U[t,v] = rhs[t,v] - sum_{s<t} coupling[t,s]*U[s,v] ---
    // Independent per value-column v (= tid here, THREADS == DV == 128): no cross-thread
    // dependency, since the recurrence never mixes different v. No sync needed inside this loop.
    {
        const int vv = tid;
        for (int t = 1; t < C; ++t) {
            float corr = 0.0f;
            for (int s = 0; s < t; ++s) {
                corr += cxc[t * C + s] * U[s * DV + vv];
            }
            U[t * DV + vv] -= corr;
        }
    }
    __syncthreads();

    // --- Step F: intra[t,s] = dot_h(Q_tilde[t,h], K_tilde[s,h]) for s<=t, else 0 (reuse cxc) ---
    for (int p = tid; p < C * C; p += THREADS) {
        const int t = p / C;
        const int s = p % C;
        float val = 0.0f;
        if (s <= t) {
            float acc = 0.0f;
            for (int h = 0; h < DH; ++h) {
                float qtil_t = qb[t * DH + h] * expf(log_gamma[t * DH + h]);
                float ktil_s = kb[s * DH + h] * expf(-log_gamma[s * DH + h]);
                acc += qtil_t * ktil_s;
            }
            val = acc;
        }
        cxc[t * C + s] = val;
    }
    __syncthreads();

    // --- Step G: out[t,v] = dot_h(Q_tilde[t,h], S0[h,v]) + sum_{s<=t} intra[t,s]*U[s,v] ---
    for (int p = tid; p < C * DV; p += THREADS) {
        const int t = p / DV;
        const int vv = p % DV;
        float acc = 0.0f;
        for (int h = 0; h < DH; ++h) {
            float qtil_t = qb[t * DH + h] * expf(log_gamma[t * DH + h]);
            acc += qtil_t * S0b[h * DV + vv];
        }
        for (int s = 0; s <= t; ++s) {
            acc += cxc[t * C + s] * U[s * DV + vv];
        }
        outb[t * DV + vv] = acc;
    }
    __syncthreads();

    // --- Step H: S_end[h,v] = Gamma_last[h]*S0[h,v] + sum_t (k[t,h]*Gamma_last[h]/Gamma[t,h])*U[t,v] ---
    for (int p = tid; p < DH * DV; p += THREADS) {
        const int h = p / DV;
        const int vv = p % DV;
        float g_last = log_gamma[(C - 1) * DH + h];
        float gamma_last = expf(g_last);
        float acc = gamma_last * S0b[h * DV + vv];
        for (int t = 0; t < C; ++t) {
            float ratio = expf(g_last - log_gamma[t * DH + h]);
            acc += kb[t * DH + h] * ratio * U[t * DV + vv];
        }
        S0b[h * DV + vv] = acc;
    }
}

extern "C" int kda_chunk_launch(
    const float* q, const float* k, const float* v, const float* alpha, const float* beta,
    float* S0, float* out, int batch_eff, cudaStream_t stream
) {
    size_t smem_bytes = (size_t)(C * DH + C * C + C * DV) * sizeof(float);
    cudaFuncSetAttribute(kda_chunk_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem_bytes);
    kda_chunk_kernel<<<batch_eff, THREADS, smem_bytes, stream>>>(q, k, v, alpha, beta, S0, out);
    return (int)cudaGetLastError();
}
