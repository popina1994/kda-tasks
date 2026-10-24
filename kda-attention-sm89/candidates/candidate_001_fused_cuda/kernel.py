"""ctypes wrapper for candidate_001: fused-per-chunk CUDA kernel, host-side
chunk loop (mirrors the reference's own inter-chunk loop structure)."""
import ctypes
import os

import torch

_LIB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "libkda.so")
_lib = ctypes.CDLL(_LIB_PATH)
_lib.kda_chunk_launch.restype = ctypes.c_int
_lib.kda_chunk_launch.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p,
]

CHUNK_SIZE = 64


def _ptr(t: torch.Tensor) -> ctypes.c_void_p:
    return ctypes.c_void_p(t.data_ptr())


def kda_candidate_001(q, k, v, alpha, beta, chunk_size: int = CHUNK_SIZE):
    """Fused-CUDA-kernel-per-chunk forward pass. Same interface/semantics as
    kda.chunkwise.chunkwise_dplr_forward: inputs (B,L,Dh)/(B,L,Dh)/(B,L,Dv)/
    (B,L,Dh)/(B,L,1), returns (outputs (B,L,Dv), final_state (B,Dh,Dv))."""
    assert q.is_cuda and q.dtype == torch.float32
    batch, seq_len, head_dim = q.shape
    value_dim = v.shape[-1]
    device = q.device

    num_chunks = (seq_len + chunk_size - 1) // chunk_size
    pad_len = num_chunks * chunk_size - seq_len
    if pad_len > 0:
        q = torch.nn.functional.pad(q, (0, 0, 0, pad_len))
        k = torch.nn.functional.pad(k, (0, 0, 0, pad_len))
        v = torch.nn.functional.pad(v, (0, 0, 0, pad_len))
        alpha = torch.nn.functional.pad(alpha, (0, 0, 0, pad_len), value=1.0)
        beta = torch.nn.functional.pad(beta, (0, 0, 0, pad_len), value=0.0)

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    alpha = alpha.contiguous()
    beta = beta.squeeze(-1).contiguous()  # (B, L) -- kernel expects no trailing 1-dim

    state = torch.zeros(batch, head_dim, value_dim, device=device, dtype=torch.float32).contiguous()
    chunk_outputs = []

    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)

    for c in range(num_chunks):
        # Each slice gets its own freshly-allocated contiguous buffer -- a
        # strided view into a larger pre-allocated tensor would silently
        # corrupt writes, since the kernel's indexing assumes a tightly
        # packed (B, chunk_size, Dv) layout with no per-batch gaps.
        qc = q[:, c * chunk_size:(c + 1) * chunk_size, :].contiguous()
        kc = k[:, c * chunk_size:(c + 1) * chunk_size, :].contiguous()
        vc = v[:, c * chunk_size:(c + 1) * chunk_size, :].contiguous()
        ac = alpha[:, c * chunk_size:(c + 1) * chunk_size, :].contiguous()
        bc = beta[:, c * chunk_size:(c + 1) * chunk_size].contiguous()
        out_c = torch.empty(batch, chunk_size, value_dim, device=device, dtype=torch.float32)

        err = _lib.kda_chunk_launch(
            _ptr(qc), _ptr(kc), _ptr(vc), _ptr(ac), _ptr(bc),
            _ptr(state), _ptr(out_c), batch, stream,
        )
        if err != 0:
            raise RuntimeError(f"kda_chunk_launch CUDA error code {err} at chunk {c}")
        chunk_outputs.append(out_c)

    outputs = torch.cat(chunk_outputs, dim=1)
    if pad_len > 0:
        outputs = outputs[:, :-pad_len]
    return outputs, state


# Standardized entry point every candidate's kernel.py exposes, so the
# generic validate.py/benchmark.py harnesses never need to know a
# candidate-specific function name.
forward = kda_candidate_001
