"""Correctness tests that run against every candidates/*/kernel.py.

A new candidate is picked up automatically: drop a directory with a kernel.py
that exposes forward(q, k, v, alpha, beta, chunk_size=64) and it is tested
against the same oracle with the same tolerance as validate.py.
"""
import importlib.util
import os
import sys

import pytest
import torch

from conftest import TASK_ROOT

pytest.importorskip("kda", reason="oracle repo not found; set KDA_ORACLE_DIR")
from reference import CHUNK_SIZE, make_inputs, reference_chunkwise, reference_naive  # noqa: E402
from validate import ATOL, RTOL  # noqa: E402

CANDIDATES_DIR = os.path.join(TASK_ROOT, "candidates")
CANDIDATES = sorted(
    name for name in os.listdir(CANDIDATES_DIR) if os.path.exists(os.path.join(CANDIDATES_DIR, name, "kernel.py"))
)


@pytest.fixture(scope="module", params=CANDIDATES)
def forward(request):
    name = request.param
    path = os.path.join(CANDIDATES_DIR, name)
    if name == "candidate_001_fused_cuda" and not os.path.exists(os.path.join(path, "libkda.so")):
        pytest.skip("libkda.so not built; run compile.sh")
    sys.path.insert(0, path)
    try:
        spec = importlib.util.spec_from_file_location(f"kda_candidate_{name}", os.path.join(path, "kernel.py"))
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(path)
    return module.forward


def assert_close(actual, expected, what):
    err = (actual - expected).abs().max().item()
    assert torch.allclose(actual, expected, rtol=RTOL, atol=ATOL), f"{what}: max abs error {err:.3e}"


@pytest.mark.parametrize("seed", [0, 1])
@pytest.mark.parametrize("seq_len", [64, 512, 2048])
def test_matches_chunkwise_reference(forward, seq_len, seed):
    inputs = make_inputs(seq_len=seq_len, seed=seed)
    out_ref, state_ref = reference_chunkwise(*inputs)
    out, state = forward(*inputs)
    assert_close(out, out_ref, "output")
    assert_close(state, state_ref, "final state")


def test_matches_naive_recurrence(forward):
    """Independent of the chunked reference: compare to the O(L) KDACore recurrence."""
    inputs = make_inputs(seq_len=256, seed=3)
    out_ref, state_ref = reference_naive(*inputs)
    out, state = forward(*inputs)
    assert_close(out, out_ref, "output vs naive recurrence")
    assert_close(state, state_ref, "final state vs naive recurrence")


@pytest.mark.parametrize("seq_len", [1, CHUNK_SIZE - 1, CHUNK_SIZE + 1, 3 * CHUNK_SIZE + 17])
def test_seq_len_not_a_multiple_of_chunk_size(forward, seq_len):
    inputs = make_inputs(seq_len=seq_len, seed=5)
    out_ref, state_ref = reference_chunkwise(*inputs)
    out, state = forward(*inputs)
    assert out.shape == out_ref.shape
    assert_close(out, out_ref, "output")
    assert_close(state, state_ref, "final state")


@pytest.mark.parametrize("batch", [1, 3])
def test_batch_sizes(forward, batch):
    inputs = tuple(t[:batch].contiguous() for t in make_inputs(seq_len=256, seed=7))
    out_ref, state_ref = reference_chunkwise(*inputs)
    out, state = forward(*inputs)
    assert_close(out, out_ref, "output")
    assert_close(state, state_ref, "final state")


def test_non_contiguous_inputs(forward):
    """Inputs that are transposed views must give the same answer as packed ones."""
    inputs = make_inputs(seq_len=256, seed=9)
    strided = tuple(t.transpose(1, 2).contiguous().transpose(1, 2) for t in inputs)
    assert not strided[0].is_contiguous()
    out_ref, state_ref = reference_chunkwise(*inputs)
    out, state = forward(*strided)
    assert_close(out, out_ref, "output")
    assert_close(state, state_ref, "final state")


def test_repeat_calls_do_not_leak_state(forward):
    """The kernel updates its state buffer in place; a second call must start fresh."""
    inputs = make_inputs(seq_len=256, seed=11)
    first_out, first_state = forward(*inputs)
    second_out, second_state = forward(*inputs)
    assert torch.equal(first_out, second_out)
    assert torch.equal(first_state, second_state)


def test_output_shapes_and_dtype(forward):
    q, k, v, alpha, beta = make_inputs(seq_len=128, seed=0)
    out, state = forward(q, k, v, alpha, beta)
    assert out.shape == v.shape
    assert state.shape == (q.shape[0], q.shape[-1], v.shape[-1])
    assert out.dtype == state.dtype == torch.float32
    assert out.is_cuda and state.is_cuda
