"""Shared pytest setup: put the task and the oracle on sys.path, and skip GPU
tests on machines that cannot run them."""
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK_ROOT = os.path.join(REPO_ROOT, "kda-attention-sm89")
sys.path.insert(0, TASK_ROOT)
sys.path.insert(0, os.environ.get("KDA_ORACLE_DIR", os.path.expanduser("~/kimi-delta-attention")))


def _skip_reason():
    try:
        import torch
    except ImportError:
        return "torch is not installed"
    if not torch.cuda.is_available():
        return "no CUDA device"
    if torch.cuda.get_device_capability() != (8, 9):
        return f"candidates are built for sm_89, found sm_{''.join(map(str, torch.cuda.get_device_capability()))}"
    return None


def pytest_collection_modifyitems(config, items):
    reason = _skip_reason()
    if reason is None:
        return
    marker = pytest.mark.skip(reason=reason)
    for item in items:
        item.add_marker(marker)
