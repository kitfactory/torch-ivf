from __future__ import annotations

import json
import subprocess
import sys

import pytest
import torch


def _run_benchmark(device: str, extra: list[str] | None = None):
    cmd = [
        sys.executable,
        "scripts/benchmark.py",
        "--nb",
        "512",
        "--nq",
        "16",
        "--nlist",
        "8",
        "--nprobe",
        "4",
        "--topk",
        "4",
        "--device",
        device,
        "--json",
    ]
    if extra:
        cmd.extend(extra)
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(completed.stdout.strip())


def test_benchmark_cpu_runs():
    data = _run_benchmark("cpu")
    assert data["library"] == "torch_ivf"
    assert data["device"] == "cpu"
    assert data["nb"] == 512
    assert "train_ms" in data
    assert data["backend"] == "CPU"
    assert data["device_name"]
    assert data["host_os"]
@pytest.mark.skipif(not torch.version.hip, reason="ROCm device not available")
def test_benchmark_rocm_runs():
    data = _run_benchmark("cuda")
    assert data["device"].startswith("cuda")


def test_benchmark_faiss_cpu_runs():
    cmd = [
        sys.executable,
        "scripts/benchmark_faiss_cpu.py",
        "--nb",
        "2048",
        "--nq",
        "32",
        "--nlist",
        "16",
        "--nprobe",
        "4",
        "--topk",
        "5",
    ]
    completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(completed.stdout.strip())
    assert data["library"] == "faiss_cpu"
    assert data["device"] == "cpu"
    assert data["nlist"] == 16
