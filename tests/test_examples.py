from __future__ import annotations

import subprocess
import sys


def test_ivf_demo_runs_quickly(tmp_path):
    cmd = [
        sys.executable,
        "examples/ivf_demo.py",
        "--nb",
        "256",
        "--nq",
        "4",
        "--nlist",
        "8",
        "--nprobe",
        "4",
        "--topk",
        "4",
        "--device",
        "cpu",
        "--verify",
        "--seed",
        "123",
    ]
    subprocess.run(cmd, check=True)
