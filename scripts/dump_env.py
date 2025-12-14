"""Dump a reproducible environment snapshot into benchmarks/env.json.

This is intentionally lightweight (no external deps) and focuses on:
- OS / Python / Torch build info (ROCm/CUDA)
- CPU / GPU identification
- Thread settings
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch


@dataclass
class EnvSnapshot:
    timestamp: str
    host_os: str
    host_platform: str
    host_cpu: str
    python_version: str
    torch_version: str
    torch_git_version: Optional[str]
    torch_hip_version: Optional[str]
    torch_cuda_version: Optional[str]
    torch_num_threads: int
    torch_num_interop_threads: int
    cuda_is_available: bool
    device_name: Optional[str]
    device_total_memory_bytes: Optional[int]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


def main() -> None:
    device_name: Optional[str] = None
    device_total_memory_bytes: Optional[int] = None
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        device_total_memory_bytes = int(props.total_memory)

    snap = EnvSnapshot(
        timestamp=datetime.now().isoformat(timespec="seconds"),
        host_os=f"{platform.system()} {platform.release()}",
        host_platform=platform.platform(),
        host_cpu=platform.processor() or "unknown",
        python_version=platform.python_version(),
        torch_version=torch.__version__,
        torch_git_version=getattr(torch.version, "git_version", None),
        torch_hip_version=getattr(torch.version, "hip", None),
        torch_cuda_version=getattr(torch.version, "cuda", None),
        torch_num_threads=int(torch.get_num_threads()),
        torch_num_interop_threads=int(torch.get_num_interop_threads()),
        cuda_is_available=bool(torch.cuda.is_available()),
        device_name=device_name,
        device_total_memory_bytes=device_total_memory_bytes,
    )

    out_path = Path("benchmarks") / "env.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(snap.to_json() + "\n", encoding="utf-8")
    print(str(out_path))


if __name__ == "__main__":
    main()

