"""Quick benchmark harness for torch-ivf on CPU / ROCm."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Optional

import torch

from torch_ivf.index import IndexIVFFlat, SearchParams


@dataclass
class BenchmarkResult:
    library: str
    device: str
    device_name: str
    backend: str
    search_mode: str
    chosen_mode: str
    auto_avg_group_size: Optional[float]
    auto_threshold: Optional[float]
    auto_search_avg_group_threshold: Optional[float]
    metric: str
    dim: int
    nb: int
    train_n: int
    nq: int
    nlist: int
    nprobe: int
    max_codes: int
    topk: int
    dtype: str
    train_ms: float
    add_ms: float
    search_ms: float
    search_ms_min: float
    warmup: int
    repeat: int
    qps: float
    torch_version: str
    torch_git_version: Optional[str]
    python_version: str
    host_os: str
    host_platform: str
    host_cpu: str
    torch_num_threads: int
    torch_num_interop_threads: int
    device_total_memory_bytes: Optional[int]
    mem_allocated_bytes: Optional[int]
    mem_reserved_bytes: Optional[int]
    train_peak_allocated_bytes: Optional[int]
    train_peak_reserved_bytes: Optional[int]
    add_peak_allocated_bytes: Optional[int]
    add_peak_reserved_bytes: Optional[int]
    search_peak_allocated_bytes: Optional[int]
    search_peak_reserved_bytes: Optional[int]
    timestamp: str
    label: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="torch-ivf benchmark (CPU / ROCm friendly).")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--nb", type=int, default=65536, help="number of base vectors")
    p.add_argument("--train-n", type=int, default=0, help="training sample count (0=auto)")
    p.add_argument("--nq", type=int, default=128, help="number of query vectors")
    p.add_argument("--nlist", type=int, default=1024)
    p.add_argument("--nprobe", type=int, default=16)
    p.add_argument("--max-codes", type=int, default=0, help="cap scanned candidates per query (0=unlimited)")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--metric", choices=["l2", "ip"], default="l2")
    p.add_argument("--device", default=None, help="torch device string (cpu, cuda, rocm, dml)")
    p.add_argument("--search-mode", choices=["matrix", "csr", "auto"], default="matrix")
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    p.add_argument("--warmup", type=int, default=1, help="warmup iterations for search timing")
    p.add_argument("--repeat", type=int, default=5, help="timed iterations for search timing")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--label", default=None, help="record label (default: torch_cpu/torch_rocm/...)")
    p.add_argument("--json", action="store_true", help="print JSON only (no extra logs)")
    return p.parse_args()

def _default_label(device: torch.device) -> str:
    if device.type == "cpu":
        return "torch_cpu"
    if device.type == "cuda" and torch.version.hip:
        return "torch_rocm"
    if device.type == "cuda":
        return "torch_cuda"
    if device.type == "dml":
        return "torch_dml"
    return f"torch_{device.type}"


def run_benchmark(args: argparse.Namespace) -> BenchmarkResult:
    device = torch.device(args.device) if args.device else torch.device("cpu")
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    train_generator = torch.Generator(device="cpu").manual_seed(args.seed + 1)

    nb, nq, dim = args.nb, args.nq, args.dim
    base = torch.randn(nb, dim, dtype=dtype, device=device, generator=generator)
    queries = torch.randn(nq, dim, dtype=dtype, device=device, generator=generator)
    train_n = int(args.train_n) if args.train_n else max(args.nlist * 2, args.nlist + 1)
    train_n = max(1, min(nb, train_n))
    train_x = base[:train_n]

    index = IndexIVFFlat(
        dim,
        metric=args.metric,
        nlist=args.nlist,
        nprobe=args.nprobe,
        device=device,
        dtype=dtype,
    )
    index.max_codes = args.max_codes
    index.search_mode = args.search_mode
    params = SearchParams(
        profile="speed",
        approximate=index.approximate_mode,
        nprobe=index.nprobe,
        max_codes=index.max_codes,
        debug_stats=True,
    )

    device_total_memory_bytes: Optional[int] = None
    mem_allocated_bytes: Optional[int] = None
    mem_reserved_bytes: Optional[int] = None
    train_peak_allocated_bytes: Optional[int] = None
    train_peak_reserved_bytes: Optional[int] = None
    add_peak_allocated_bytes: Optional[int] = None
    add_peak_reserved_bytes: Optional[int] = None
    search_peak_allocated_bytes: Optional[int] = None
    search_peak_reserved_bytes: Optional[int] = None
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        device_total_memory_bytes = int(props.total_memory)
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    index.train(train_x, generator=train_generator)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        train_peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
        train_peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
    t1 = time.perf_counter()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    index.add(base)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        add_peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
        add_peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
    t2 = time.perf_counter()

    train_ms = (t1 - t0) * 1000
    add_ms = (t2 - t1) * 1000

    warmup = max(0, int(args.warmup))
    repeat = max(1, int(args.repeat))
    for _ in range(warmup):
        index.search(queries, args.topk, params=params)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    times_ms: list[float] = []
    for _ in range(repeat):
        s0 = time.perf_counter()
        index.search(queries, args.topk, params=params)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times_ms.append((time.perf_counter() - s0) * 1000)

    if device.type == "cuda":
        search_peak_allocated_bytes = int(torch.cuda.max_memory_allocated(device))
        search_peak_reserved_bytes = int(torch.cuda.max_memory_reserved(device))
        mem_allocated_bytes = int(torch.cuda.memory_allocated(device))
        mem_reserved_bytes = int(torch.cuda.memory_reserved(device))

    search_ms = float(statistics.median(times_ms))
    search_ms_min = float(min(times_ms))
    qps = args.nq / (search_ms / 1000) if search_ms > 0 else float("inf")

    backend = _detect_backend(device)
    stats = index.last_search_stats or {}
    chosen_mode = str(stats.get("chosen_mode", args.search_mode))
    return BenchmarkResult(
        library="torch_ivf",
        device=str(device),
        device_name=_device_name(device),
        backend=backend,
        search_mode=args.search_mode,
        chosen_mode=chosen_mode,
        auto_avg_group_size=stats.get("auto_avg_group_size"),
        auto_threshold=stats.get("auto_threshold"),
        auto_search_avg_group_threshold=stats.get("auto_search_avg_group_threshold"),
        metric=args.metric,
        dim=dim,
        nb=nb,
        train_n=train_n,
        nq=nq,
        nlist=args.nlist,
        nprobe=index.nprobe,
        max_codes=index.max_codes,
        topk=args.topk,
        dtype=args.dtype,
        train_ms=round(train_ms, 3),
        add_ms=round(add_ms, 3),
        search_ms=round(search_ms, 3),
        search_ms_min=round(search_ms_min, 3),
        warmup=warmup,
        repeat=repeat,
        qps=round(qps, 3),
        torch_version=torch.__version__,
        torch_git_version=getattr(torch.version, "git_version", None),
        python_version=platform.python_version(),
        host_os=f"{platform.system()} {platform.release()}",
        host_platform=platform.platform(),
        host_cpu=platform.processor() or "unknown",
        torch_num_threads=int(torch.get_num_threads()),
        torch_num_interop_threads=int(torch.get_num_interop_threads()),
        device_total_memory_bytes=device_total_memory_bytes,
        mem_allocated_bytes=mem_allocated_bytes,
        mem_reserved_bytes=mem_reserved_bytes,
        train_peak_allocated_bytes=train_peak_allocated_bytes,
        train_peak_reserved_bytes=train_peak_reserved_bytes,
        add_peak_allocated_bytes=add_peak_allocated_bytes,
        add_peak_reserved_bytes=add_peak_reserved_bytes,
        search_peak_allocated_bytes=search_peak_allocated_bytes,
        search_peak_reserved_bytes=search_peak_reserved_bytes,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        label=args.label or _default_label(device),
    )


def _device_name(device: torch.device) -> str:
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(device)
    if device.type == "cpu":
        return platform.processor() or "CPU"
    if device.type == "dml":
        return "DirectML"
    return device.type




def _detect_backend(device: torch.device) -> str:
    if device.type == "cuda":
        if torch.version.hip:
            return f"ROCm {torch.version.hip}"
        if torch.version.cuda:
            return f"CUDA {torch.version.cuda}"
        return "CUDA"
    if device.type == "cpu":
        return "CPU"
    if device.type == "dml":
        return "DirectML"
    return device.type


def main() -> None:
    args = parse_args()
    result = run_benchmark(args)
    if args.json:
        print(result.to_json())
    else:
        print(result.to_json())


if __name__ == "__main__":
    main()
