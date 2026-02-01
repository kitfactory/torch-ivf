"""(nlist, nprobe) sweep benchmark (torch-ivf GPU / faiss-cpu).

目的 / Purpose:
- throughput 領域（例: nq=19600）で torch-ivf と faiss-cpu の比率を最大化する設定を探す
- 同一データ（同一 base/query/train）で両ライブラリを測る

Usage:
  uv run python scripts/benchmark_sweep_ivf_params.py --torch-device cuda --pairs 512:32,256:16,128:8,1024:64
  uv run python scripts/benchmark_sweep_ivf_params.py --torch-device cuda --dtype float16 --pairs 512:32

Real data (.npy):
  uv run python scripts/benchmark_sweep_ivf_params.py --torch-device cuda --dtype float16 --pairs 512:32 ^
    --base-npy path\\to\\base.npy --query-npy path\\to\\query.npy --dataset mydata
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime

import faiss
import numpy as np
import torch

from torch_ivf.index import IndexIVFFlat, SearchParams


@dataclass
class ParamSweepResult:
    library: str
    device: str
    device_name: str
    backend: str
    dataset: str
    search_mode: str
    chosen_mode: str
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
    warmup: int
    repeat: int
    train_ms: float
    add_ms: float
    search_ms: float
    search_ms_min: float
    qps: float
    # Optional debug stats (torch only).
    search_stats: dict[str, float | int | str] | None
    timestamp: str
    host_os: str
    host_cpu: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep (nlist, nprobe) for torch-ivf GPU and faiss-cpu.")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--nb", type=int, default=262144)
    p.add_argument("--train-n", type=int, default=20480)
    p.add_argument("--nq", type=int, default=19600)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--metric", choices=["l2", "ip"], default="l2")
    # NOTE: faiss-cpu uses float32 in Python API. This flag refers to torch dtype.
    p.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--max-codes", type=int, default=0)
    p.add_argument("--pairs", default="512:32,256:16,128:8,1024:64", help="comma separated nlist:nprobe pairs")
    p.add_argument("--torch-device", default="cuda", help="torch device string (default: cuda)")
    p.add_argument("--torch-search-mode", choices=["csr", "matrix", "auto"], default="csr")
    p.add_argument("--dataset", default="synthetic", help="label stored in JSONL (no file paths are recorded)")
    p.add_argument("--base-npy", default=None, help="real data base vectors (.npy, shape [nb, dim])")
    p.add_argument("--query-npy", default=None, help="real data query vectors (.npy, shape [nq, dim])")
    p.add_argument("--train-npy", default=None, help="optional training vectors (.npy, shape [train_n, dim])")
    p.add_argument("--jsonl", default="benchmarks/benchmarks.jsonl", help="append results to this JSONL file")
    p.add_argument("--skip-faiss", action="store_true", help="skip faiss-cpu benchmark")
    p.add_argument(
        "--faiss-threads",
        type=int,
        default=0,
        help="faiss OMP threads (0=leave default, -1=os.cpu_count())",
    )
    p.add_argument("--json", action="store_true", help="print JSON only (still appends to --jsonl)")
    return p.parse_args()


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


def _parse_pairs(text: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        nlist_s, nprobe_s = part.split(":")
        out.append((int(nlist_s), int(nprobe_s)))
    return out


def _load_npy(path: str) -> np.ndarray:
    # Use mmap for large datasets; slices are materialized as needed.
    arr = np.load(path, mmap_mode="r")
    if not isinstance(arr, np.ndarray):
        raise TypeError(f"expected .npy array, got {type(arr)}")
    if arr.ndim != 2:
        raise ValueError(f"expected 2D array [n, d], got shape={arr.shape}")
    return arr


def _time_torch_search(
    index: IndexIVFFlat,
    xq: torch.Tensor,
    k: int,
    *,
    warmup: int,
    repeat: int,
) -> tuple[float, float]:
    device = xq.device
    for _ in range(warmup):
        index.search(xq, k)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    times_ms: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        index.search(xq, k)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times_ms.append((time.perf_counter() - t0) * 1000)
    return float(statistics.median(times_ms)), float(min(times_ms))


def _time_faiss_search(index, xq: np.ndarray, k: int, *, warmup: int, repeat: int) -> tuple[float, float]:
    for _ in range(warmup):
        index.search(xq, k)
    times_ms: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        index.search(xq, k)
        times_ms.append((time.perf_counter() - t0) * 1000)
    return float(statistics.median(times_ms)), float(min(times_ms))


def main() -> None:
    args = parse_args()
    torch_device = torch.device(args.torch_device)
    pairs = _parse_pairs(args.pairs)
    dataset = str(args.dataset)

    if not args.skip_faiss and int(args.faiss_threads) != 0:
        threads = (os.cpu_count() or 0) if int(args.faiss_threads) == -1 else int(args.faiss_threads)
        if threads and hasattr(faiss, "omp_set_num_threads"):
            faiss.omp_set_num_threads(int(threads))

    # Data: same base/query for both libraries.
    if args.base_npy and args.query_npy:
        base_all = _load_npy(args.base_npy)
        queries_all = _load_npy(args.query_npy)
        train_all = _load_npy(args.train_npy) if args.train_npy else None

        dim = int(base_all.shape[1])
        if int(queries_all.shape[1]) != dim:
            raise ValueError(f"dim mismatch: base dim={dim}, query dim={int(queries_all.shape[1])}")
        if train_all is not None and int(train_all.shape[1]) != dim:
            raise ValueError(f"dim mismatch: base dim={dim}, train dim={int(train_all.shape[1])}")
        if int(args.dim) != dim:
            raise ValueError(f"--dim={args.dim} but loaded data has dim={dim}")

        nb = int(base_all.shape[0]) if int(args.nb) <= 0 else min(int(args.nb), int(base_all.shape[0]))
        nq = int(queries_all.shape[0]) if int(args.nq) <= 0 else min(int(args.nq), int(queries_all.shape[0]))
        base_np = np.ascontiguousarray(base_all[:nb], dtype=np.float32)
        queries_np = np.ascontiguousarray(queries_all[:nq], dtype=np.float32)

        # Training pool size:
        # - fixed train-n (>0): use that many points for all pairs
        # - train-n=0: allocate an upper bound sufficient for the largest nlist in --pairs
        if int(args.train_n) > 0:
            train_pool_n = int(args.train_n)
        else:
            train_pool_n = max(max(nlist * 2, nlist + 1) for nlist, _ in pairs)

        if train_all is None:
            train_pool_n = min(train_pool_n, nb)
            train_pool_np = np.ascontiguousarray(base_all[:train_pool_n], dtype=np.float32)
        else:
            train_pool_n = min(train_pool_n, int(train_all.shape[0]))
            train_pool_np = np.ascontiguousarray(train_all[:train_pool_n], dtype=np.float32)

        # Update record fields to reflect actual slices used.
        args.nb = nb
        args.nq = nq
        args.dim = dim
    else:
        rng = np.random.default_rng(args.seed)
        base_np = rng.standard_normal((args.nb, args.dim), dtype=np.float32)
        queries_np = rng.standard_normal((args.nq, args.dim), dtype=np.float32)
        if int(args.train_n) > 0:
            train_pool_n = min(int(args.nb), int(args.train_n))
        else:
            train_pool_n = min(int(args.nb), max(max(nlist * 2, nlist + 1) for nlist, _ in pairs))
        train_pool_n = max(1, int(train_pool_n))
        train_pool_np = np.ascontiguousarray(base_np[:train_pool_n], dtype=np.float32)

    torch_dtype = getattr(torch, args.dtype)
    xb = torch.from_numpy(base_np).to(device=torch_device, dtype=torch_dtype)
    xq = torch.from_numpy(queries_np).to(device=torch_device, dtype=torch_dtype)
    train_x_pool = torch.from_numpy(train_pool_np).to(device=torch_device, dtype=torch_dtype)

    warmup = max(0, int(args.warmup))
    repeat = max(1, int(args.repeat))
    now = datetime.now().isoformat(timespec="seconds")
    host_os = f"{platform.system()} {platform.release()}"
    host_cpu = platform.processor() or "unknown"

    records: list[ParamSweepResult] = []

    for nlist, nprobe in pairs:
        train_n = int(args.train_n) if int(args.train_n) > 0 else max(nlist * 2, nlist + 1)
        train_n = max(1, min(int(train_pool_np.shape[0]), int(train_n)))
        train_x = train_x_pool[:train_n]

        # torch-ivf
        torch_index = IndexIVFFlat(
            args.dim,
            metric=args.metric,
            nlist=nlist,
            nprobe=nprobe,
            device=torch_device,
            dtype=torch_dtype,
        )
        torch_index.search_mode = args.torch_search_mode
        torch_index.max_codes = int(args.max_codes)

        t0 = time.perf_counter()
        torch_index.train(train_x, generator=torch.Generator(device="cpu").manual_seed(args.seed + 1))
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        t1 = time.perf_counter()
        torch_index.add(xb)
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        t2 = time.perf_counter()

        torch_train_ms = (t1 - t0) * 1000
        torch_add_ms = (t2 - t1) * 1000
        search_ms, search_ms_min = _time_torch_search(torch_index, xq, args.topk, warmup=warmup, repeat=repeat)

        params_debug = SearchParams(
            profile="speed",
            approximate=torch_index.approximate_mode,
            nprobe=torch_index.nprobe,
            max_codes=torch_index.max_codes,
            debug_stats=True,
        )
        torch_index.search(xq, args.topk, params=params_debug)
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        stats = torch_index.last_search_stats or {}
        chosen_mode = str(stats.get("chosen_mode", args.torch_search_mode))
        qps = args.nq / (search_ms / 1000) if search_ms > 0 else float("inf")
        records.append(
            ParamSweepResult(
                library="torch_ivf",
                device=str(torch_device),
                device_name=_device_name(torch_device),
                backend=_detect_backend(torch_device),
                dataset=dataset,
                search_mode=args.torch_search_mode,
                chosen_mode=chosen_mode,
                metric=args.metric,
                dim=args.dim,
                nb=args.nb,
                train_n=train_n,
                nq=args.nq,
                nlist=nlist,
                nprobe=nprobe,
                max_codes=int(args.max_codes),
                topk=args.topk,
                dtype=args.dtype,
                warmup=warmup,
                repeat=repeat,
                train_ms=round(torch_train_ms, 3),
                add_ms=round(torch_add_ms, 3),
                search_ms=round(search_ms, 3),
                search_ms_min=round(search_ms_min, 3),
                qps=round(qps, 3),
                search_stats=stats if stats else None,
                timestamp=now,
                host_os=host_os,
                host_cpu=host_cpu,
                )
            )

        if not args.skip_faiss:
            metric = faiss.METRIC_L2 if args.metric == "l2" else faiss.METRIC_INNER_PRODUCT
            quantizer = faiss.IndexFlatL2(args.dim) if args.metric == "l2" else faiss.IndexFlatIP(args.dim)
            faiss_index = faiss.IndexIVFFlat(quantizer, args.dim, nlist, metric)

            t0 = time.perf_counter()
            faiss_index.train(train_pool_np[:train_n])
            t1 = time.perf_counter()
            faiss_index.add(base_np)
            t2 = time.perf_counter()
            faiss_train_ms = (t1 - t0) * 1000
            faiss_add_ms = (t2 - t1) * 1000
            faiss_index.nprobe = int(nprobe)
            faiss_index.max_codes = int(args.max_codes)

            search_ms, search_ms_min = _time_faiss_search(
                faiss_index, queries_np, args.topk, warmup=warmup, repeat=repeat
            )
            qps = args.nq / (search_ms / 1000) if search_ms > 0 else float("inf")
            records.append(
                ParamSweepResult(
                    library="faiss_cpu",
                    device="cpu",
                    device_name=platform.processor() or "CPU",
                    backend="faiss-cpu",
                    dataset=dataset,
                    search_mode="faiss",
                    chosen_mode="faiss",
                    metric=args.metric,
                    dim=args.dim,
                    nb=args.nb,
                    train_n=train_n,
                    nq=args.nq,
                    nlist=nlist,
                    nprobe=nprobe,
                    max_codes=int(args.max_codes),
                    topk=args.topk,
                    dtype="float32",
                    warmup=warmup,
                    repeat=repeat,
                    train_ms=round(faiss_train_ms, 3),
                    add_ms=round(faiss_add_ms, 3),
                    search_ms=round(search_ms, 3),
                    search_ms_min=round(search_ms_min, 3),
                    qps=round(qps, 3),
                    search_stats=None,
                    timestamp=now,
                    host_os=host_os,
                    host_cpu=host_cpu,
                )
            )

    with open(args.jsonl, "a", encoding="utf-8") as f:
        for r in records:
            f.write(r.to_json() + "\n")

    for r in records:
        print(r.to_json())


if __name__ == "__main__":
    main()
