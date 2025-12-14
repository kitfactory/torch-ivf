"""Faiss (CPU) benchmark for IVF-Flat."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime

import faiss
import numpy as np


@dataclass
class BenchmarkResult:
    library: str
    device: str
    device_name: str
    backend: str
    search_mode: str
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
    faiss_version: str
    numpy_version: str
    python_version: str
    host_os: str
    host_platform: str
    host_cpu: str
    faiss_omp_max_threads: int | None
    timestamp: str
    label: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="faiss-cpu IVF benchmark.")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--nb", type=int, default=65536)
    p.add_argument("--train-n", type=int, default=0, help="training sample count (0=auto)")
    p.add_argument("--nq", type=int, default=128)
    p.add_argument("--nlist", type=int, default=1024)
    p.add_argument("--nprobe", type=int, default=16)
    p.add_argument("--max-codes", type=int, default=0, help="cap scanned candidates per query (0=unlimited)")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--metric", choices=["l2", "ip"], default="l2")
    p.add_argument("--dtype", choices=["float32"], default="float32")
    p.add_argument("--warmup", type=int, default=1, help="warmup iterations for search timing")
    p.add_argument("--repeat", type=int, default=5, help="timed iterations for search timing")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--label", default="faiss_cpu")
    p.add_argument("--json", action="store_true")
    return p.parse_args()


def run_benchmark(args: argparse.Namespace) -> BenchmarkResult:
    rng = np.random.default_rng(args.seed)
    nb, nq, dim = args.nb, args.nq, args.dim

    base = rng.standard_normal((nb, dim), dtype=args.dtype)
    queries = rng.standard_normal((nq, dim), dtype=args.dtype)
    train_n = int(args.train_n) if args.train_n else max(args.nlist * 2, args.nlist + 1)
    train_n = max(1, min(nb, train_n))
    train_x = base[:train_n]

    metric = faiss.METRIC_L2 if args.metric == "l2" else faiss.METRIC_INNER_PRODUCT
    quantizer = faiss.IndexFlatL2(dim) if args.metric == "l2" else faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, args.nlist, metric)

    t0 = time.perf_counter()
    index.train(train_x)
    t1 = time.perf_counter()
    index.add(base)
    t2 = time.perf_counter()
    index.nprobe = args.nprobe
    index.max_codes = args.max_codes
    train_ms = (t1 - t0) * 1000
    add_ms = (t2 - t1) * 1000

    warmup = max(0, int(args.warmup))
    repeat = max(1, int(args.repeat))
    for _ in range(warmup):
        index.search(queries, args.topk)

    times_ms: list[float] = []
    for _ in range(repeat):
        s0 = time.perf_counter()
        index.search(queries, args.topk)
        times_ms.append((time.perf_counter() - s0) * 1000)

    search_ms = float(statistics.median(times_ms))
    search_ms_min = float(min(times_ms))
    qps = args.nq / (search_ms / 1000) if search_ms > 0 else float("inf")

    return BenchmarkResult(
        library="faiss_cpu",
        device="cpu",
        device_name=platform.processor() or "CPU",
        backend="faiss-cpu",
        search_mode="faiss",
        metric=args.metric,
        dim=dim,
        nb=nb,
        train_n=train_n,
        nq=nq,
        nlist=args.nlist,
        nprobe=args.nprobe,
        max_codes=args.max_codes,
        topk=args.topk,
        dtype=args.dtype,
        train_ms=round(train_ms, 3),
        add_ms=round(add_ms, 3),
        search_ms=round(search_ms, 3),
        search_ms_min=round(search_ms_min, 3),
        warmup=warmup,
        repeat=repeat,
        qps=round(qps, 3),
        faiss_version=faiss.__version__,
        numpy_version=np.__version__,
        python_version=platform.python_version(),
        host_os=f"{platform.system()} {platform.release()}",
        host_platform=platform.platform(),
        host_cpu=platform.processor() or "CPU",
        faiss_omp_max_threads=int(faiss.omp_get_max_threads()) if hasattr(faiss, "omp_get_max_threads") else None,
        timestamp=datetime.now().isoformat(timespec="seconds"),
        label=args.label,
    )


def main() -> None:
    args = parse_args()
    result = run_benchmark(args)
    print(result.to_json())


if __name__ == "__main__":
    main()
