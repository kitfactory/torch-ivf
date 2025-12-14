"""nq のスイープベンチ（torch-ivf GPU / faiss-cpu）。

Benchmark sweep over nq (number of queries) for torch-ivf (GPU) and faiss-cpu.

目的:
- 小バッチ（nq が小さい）での “kernel launch 支配” を見える化する
- `search_mode=matrix` と `search_mode=csr` の境界（どこから伸び始めるか）を把握する
"""

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
import torch

from torch_ivf.index import IndexIVFFlat


@dataclass
class NQSweepResult:
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
    warmup: int
    repeat: int
    train_ms: float
    add_ms: float
    search_ms: float
    search_ms_min: float
    qps: float
    timestamp: str
    host_os: str
    host_cpu: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep nq for torch-ivf GPU and faiss-cpu.")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--nb", type=int, default=262144)
    p.add_argument("--train-n", type=int, default=20480)
    p.add_argument("--nlist", type=int, default=512)
    p.add_argument("--nprobe", type=int, default=32)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--metric", choices=["l2", "ip"], default="l2")
    p.add_argument("--dtype", choices=["float32"], default="float32")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--max-codes", type=int, default=0)
    p.add_argument("--nq-list", default="1,8,32,128,512,2048,19600", help="comma separated nq values")
    p.add_argument("--torch-device", default="cuda", help="torch device string (default: cuda)")
    p.add_argument("--torch-search-modes", default="matrix,csr", help="comma separated torch search_mode values (matrix, csr, auto)")
    p.add_argument("--jsonl", default="benchmarks/benchmarks.jsonl", help="append results to this JSONL file")
    p.add_argument("--json", action="store_true", help="print JSON only (still appends to --jsonl)")
    return p.parse_args()


def _device_name(device: torch.device) -> str:
    if device.type == "cuda" and torch.cuda.is_available():
        return torch.cuda.get_device_name(device)
    if device.type == "cpu":
        return platform.processor() or "CPU"
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
    return device.type


def _parse_int_list(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _time_torch_search(
    index: IndexIVFFlat, xq: torch.Tensor, k: int, *, warmup: int, repeat: int
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
    nq_list = _parse_int_list(args.nq_list) or [1]
    search_modes = [s.strip() for s in args.torch_search_modes.split(",") if s.strip()]
    if not search_modes:
        search_modes = ["matrix", "csr"]

    warmup = max(0, int(args.warmup))
    repeat = max(1, int(args.repeat))

    rng = np.random.default_rng(args.seed)
    base_np = rng.standard_normal((args.nb, args.dim), dtype=np.float32)
    max_nq = max(nq_list)
    queries_np = rng.standard_normal((max_nq, args.dim), dtype=np.float32)
    train_n = max(1, min(args.nb, int(args.train_n)))
    train_np = base_np[:train_n]

    torch_device = torch.device(args.torch_device)
    xb = torch.from_numpy(base_np).to(torch_device)
    xq_full = torch.from_numpy(queries_np).to(torch_device)
    train_x = xb[:train_n]

    torch_index = IndexIVFFlat(
        args.dim,
        metric=args.metric,
        nlist=args.nlist,
        nprobe=args.nprobe,
        device=torch_device,
        dtype=torch.float32,
    )
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

    metric = faiss.METRIC_L2 if args.metric == "l2" else faiss.METRIC_INNER_PRODUCT
    quantizer = faiss.IndexFlatL2(args.dim) if args.metric == "l2" else faiss.IndexFlatIP(args.dim)
    faiss_index = faiss.IndexIVFFlat(quantizer, args.dim, args.nlist, metric)
    t0 = time.perf_counter()
    faiss_index.train(train_np)
    t1 = time.perf_counter()
    faiss_index.add(base_np)
    t2 = time.perf_counter()
    faiss_train_ms = (t1 - t0) * 1000
    faiss_add_ms = (t2 - t1) * 1000
    faiss_index.nprobe = int(args.nprobe)
    faiss_index.max_codes = int(args.max_codes)

    now = datetime.now().isoformat(timespec="seconds")
    host_os = f"{platform.system()} {platform.release()}"
    host_cpu = platform.processor() or "unknown"

    records: list[NQSweepResult] = []

    for nq in nq_list:
        nq = int(nq)
        xq_t = xq_full[:nq]
        xq_f = queries_np[:nq]

        for mode in search_modes:
            torch_index.search_mode = mode
            search_ms, search_ms_min = _time_torch_search(torch_index, xq_t, args.topk, warmup=warmup, repeat=repeat)
            qps = nq / (search_ms / 1000) if search_ms > 0 else float("inf")
            records.append(
                NQSweepResult(
                    library="torch_ivf",
                    device=str(torch_device),
                    device_name=_device_name(torch_device),
                    backend=_detect_backend(torch_device),
                    search_mode=mode,
                    metric=args.metric,
                    dim=args.dim,
                    nb=args.nb,
                    train_n=train_n,
                    nq=nq,
                    nlist=args.nlist,
                    nprobe=args.nprobe,
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
                    timestamp=now,
                    host_os=host_os,
                    host_cpu=host_cpu,
                )
            )

        search_ms, search_ms_min = _time_faiss_search(faiss_index, xq_f, args.topk, warmup=warmup, repeat=repeat)
        qps = nq / (search_ms / 1000) if search_ms > 0 else float("inf")
        records.append(
            NQSweepResult(
                library="faiss_cpu",
                device="cpu",
                device_name=platform.processor() or "CPU",
                backend="faiss-cpu",
                search_mode="faiss",
                metric=args.metric,
                dim=args.dim,
                nb=args.nb,
                train_n=train_n,
                nq=nq,
                nlist=args.nlist,
                nprobe=args.nprobe,
                max_codes=int(args.max_codes),
                topk=args.topk,
                dtype=args.dtype,
                warmup=warmup,
                repeat=repeat,
                train_ms=round(faiss_train_ms, 3),
                add_ms=round(faiss_add_ms, 3),
                search_ms=round(search_ms, 3),
                search_ms_min=round(search_ms_min, 3),
                qps=round(qps, 3),
                timestamp=now,
                host_os=host_os,
                host_cpu=host_cpu,
            )
        )

    with open(args.jsonl, "a", encoding="utf-8") as f:
        for r in records:
            f.write(r.to_json() + "\n")

    if args.json:
        for r in records:
            print(r.to_json())
    else:
        for r in records:
            print(r.to_json())


if __name__ == "__main__":
    main()
