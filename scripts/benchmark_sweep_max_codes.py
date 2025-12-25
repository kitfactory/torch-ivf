"""max_codes のスイープベンチ（torch-ivf GPU / faiss-cpu）。

Benchmark sweep over max_codes for torch-ivf (GPU) and faiss-cpu.
PatchCore/SPADE のような用途で、Faiss互換 `max_codes` の速度/精度トレードオフを把握する。
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

from torch_ivf.index import IndexIVFFlat, SearchParams


@dataclass
class SweepResult:
    library: str
    device: str
    device_name: str
    backend: str
    search_mode: str
    chosen_mode: str
    auto_avg_group_size: float | None
    auto_threshold: float | None
    auto_search_avg_group_threshold: float | None
    auto_enabled: int | None
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
    recall_at_k_vs_unlimited: float
    timestamp: str
    host_os: str
    host_cpu: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep max_codes for torch-ivf GPU and faiss-cpu.")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--nb", type=int, default=262144)
    p.add_argument("--train-n", type=int, default=20480)
    p.add_argument("--nq", type=int, default=19600)
    p.add_argument("--nlist", type=int, default=512)
    p.add_argument("--nprobe", type=int, default=32)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--metric", choices=["l2", "ip"], default="l2")
    p.add_argument("--dtype", choices=["float32"], default="float32")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument(
        "--max-codes-list",
        default="0,16384,32768,65536,131072",
        help="comma separated max_codes values (0=unlimited)",
    )
    p.add_argument("--torch-device", default="cuda", help="torch device string (default: cuda)")
    p.add_argument("--torch-search-mode", choices=["matrix", "csr", "auto"], default="matrix")
    p.add_argument("--jsonl", default="benchmarks/benchmarks.jsonl", help="append results to this JSONL file")
    p.add_argument("--json", action="store_true", help="print JSON only (still appends to --jsonl)")
    p.add_argument("--skip-faiss", action="store_true", help="skip faiss-cpu benchmark")
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


def _recall_at_k_vs_unlimited(base_I: np.ndarray, test_I: np.ndarray) -> float:
    if base_I.size == 0:
        return 0.0
    base = base_I.astype(np.int64, copy=False)
    test = test_I.astype(np.int64, copy=False)
    base_mask = base != -1
    matches = (base[:, :, None] == test[:, None, :]).any(axis=2)
    hits = matches & base_mask
    denom = base_mask.sum(axis=1)
    denom = np.maximum(denom, 1)
    per_q = hits.sum(axis=1) / denom
    return float(per_q.mean())


def _parse_max_codes_list(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    if not out:
        out = [0]
    return out


def _time_torch_search(
    index: IndexIVFFlat,
    xq: torch.Tensor,
    k: int,
    *,
    warmup: int,
    repeat: int,
    params: SearchParams | None = None,
) -> tuple[float, float]:
    device = xq.device
    for _ in range(warmup):
        if params is None:
            index.search(xq, k)
        else:
            index.search(xq, k, params=params)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    times_ms: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        if params is None:
            index.search(xq, k)
        else:
            index.search(xq, k, params=params)
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
    max_codes_list = _parse_max_codes_list(args.max_codes_list)
    warmup = max(0, int(args.warmup))
    repeat = max(1, int(args.repeat))

    rng = np.random.default_rng(args.seed)
    base_np = rng.standard_normal((args.nb, args.dim), dtype=np.float32)
    queries_np = rng.standard_normal((args.nq, args.dim), dtype=np.float32)
    train_n = max(1, min(args.nb, int(args.train_n)))
    train_np = base_np[:train_n]

    # Build torch-ivf once (max_codes affects only search).
    torch_device = torch.device(args.torch_device)
    xb = torch.from_numpy(base_np).to(torch_device)
    xq = torch.from_numpy(queries_np).to(torch_device)
    train_x = xb[:train_n]
    torch_index = IndexIVFFlat(
        args.dim,
        metric=args.metric,
        nlist=args.nlist,
        nprobe=args.nprobe,
        device=torch_device,
        dtype=torch.float32,
    )
    torch_index.search_mode = args.torch_search_mode
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

    torch_index.max_codes = 0
    D0_torch, I0_torch = torch_index.search(xq, args.topk)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    I0_torch_np = I0_torch.detach().to("cpu").numpy().astype(np.int64, copy=False)

    # Build faiss-cpu once.
    faiss_index = None
    faiss_train_ms = 0.0
    faiss_add_ms = 0.0
    I0_faiss = None
    if not args.skip_faiss:
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

        faiss_index.max_codes = 0
        _, I0_faiss = faiss_index.search(queries_np, args.topk)
        I0_faiss = I0_faiss.astype(np.int64, copy=False)

    records: list[SweepResult] = []
    now = datetime.now().isoformat(timespec="seconds")
    host_os = f"{platform.system()} {platform.release()}"
    host_cpu = platform.processor() or "unknown"

    for max_codes in max_codes_list:
        torch_index.max_codes = int(max_codes)
        params_perf = SearchParams(
            profile="speed",
            approximate=torch_index.approximate_mode,
            nprobe=torch_index.nprobe,
            max_codes=torch_index.max_codes,
            debug_stats=False,
        )
        search_ms, search_ms_min = _time_torch_search(
            torch_index, xq, args.topk, warmup=warmup, repeat=repeat, params=params_perf
        )
        D_t, I_t = torch_index.search(xq, args.topk, params=params_perf)
        if torch_device.type == "cuda":
            torch.cuda.synchronize(torch_device)
        I_t_np = I_t.detach().to("cpu").numpy().astype(np.int64, copy=False)
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
        recall_t = _recall_at_k_vs_unlimited(I0_torch_np, I_t_np)
        qps_t = args.nq / (search_ms / 1000) if search_ms > 0 else float("inf")
        records.append(
            SweepResult(
                library="torch_ivf",
                device=str(torch_device),
                device_name=_device_name(torch_device),
                backend=_detect_backend(torch_device),
                search_mode=args.torch_search_mode,
                chosen_mode=chosen_mode,
                auto_avg_group_size=stats.get("auto_avg_group_size"),
                auto_threshold=stats.get("auto_threshold"),
                auto_search_avg_group_threshold=stats.get("auto_search_avg_group_threshold"),
                auto_enabled=stats.get("auto_enabled"),
                metric=args.metric,
                dim=args.dim,
                nb=args.nb,
                train_n=train_n,
                nq=args.nq,
                nlist=args.nlist,
                nprobe=args.nprobe,
                max_codes=int(max_codes),
                topk=args.topk,
                dtype=args.dtype,
                warmup=warmup,
                repeat=repeat,
                train_ms=round(torch_train_ms, 3),
                add_ms=round(torch_add_ms, 3),
                search_ms=round(search_ms, 3),
                search_ms_min=round(search_ms_min, 3),
                qps=round(qps_t, 3),
                recall_at_k_vs_unlimited=round(recall_t, 6),
                timestamp=now,
                host_os=host_os,
                host_cpu=host_cpu,
            )
        )

        if faiss_index is not None and I0_faiss is not None:
            faiss_index.max_codes = int(max_codes)
            search_ms, search_ms_min = _time_faiss_search(faiss_index, queries_np, args.topk, warmup=warmup, repeat=repeat)
            _, I_f = faiss_index.search(queries_np, args.topk)
            I_f = I_f.astype(np.int64, copy=False)
            recall_f = _recall_at_k_vs_unlimited(I0_faiss, I_f)
            qps_f = args.nq / (search_ms / 1000) if search_ms > 0 else float("inf")
            records.append(
                SweepResult(
                    library="faiss_cpu",
                    device="cpu",
                    device_name=platform.processor() or "CPU",
                    backend="faiss-cpu",
                    search_mode="faiss",
                    chosen_mode="faiss",
                    auto_avg_group_size=None,
                    auto_threshold=None,
                    auto_search_avg_group_threshold=None,
                    auto_enabled=None,
                    metric=args.metric,
                    dim=args.dim,
                    nb=args.nb,
                    train_n=train_n,
                    nq=args.nq,
                    nlist=args.nlist,
                    nprobe=args.nprobe,
                    max_codes=int(max_codes),
                    topk=args.topk,
                    dtype=args.dtype,
                    warmup=warmup,
                    repeat=repeat,
                    train_ms=round(faiss_train_ms, 3),
                    add_ms=round(faiss_add_ms, 3),
                    search_ms=round(search_ms, 3),
                    search_ms_min=round(search_ms_min, 3),
                    qps=round(qps_f, 3),
                    recall_at_k_vs_unlimited=round(recall_f, 6),
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
