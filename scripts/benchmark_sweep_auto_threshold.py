from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import datetime

import numpy as np
import torch

from torch_ivf.index import IndexIVFFlat, SearchParams


@dataclass
class AutoThresholdResult:
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
    qps_recall: float
    timestamp: str
    host_os: str
    host_cpu: str

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sweep auto_search_avg_group_threshold for torch-ivf (GPU).")
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
    p.add_argument(
        "--nq-list",
        default="512,2048,19600",
        help="comma separated nq values",
    )
    p.add_argument(
        "--auto-thresholds",
        default="2,4,6,8,10,12,16",
        help="comma separated auto_search_avg_group_threshold values",
    )
    p.add_argument("--baseline-mode", choices=["matrix", "csr"], default="csr")
    p.add_argument("--torch-device", default="cuda", help="torch device string (default: cuda)")
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


def _parse_float_list(text: str) -> list[float]:
    out: list[float] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    return out


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


def _time_torch_search(
    index: IndexIVFFlat,
    xq: torch.Tensor,
    k: int,
    *,
    warmup: int,
    repeat: int,
    params: SearchParams,
) -> tuple[float, float]:
    device = xq.device
    for _ in range(warmup):
        index.search(xq, k, params=params)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    times_ms: list[float] = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        index.search(xq, k, params=params)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times_ms.append((time.perf_counter() - t0) * 1000)
    return float(statistics.median(times_ms)), float(min(times_ms))


def main() -> None:
    args = parse_args()
    nq_list = _parse_int_list(args.nq_list) or [1]
    auto_thresholds = _parse_float_list(args.auto_thresholds) or [8.0]

    warmup = max(0, int(args.warmup))
    repeat = max(1, int(args.repeat))

    rng = np.random.default_rng(args.seed)
    base_np = rng.standard_normal((args.nb, args.dim), dtype=np.float32)
    max_nq = max(nq_list)
    queries_np = rng.standard_normal((max_nq, args.dim), dtype=np.float32)
    train_n = max(1, min(args.nb, int(args.train_n)))

    torch_device = torch.device(args.torch_device)
    xb = torch.from_numpy(base_np).to(torch_device)
    xq_full = torch.from_numpy(queries_np).to(torch_device)
    train_x = xb[:train_n]

    index = IndexIVFFlat(
        args.dim,
        metric=args.metric,
        nlist=args.nlist,
        nprobe=args.nprobe,
        device=torch_device,
        dtype=torch.float32,
    )
    index.max_codes = int(args.max_codes)

    t0 = time.perf_counter()
    index.train(train_x, generator=torch.Generator(device="cpu").manual_seed(args.seed + 1))
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    t1 = time.perf_counter()
    index.add(xb)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    t2 = time.perf_counter()
    train_ms = (t1 - t0) * 1000
    add_ms = (t2 - t1) * 1000

    index.search_mode = args.baseline_mode
    base_params = SearchParams(
        profile="exact",
        approximate=False,
        nprobe=index.nprobe,
        max_codes=index.max_codes,
        debug_stats=False,
    )
    _, base_labels_full = index.search(xq_full, args.topk, params=base_params)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    base_labels_np = base_labels_full.detach().to("cpu").numpy().astype(np.int64, copy=False)

    index.search_mode = "auto"

    now = datetime.now().isoformat(timespec="seconds")
    host_os = f"{platform.system()} {platform.release()}"
    host_cpu = platform.processor() or "unknown"

    records: list[AutoThresholdResult] = []

    for threshold in auto_thresholds:
        index.auto_search_avg_group_threshold = float(threshold)
        for nq in nq_list:
            xq_t = xq_full[:nq]
            params_perf = SearchParams(
                profile="speed",
                approximate=index.approximate_mode,
                nprobe=index.nprobe,
                max_codes=index.max_codes,
                debug_stats=False,
            )
            search_ms, search_ms_min = _time_torch_search(
                index, xq_t, args.topk, warmup=warmup, repeat=repeat, params=params_perf
            )
            params_debug = SearchParams(
                profile="speed",
                approximate=index.approximate_mode,
                nprobe=index.nprobe,
                max_codes=index.max_codes,
                debug_stats=True,
            )
            _, labels = index.search(xq_t, args.topk, params=params_debug)
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
            labels_np = labels.detach().to("cpu").numpy().astype(np.int64, copy=False)
            recall = _recall_at_k_vs_unlimited(base_labels_np[:nq], labels_np)
            qps = nq / (search_ms / 1000) if search_ms > 0 else float("inf")
            stats = index.last_search_stats or {}
            chosen_mode = str(stats.get("chosen_mode", "auto"))
            records.append(
                AutoThresholdResult(
                    library="torch_ivf",
                    device=str(torch_device),
                    device_name=_device_name(torch_device),
                    backend=_detect_backend(torch_device),
                    search_mode="auto",
                    chosen_mode=chosen_mode,
                    auto_avg_group_size=stats.get("auto_avg_group_size"),
                    auto_threshold=stats.get("auto_threshold"),
                    auto_search_avg_group_threshold=float(threshold),
                    auto_enabled=stats.get("auto_enabled"),
                    metric=args.metric,
                    dim=args.dim,
                    nb=args.nb,
                    train_n=train_n,
                    nq=int(nq),
                    nlist=args.nlist,
                    nprobe=args.nprobe,
                    max_codes=int(args.max_codes),
                    topk=args.topk,
                    dtype=args.dtype,
                    warmup=warmup,
                    repeat=repeat,
                    train_ms=round(train_ms, 3),
                    add_ms=round(add_ms, 3),
                    search_ms=round(search_ms, 3),
                    search_ms_min=round(search_ms_min, 3),
                    qps=round(qps, 3),
                    recall_at_k_vs_unlimited=round(recall, 6),
                    qps_recall=round(qps * recall, 6),
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
