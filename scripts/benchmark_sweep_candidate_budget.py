"""candidate_budget のスイープベンチ（torch-ivf GPU）。

Benchmark sweep over candidate_budget for torch-ivf (GPU).
"""

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
class CandidateBudgetSweepResult:
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
    candidate_budget: int
    budget_strategy: str
    list_ordering: str | None
    dynamic_nprobe: bool
    min_codes_per_list: int
    max_codes_cap_per_list: int
    strict_budget: bool
    use_per_list_sizes: bool
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
    p = argparse.ArgumentParser(description="Sweep candidate_budget for torch-ivf GPU.")
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
    p.add_argument("--max-codes", type=int, default=0)
    p.add_argument(
        "--candidate-budgets",
        default="16384,32768,65536,131072",
        help="comma separated candidate_budget values",
    )
    p.add_argument(
        "--budget-strategy",
        choices=["uniform", "distance_weighted"],
        default="distance_weighted",
    )
    p.add_argument(
        "--list-ordering",
        choices=["residual_norm_asc", "none"],
        default="residual_norm_asc",
    )
    p.add_argument("--dynamic-nprobe", action="store_true")
    p.add_argument("--min-codes-per-list", type=int, default=0)
    p.add_argument("--max-codes-cap-per-list", type=int, default=0)
    p.add_argument("--strict-budget", action="store_true")
    p.add_argument(
        "--budget-mode",
        choices=["max_codes", "per_list"],
        default="max_codes",
        help="budget interpretation (max_codes=prefix total, per_list=per-list cap)",
    )
    p.add_argument("--torch-device", default="cuda", help="torch device string (default: cuda)")
    p.add_argument("--torch-search-mode", choices=["matrix", "csr", "auto"], default="csr")
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


def _parse_int_list(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value > 0:
            out.append(value)
    if not out:
        out = [16384, 32768, 65536, 131072]
    return out


def _time_torch_search(
    index: IndexIVFFlat, xq: torch.Tensor, k: int, *, warmup: int, repeat: int, params: SearchParams
) -> tuple[float, float, torch.Tensor]:
    device = xq.device
    for _ in range(warmup):
        index.search(xq, k, params=params)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    times_ms: list[float] = []
    out_labels = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        _, labels = index.search(xq, k, params=params)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times_ms.append((time.perf_counter() - t0) * 1000)
        out_labels = labels
    if out_labels is None:
        out_labels = torch.empty((xq.shape[0], k), dtype=torch.long, device=xq.device)
    return float(statistics.median(times_ms)), float(min(times_ms)), out_labels


def main() -> None:
    args = parse_args()
    candidate_budgets = _parse_int_list(args.candidate_budgets)
    warmup = max(0, int(args.warmup))
    repeat = max(1, int(args.repeat))

    rng = np.random.default_rng(args.seed)
    base_np = rng.standard_normal((args.nb, args.dim), dtype=np.float32)
    queries_np = rng.standard_normal((args.nq, args.dim), dtype=np.float32)
    train_n = max(1, min(args.nb, int(args.train_n)))

    torch_device = torch.device(args.torch_device)
    xb = torch.from_numpy(base_np).to(torch_device)
    xq = torch.from_numpy(queries_np).to(torch_device)
    train_x = xb[:train_n]

    index = IndexIVFFlat(
        args.dim,
        metric=args.metric,
        nlist=args.nlist,
        nprobe=args.nprobe,
        device=torch_device,
        dtype=torch.float32,
    )
    index.search_mode = args.torch_search_mode
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

    if args.list_ordering != "none":
        index.rebuild_lists(ordering=args.list_ordering)

    base_params = SearchParams(profile="exact", approximate=False)
    _, base_labels = index.search(xq, args.topk, params=base_params)
    if torch_device.type == "cuda":
        torch.cuda.synchronize(torch_device)
    base_labels_np = base_labels.detach().to("cpu").numpy().astype(np.int64, copy=False)

    now = datetime.now().isoformat(timespec="seconds")
    host_os = f"{platform.system()} {platform.release()}"
    host_cpu = platform.processor() or "unknown"

    records: list[CandidateBudgetSweepResult] = []
    list_ordering = None if args.list_ordering == "none" else args.list_ordering
    use_per_list_sizes = args.budget_mode == "per_list"
    if not use_per_list_sizes:
        if args.dynamic_nprobe or args.min_codes_per_list > 0 or args.max_codes_cap_per_list > 0 or args.strict_budget:
            raise ValueError("per-list options require --budget-mode per_list.")

    for budget in candidate_budgets:
        params = SearchParams(
            profile="approx",
            approximate=True,
            nprobe=args.nprobe,
            max_codes=args.max_codes,
            candidate_budget=budget,
            budget_strategy=args.budget_strategy,
            list_ordering=list_ordering,
            rebuild_policy="manual",
            dynamic_nprobe=args.dynamic_nprobe,
            min_codes_per_list=args.min_codes_per_list,
            max_codes_cap_per_list=args.max_codes_cap_per_list,
            strict_budget=args.strict_budget,
            use_per_list_sizes=use_per_list_sizes,
        )
        search_ms, search_ms_min, labels = _time_torch_search(
            index, xq, args.topk, warmup=warmup, repeat=repeat, params=params
        )
        labels_np = labels.detach().to("cpu").numpy().astype(np.int64, copy=False)
        recall = _recall_at_k_vs_unlimited(base_labels_np, labels_np)
        qps = args.nq / (search_ms / 1000) if search_ms > 0 else float("inf")
        records.append(
            CandidateBudgetSweepResult(
                library="torch_ivf",
                device=str(torch_device),
                device_name=_device_name(torch_device),
                backend=_detect_backend(torch_device),
                search_mode=args.torch_search_mode,
                metric=args.metric,
                dim=args.dim,
                nb=args.nb,
                train_n=train_n,
                nq=args.nq,
                nlist=args.nlist,
                nprobe=args.nprobe,
                max_codes=int(args.max_codes),
                candidate_budget=int(budget),
                budget_strategy=args.budget_strategy,
                list_ordering=list_ordering,
                dynamic_nprobe=bool(args.dynamic_nprobe),
                min_codes_per_list=int(args.min_codes_per_list),
                max_codes_cap_per_list=int(args.max_codes_cap_per_list),
                strict_budget=bool(args.strict_budget),
                use_per_list_sizes=bool(use_per_list_sizes),
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
