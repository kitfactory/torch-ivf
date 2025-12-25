from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


PRESET_WEIGHTS = {
    "latency": {
        8: 0.25,
        32: 0.25,
        64: 0.20,
        128: 0.15,
        256: 0.10,
        512: 0.05,
    },
    "throughput": {
        8: 0.05,
        32: 0.10,
        64: 0.10,
        128: 0.15,
        256: 0.20,
        512: 0.40,
    },
}


@dataclass
class CellStats:
    median: float
    count: int


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Score auto_search_avg_group_threshold from benchmarks JSONL.")
    p.add_argument("--jsonl", default="benchmarks/benchmarks.jsonl")
    p.add_argument("--preset", choices=sorted(PRESET_WEIGHTS), default="latency")
    p.add_argument(
        "--nq-list",
        default="8,32,64,128,256,512",
        help="comma separated nq values (default: latency/throughput preset list)",
    )
    p.add_argument(
        "--weights",
        default=None,
        help='override weights, e.g. "8:0.35,32:0.35,64:0.15,128:0.10,256:0.04,512:0.01"',
    )
    p.add_argument("--metric", default=None, help="optional metric filter (l2/ip)")
    p.add_argument(
        "--timestamp",
        default=None,
        help="optional timestamp filter (exact match, e.g. 2025-12-25T10:03:24)",
    )
    p.add_argument("--top", type=int, default=0, help="show top N thresholds (0=all)")
    return p.parse_args()


def _parse_int_list(text: str) -> list[int]:
    out: list[int] = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _parse_weight_map(text: str) -> dict[int, float]:
    weights: dict[int, float] = {}
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"invalid weight entry: {part}")
        nq_str, weight_str = part.split(":", 1)
        nq = int(nq_str.strip())
        weight = float(weight_str.strip())
        weights[nq] = weight
    return weights


def _normalize_weights(weights: dict[int, float], nq_list: list[int]) -> dict[int, float]:
    filtered = {nq: float(weights.get(nq, 0.0)) for nq in nq_list}
    total = sum(filtered.values())
    if total <= 0:
        raise ValueError("weights sum must be > 0")
    return {nq: w / total for nq, w in filtered.items()}


def _load_scores(
    jsonl_path: Path,
    nq_list: list[int],
    *,
    metric: str | None,
    timestamp: str | None,
) -> dict[float, dict[int, list[float]]]:
    grouped: dict[float, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for line in jsonl_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("library") != "torch_ivf":
            continue
        if row.get("search_mode") != "auto":
            continue
        threshold = row.get("auto_search_avg_group_threshold")
        if threshold is None:
            continue
        nq = row.get("nq")
        if nq not in nq_list:
            continue
        if metric is not None and row.get("metric") != metric:
            continue
        if timestamp is not None and row.get("timestamp") != timestamp:
            continue
        qps_recall = row.get("qps_recall")
        if qps_recall is None:
            qps = row.get("qps")
            recall = row.get("recall_at_k_vs_unlimited")
            if qps is None or recall is None:
                continue
            qps_recall = float(qps) * float(recall)
        grouped[float(threshold)][int(nq)].append(float(qps_recall))
    return grouped


def main() -> None:
    args = parse_args()
    jsonl_path = Path(args.jsonl)
    nq_list = _parse_int_list(args.nq_list)
    if not nq_list:
        raise ValueError("nq-list must not be empty")

    if args.weights is not None:
        raw_weights = _parse_weight_map(args.weights)
    else:
        raw_weights = PRESET_WEIGHTS[args.preset]
    weights = _normalize_weights(raw_weights, nq_list)

    grouped = _load_scores(jsonl_path, nq_list, metric=args.metric, timestamp=args.timestamp)
    if not grouped:
        raise ValueError("no matching rows found (check filters and jsonl contents)")

    missing_weights = [nq for nq in nq_list if raw_weights.get(nq, 0.0) <= 0.0]
    if missing_weights:
        print(f"warning: zero weight for nq={missing_weights}")

    print("weights (normalized):")
    for nq in nq_list:
        print(f"  nq={nq}: {weights[nq]:.4f}")

    scored: list[tuple[float, float, dict[int, CellStats]]] = []
    for threshold, by_nq in grouped.items():
        per_nq: dict[int, CellStats] = {}
        for nq in nq_list:
            values = by_nq.get(nq, [])
            if values:
                per_nq[nq] = CellStats(median=statistics.median(values), count=len(values))
        total = 0.0
        for nq in nq_list:
            stats = per_nq.get(nq)
            if stats is None:
                continue
            total += weights[nq] * stats.median
        scored.append((threshold, total, per_nq))

    scored.sort(key=lambda item: (-item[1], item[0]))

    print("")
    print("ranking (higher is better):")
    for idx, (threshold, total, _) in enumerate(scored, 1):
        print(f"{idx:2d}. threshold={threshold:g} score={total:.6f}")
        if args.top and idx >= args.top:
            break

    show_count = args.top if args.top and args.top > 0 else len(scored)
    for threshold, total, per_nq in scored[:show_count]:
        print("")
        print(f"details threshold={threshold:g} score={total:.6f}")
        for nq in nq_list:
            stats = per_nq.get(nq)
            if stats is None:
                print(f"  nq={nq}: missing (weight={weights[nq]:.4f})")
                continue
            contrib = weights[nq] * stats.median
            print(
                f"  nq={nq}: median={stats.median:.6f} count={stats.count} "
                f"weight={weights[nq]:.4f} contrib={contrib:.6f}"
            )


if __name__ == "__main__":
    main()
