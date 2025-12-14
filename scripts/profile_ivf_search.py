"""IVFFlat search profiling helper (ROCm/CUDA).

This script prints a short profiler table to help confirm that the CSR path
avoids random-gather heavy ops (e.g., aten::index_select / aten::index).

Usage examples:
  uv run python scripts/profile_ivf_search.py --device cuda --search-mode csr
  uv run python scripts/profile_ivf_search.py --device cuda --search-mode matrix
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from torch_ivf.index import IndexIVFFlat


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Profile torch-ivf IndexIVFFlat.search.")
    p.add_argument("--device", default="cuda", help="torch device string (default: cuda)")
    p.add_argument("--search-mode", choices=["matrix", "csr", "auto"], default="csr")
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--nb", type=int, default=262144)
    p.add_argument("--train-n", type=int, default=20480)
    p.add_argument("--nq", type=int, default=19600)
    p.add_argument("--nlist", type=int, default=512)
    p.add_argument("--nprobe", type=int, default=32)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--max-codes", type=int, default=0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--warmup", type=int, default=1)
    return p.parse_args()


def _activities_for_device(device: torch.device) -> list[torch.profiler.ProfilerActivity]:
    if device.type == "cuda":
        return [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    return [torch.profiler.ProfilerActivity.CPU]


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    rng = np.random.default_rng(args.seed)
    xb_np = rng.standard_normal((args.nb, args.dim), dtype=np.float32)
    xq_np = rng.standard_normal((args.nq, args.dim), dtype=np.float32)
    train_n = max(1, min(args.nb, int(args.train_n)))

    xb = torch.from_numpy(xb_np).to(device)
    xq = torch.from_numpy(xq_np).to(device)
    train_x = xb[:train_n]

    index = IndexIVFFlat(
        args.dim,
        metric="l2",
        nlist=args.nlist,
        nprobe=args.nprobe,
        device=device,
        dtype=torch.float32,
    )
    index.search_mode = args.search_mode
    index.max_codes = int(args.max_codes)

    index.train(train_x, generator=torch.Generator(device="cpu").manual_seed(args.seed + 1))
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    index.add(xb)
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    for _ in range(max(0, int(args.warmup))):
        index.search(xq, args.topk)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.profiler.profile(
        activities=_activities_for_device(device),
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        index.search(xq, args.topk)
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    sort_by = "self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total"
    print(prof.key_averages().table(sort_by=sort_by, row_limit=30))


if __name__ == "__main__":
    main()

