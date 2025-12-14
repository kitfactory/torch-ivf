"""Minimal IVF demo showing train/add/search flow."""

from __future__ import annotations

import argparse
from typing import Literal

import torch

from torch_ivf.index import IndexFlatIP, IndexFlatL2, IndexIVFFlat

Metric = Literal["l2", "ip"]


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="torch-ivf demo with synthetic data.")
    p.add_argument("--dim", type=int, default=64, help="Embedding dimension.")
    p.add_argument("--nb", type=int, default=4096, help="Number of base vectors.")
    p.add_argument("--nq", type=int, default=16, help="Number of query vectors.")
    p.add_argument("--nlist", type=int, default=64, help="Number of coarse centroids.")
    p.add_argument("--nprobe", type=int, default=8, help="Number of probed lists.")
    p.add_argument("--topk", type=int, default=10, help="K for nearest neighbor search.")
    p.add_argument("--add-batch-size", type=int, default=0, help="Split base vectors into this batch size when calling add() (0=all at once).")
    p.add_argument("--metric", choices=["l2", "ip"], default="l2")
    p.add_argument("--device", default=None, help="torch device string (e.g., cuda, cpu, dml).")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--verify", action="store_true", help="Compare against IndexFlat baseline.")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    device = torch.device(args.device) if args.device else torch.device("cpu")
    generator = torch.Generator(device=device).manual_seed(args.seed)

    nb = args.nb
    nq = args.nq
    dim = args.dim

    base = torch.randn(nb, dim, generator=generator, device=device)
    queries = torch.randn(nq, dim, generator=generator, device=device)
    train_x = base[: max(args.nlist * 2, args.nlist + 1)].contiguous()

    print(f"[demo] device={device}, nb={nb}, nq={nq}, dim={dim}")
    print(f"[demo] training k-means with nlist={args.nlist}")

    index = IndexIVFFlat(
        dim,
        metric=args.metric,
        nlist=args.nlist,
        nprobe=args.nprobe,
    ).to(device)
    index.train(train_x)
    add_bs = args.add_batch_size
    if add_bs and add_bs > 0:
        for chunk in torch.split(base, add_bs):
            index.add(chunk)
    else:
        index.add(base)

    print(f"[demo] index.ntotal={index.ntotal}, nprobe={index.nprobe}")
    distances, labels = index.search(queries, k=args.topk)
    print("[demo] first query distances:", distances[0].tolist())
    print("[demo] first query labels:", labels[0].tolist())

    if args.verify:
        flat = IndexFlatL2(dim) if args.metric == "l2" else IndexFlatIP(dim)
        flat.add(base.cpu())
        _, I_flat = flat.search(queries.cpu(), args.topk)
        recall = _recall_at_k(labels.cpu(), I_flat, args.topk)
        print(f"[demo] recall@{args.topk}: {recall:.3f}")


def _recall_at_k(pred: torch.Tensor, ref: torch.Tensor, k: int) -> float:
    """Compute recall@k between two label tensors."""
    if pred.shape != ref.shape:
        ref = ref[:, : pred.shape[1]]
    hits = 0
    total = pred.shape[0] * pred.shape[1]
    for q in range(pred.shape[0]):
        hits += len(set(pred[q].tolist()) & set(ref[q].tolist()))
    return hits / total if total else 0.0


if __name__ == "__main__":
    main()
