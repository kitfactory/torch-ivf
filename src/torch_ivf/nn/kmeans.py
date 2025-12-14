from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch

MetricType = Literal["l2", "ip"]


@dataclass(slots=True)
class KMeansResult:
    centroids: torch.Tensor
    iters: int
    inertia: float


def kmeans(
    xb: torch.Tensor,
    n_clusters: int,
    *,
    metric: MetricType = "l2",
    max_iter: int = 20,
    batch_size: Optional[int] = None,
    tol: float = 1e-3,
    generator: Optional[torch.Generator] = None,
    verbose: bool = False,
) -> KMeansResult:
    """Mini-batch k-means implemented in pure PyTorch."""
    if not isinstance(xb, torch.Tensor):
        raise TypeError("xb must be a torch.Tensor.")
    if xb.ndim != 2:
        raise ValueError("xb must be 2-D (num_samples, dim).")
    if xb.shape[0] < n_clusters:
        raise ValueError("Number of samples must be >= n_clusters.")
    if metric not in {"l2", "ip"}:
        raise ValueError("metric must be 'l2' or 'ip'.")

    xb = xb.contiguous()
    device = xb.device
    dtype = xb.dtype
    n, d = xb.shape
    batch = batch_size or min(4096, n)

    if generator is None:
        generator = torch.Generator()
        generator.manual_seed(torch.seed())

    cpu_generator = generator
    if getattr(generator, "device", torch.device("cpu")).type != "cpu":
        cpu_generator = torch.Generator(device="cpu")
        cpu_generator.manual_seed(generator.initial_seed())

    perm = torch.randperm(n, generator=cpu_generator)
    init_idx = perm[:n_clusters].to(device=device)
    centroids = xb[init_idx].clone()

    inertia = float("inf")
    for it in range(1, max_iter + 1):
        perm = torch.randperm(n, generator=cpu_generator)
        sums = torch.zeros((n_clusters, d), dtype=dtype, device=device)
        counts = torch.zeros(n_clusters, dtype=torch.long, device=device)
        total_loss = 0.0

        for start in range(0, n, batch):
            batch_idx = perm[start : start + batch].to(device=device)
            x = xb[batch_idx]

            scores = _pairwise_scores(x, centroids, metric)
            if metric == "l2":
                assign = torch.argmin(scores, dim=1)
                batch_loss = scores.gather(1, assign.unsqueeze(1)).sum()
            else:
                assign = torch.argmax(scores, dim=1)
                batch_loss = -scores.gather(1, assign.unsqueeze(1)).sum()

            sums.index_add_(0, assign, x)
            counts.index_add_(0, assign, torch.ones_like(assign, dtype=torch.long))
            total_loss += float(batch_loss)

        empty = counts == 0
        if empty.any():
            replace_idx = torch.randperm(n, generator=cpu_generator)[: empty.sum()].to(device=device)
            centroids[empty] = xb[replace_idx]
            counts[empty] = 1
            sums[empty] = centroids[empty]

        new_centroids = sums / counts.clamp(min=1).unsqueeze(1).to(dtype)
        shift = torch.norm(new_centroids - centroids).item()
        centroids = new_centroids
        inertia = total_loss / n

        if verbose:
            print(f"[kmeans] iter={it} shift={shift:.6f} inertia={inertia:.6f}")

        if shift <= tol:
            return KMeansResult(centroids=centroids, iters=it, inertia=inertia)

    return KMeansResult(centroids=centroids, iters=max_iter, inertia=inertia)


def _pairwise_scores(x: torch.Tensor, centroids: torch.Tensor, metric: MetricType) -> torch.Tensor:
    if metric == "l2":
        x_norm = (x * x).sum(dim=1, keepdim=True)
        c_norm = (centroids * centroids).sum(dim=1).unsqueeze(0)
        dist = x_norm + c_norm - (2.0 * (x @ centroids.T))
        return dist.clamp_min_(0)
    return x @ centroids.T
