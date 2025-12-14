from __future__ import annotations

import torch

from torch_ivf.nn import kmeans


def _cluster_data(offset: float, n=64, dim=2, seed=0):
    g = torch.Generator().manual_seed(seed)
    center = torch.full((dim,), offset)
    return center + 0.05 * torch.randn(n, dim, generator=g)


def test_kmeans_two_clusters_converges():
    xb = torch.cat([_cluster_data(0.0), _cluster_data(5.0, seed=1)], dim=0)
    gen = torch.Generator().manual_seed(42)
    result = kmeans(xb, n_clusters=2, max_iter=50, batch_size=32, generator=gen, tol=1e-4)

    centers = result.centroids.cpu()
    centers = centers[centers[:, 0].argsort()]
    assert torch.allclose(centers[0], torch.zeros_like(centers[0]), atol=0.5)
    assert torch.allclose(centers[1], torch.full_like(centers[1], 5.0), atol=0.5)


def test_kmeans_deterministic_with_same_seed():
    xb = torch.cat([_cluster_data(0.0), _cluster_data(5.0, seed=2)], dim=0)
    gen1 = torch.Generator().manual_seed(0)
    gen2 = torch.Generator().manual_seed(0)
    r1 = kmeans(xb, n_clusters=2, generator=gen1)
    r2 = kmeans(xb, n_clusters=2, generator=gen2)

    assert torch.allclose(r1.centroids, r2.centroids, atol=1e-6)
    assert r1.iters == r2.iters


def test_kmeans_handles_ip_metric():
    xb = torch.cat([_cluster_data(0.0), _cluster_data(3.0, seed=3)], dim=0)
    result = kmeans(xb, n_clusters=2, metric="ip", batch_size=16)
    assert result.centroids.shape == (2, xb.shape[1])
