from __future__ import annotations

import numpy as np
import torch

import faiss
import pytest

from torch_ivf.index import IndexFlatIP, IndexFlatL2


def _generate_data(d=16, nb=256, nq=4, *, seed=0):
    g = torch.Generator().manual_seed(seed)
    xb = torch.randn(nb, d, generator=g)
    xq = torch.randn(nq, d, generator=g)
    return xb, xq


def _as_numpy(x: torch.Tensor) -> np.ndarray:
    return x.cpu().numpy().astype("float32")


@pytest.mark.parametrize("k", [1, 5])
def test_index_flat_l2_matches_faiss(k):
    d = 16
    xb, xq = _generate_data(d=d, nb=128, nq=8)

    torch_index = IndexFlatL2(d)
    torch_index.add(xb)
    Dt, It = torch_index.search(xq, k)

    faiss_index = faiss.IndexFlatL2(d)
    faiss_index.add(_as_numpy(xb))
    Df, If = faiss_index.search(_as_numpy(xq), k)

    assert torch.allclose(Dt.cpu(), torch.from_numpy(Df), atol=1e-5)
    assert torch.equal(It.cpu(), torch.from_numpy(If))


def test_index_flat_ip_matches_faiss():
    d = 32
    xb, xq = _generate_data(d=d, nb=200, nq=5)

    torch_index = IndexFlatIP(d)
    torch_index.add(xb)
    Dt, It = torch_index.search(xq, k=10)

    faiss_index = faiss.IndexFlatIP(d)
    faiss_index.add(_as_numpy(xb))
    Df, If = faiss_index.search(_as_numpy(xq), 10)

    assert torch.allclose(Dt.cpu(), torch.from_numpy(Df), atol=1e-5)
    assert torch.equal(It.cpu(), torch.from_numpy(If))


def test_add_with_ids_and_range_search_l2():
    d = 8
    xb, xq = _generate_data(d=d, nb=50, nq=2)
    custom_ids = torch.arange(100, 150, dtype=torch.long)

    torch_index = IndexFlatL2(d)
    torch_index.add_with_ids(xb, custom_ids)

    lims, Dvals, Lvals = torch_index.range_search(xq, radius=5.0)
    faiss_index = faiss.IndexFlatL2(d)
    faiss_index.add(_as_numpy(xb))
    lims_ref, D_ref, L_ref = faiss_index.range_search(_as_numpy(xq), 5.0)

    lims_ref_t = torch.from_numpy(lims_ref.astype("int64"))
    assert torch.equal(lims.cpu(), lims_ref_t)
    assert torch.allclose(Dvals.cpu(), torch.from_numpy(D_ref), atol=1e-5)
    # Faiss returns base indices; convert to provided IDs.
    base_idx = torch.from_numpy(L_ref.astype("int64"))
    faiss_ids = custom_ids[base_idx]
    assert torch.equal(Lvals.cpu(), faiss_ids)


def test_reset_clears_state():
    idx = IndexFlatIP(4)
    idx.add(torch.randn(10, 4))
    assert idx.ntotal == 10
    idx.reset()
    assert idx.ntotal == 0
    D, I = idx.search(torch.randn(2, 4), k=3)
    assert torch.isinf(D).all() or torch.isneginf(D).all()
    assert (I == -1).all()


def test_index_flat_chunked_l2_matches_faiss():
    d = 16
    xb, xq = _generate_data(d=d, nb=500, nq=7, seed=4)
    torch_index = IndexFlatL2(d)
    torch_index.search_mode = "chunked"
    torch_index.add(xb)
    Dt, It = torch_index.search(xq, k=5)

    faiss_index = faiss.IndexFlatL2(d)
    faiss_index.add(_as_numpy(xb))
    Df, If = faiss_index.search(_as_numpy(xq), 5)

    assert torch.allclose(Dt.cpu(), torch.from_numpy(Df), atol=1e-5)
    assert torch.equal(It.cpu(), torch.from_numpy(If))


def test_index_flat_auto_l2_matches_faiss_when_forced_chunked():
    d = 16
    xb, xq = _generate_data(d=d, nb=500, nq=7, seed=5)
    torch_index = IndexFlatL2(d)
    torch_index.search_mode = "auto"
    torch_index.auto_matrix_bytes_threshold = 1
    torch_index.add(xb)
    Dt, It = torch_index.search(xq, k=5)

    faiss_index = faiss.IndexFlatL2(d)
    faiss_index.add(_as_numpy(xb))
    Df, If = faiss_index.search(_as_numpy(xq), 5)

    assert torch.allclose(Dt.cpu(), torch.from_numpy(Df), atol=1e-5)
    assert torch.equal(It.cpu(), torch.from_numpy(If))


def test_index_flat_chunked_ip_matches_faiss():
    d = 32
    xb, xq = _generate_data(d=d, nb=600, nq=6, seed=6)
    torch_index = IndexFlatIP(d)
    torch_index.search_mode = "chunked"
    torch_index.add(xb)
    Dt, It = torch_index.search(xq, k=10)

    faiss_index = faiss.IndexFlatIP(d)
    faiss_index.add(_as_numpy(xb))
    Df, If = faiss_index.search(_as_numpy(xq), 10)

    assert torch.allclose(Dt.cpu(), torch.from_numpy(Df), atol=1e-5)
    assert torch.equal(It.cpu(), torch.from_numpy(If))
