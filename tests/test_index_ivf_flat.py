from __future__ import annotations

import torch

import pytest

from torch_ivf.index import IndexFlatL2, IndexIVFFlat


def _toy_data(d=16, nb=512, nq=8, seed=0):
    g = torch.Generator().manual_seed(seed)
    xb = torch.randn(nb, d, generator=g)
    xq = torch.randn(nq, d, generator=g)
    return xb, xq


def test_ivf_train_add_and_search_matches_flat_when_nprobe_full():
    d = 16
    xb, xq = _toy_data(d=d, nb=400, nq=6)
    index = IndexIVFFlat(d, nlist=16)
    index.train(xb)
    index.add(xb)
    index.nprobe = index.nlist

    D_ivf, I_ivf = index.search(xq, k=5)

    flat = IndexFlatL2(d)
    flat.add(xb)
    D_flat, I_flat = flat.search(xq, k=5)

    assert torch.allclose(D_ivf.cpu(), D_flat.cpu(), atol=1e-5)
    assert torch.equal(I_ivf.cpu(), I_flat.cpu())


def test_ivf_search_csr_matches_flat_when_nprobe_full():
    d = 16
    xb, xq = _toy_data(d=d, nb=400, nq=6, seed=2)
    index = IndexIVFFlat(d, nlist=16)
    index.train(xb)
    index.add(xb)
    index.nprobe = index.nlist
    index.search_mode = "csr"

    D_ivf, I_ivf = index.search(xq, k=5)

    flat = IndexFlatL2(d)
    flat.add(xb)
    D_flat, I_flat = flat.search(xq, k=5)

    assert torch.allclose(D_ivf.cpu(), D_flat.cpu(), atol=1e-5)
    assert torch.equal(I_ivf.cpu(), I_flat.cpu())


def test_ivf_search_auto_matches_flat_when_nprobe_full():
    d = 16
    xb, xq = _toy_data(d=d, nb=400, nq=6, seed=3)
    index = IndexIVFFlat(d, nlist=16)
    index.train(xb)
    index.add(xb)
    index.nprobe = index.nlist
    index.search_mode = "auto"

    D_ivf, I_ivf = index.search(xq, k=5)

    flat = IndexFlatL2(d)
    flat.add(xb)
    D_flat, I_flat = flat.search(xq, k=5)

    assert torch.allclose(D_ivf.cpu(), D_flat.cpu(), atol=1e-5)
    assert torch.equal(I_ivf.cpu(), I_flat.cpu())


def test_ivf_range_search_respects_radius():
    xb, xq = _toy_data(d=8, nb=200, nq=3, seed=1)
    index = IndexIVFFlat(8, nlist=8)
    index.train(xb)
    index.add(xb)
    lims, dvals, ids = index.range_search(xq, radius=5.0)
    assert lims.shape[0] == xq.shape[0] + 1
    assert dvals.shape[0] == ids.shape[0]
    assert torch.all(lims[1:] >= lims[:-1])
    assert lims[-1] > 0


def test_ivf_state_dict_roundtrip(tmp_path):
    xb, _ = _toy_data(nb=64)
    index = IndexIVFFlat(16, nlist=4)
    index.train(xb)
    index.add(xb)
    path = tmp_path / "ivf.pt"
    index.save(path.as_posix())

    loaded = IndexIVFFlat.load(path.as_posix())
    assert loaded.nlist == index.nlist
    assert loaded.nprobe == index.nprobe
    assert loaded.ntotal == index.ntotal


def test_ivf_add_without_train_raises():
    index = IndexIVFFlat(16, nlist=4)
    with pytest.raises(RuntimeError):
        index.add(torch.randn(4, 16))


def test_ivf_max_codes_prefix_boundary_cases():
    index = IndexIVFFlat(1, nlist=4)
    offsets = torch.tensor([0, 3, 7, 12, 12], dtype=torch.long, device=index.device)
    index._list_offsets = offsets
    index._list_offsets_cpu = offsets.to("cpu").tolist()

    top_lists = torch.tensor([[0, 1, 2]], dtype=torch.long, device=index.device)

    for max_codes, expected_lists, expected_probes in [
        (6, [0], [0]),
        (7, [0, 1], [0, 1]),
        (8, [0, 1], [0, 1]),
    ]:
        index.max_codes = max_codes

        tasks_q, tasks_l, _ = index._build_tasks_from_lists(top_lists)
        assert tasks_q.to("cpu").tolist() == [0] * len(expected_lists)
        assert tasks_l.to("cpu").tolist() == expected_lists

        tasks_q, tasks_l, tasks_p, _ = index._build_tasks_from_lists_with_probe(top_lists)
        assert tasks_q.to("cpu").tolist() == [0] * len(expected_lists)
        assert tasks_l.to("cpu").tolist() == expected_lists
        assert tasks_p.to("cpu").tolist() == expected_probes


def test_ivf_search_csr_matches_matrix_when_max_codes_unlimited():
    d = 16
    xb, xq = _toy_data(d=d, nb=400, nq=16, seed=4)
    index = IndexIVFFlat(d, nlist=16, nprobe=4)
    index.train(xb)
    index.add(xb)
    index.max_codes = 0

    index.search_mode = "matrix"
    D_matrix, I_matrix = index.search(xq, k=5)

    index.search_mode = "csr"
    D_csr, I_csr = index.search(xq, k=5)

    assert torch.allclose(D_matrix.cpu(), D_csr.cpu(), atol=1e-5)
    assert torch.equal(I_matrix.cpu(), I_csr.cpu())


def test_ivf_search_cache_invalidated_on_add_and_max_codes():
    xb, xq = _toy_data(d=8, nb=64, nq=4, seed=5)
    index = IndexIVFFlat(8, nlist=8)
    index.train(xb)
    index.add(xb)
    index.search_mode = "csr"

    index.max_codes = 10
    index.search(xq, k=3)
    assert index._effective_max_codes_cache is not None

    index.max_codes = 5
    assert index._effective_max_codes_cache is None

    index.search(xq, k=3)
    assert index._list_sizes is not None

    index.add(xb[:8])
    assert index._list_sizes is None
    assert index._effective_max_codes_cache is None


def test_ivf_workspace_reuse_does_not_change_results():
    xb, xq = _toy_data(d=16, nb=128, nq=8, seed=6)
    index = IndexIVFFlat(16, nlist=32, nprobe=8)
    index.train(xb)
    index.add(xb)
    index.search_mode = "csr"

    d1, i1 = index.search(xq, k=5)
    caps = dict(index._workspace.capacity)
    d2, i2 = index.search(xq, k=5)

    assert torch.allclose(d1.cpu(), d2.cpu(), atol=1e-6)
    assert torch.equal(i1.cpu(), i2.cpu())
    for name, cap in caps.items():
        assert index._workspace.capacity[name] >= cap
