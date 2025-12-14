from __future__ import annotations

import torch

import pytest

from torch_ivf.index import IndexBase


class DummyIndex(IndexBase):
    def __init__(self, d: int, **kwargs):
        super().__init__(d, **kwargs)
        self.buffer = torch.zeros((0, d), dtype=self.dtype, device=self.device)

    def _tensor_attributes(self):
        return ("buffer",)

    def train(self, xb: torch.Tensor) -> None:
        self._validate_input(xb)
        self._is_trained = True

    def add(self, xb: torch.Tensor) -> None:
        xb = self._validate_input(xb)
        self.buffer = torch.cat([self.buffer, xb], dim=0)
        self._ntotal = self.buffer.shape[0]

    def add_with_ids(self, xb: torch.Tensor, ids: torch.Tensor) -> None:
        _ = ids  # IDs unused for dummy implementation
        self.add(xb)

    def search(self, xq: torch.Tensor, k: int):
        xb = self._validate_input(xq)
        scores = torch.zeros((xb.shape[0], k), device=self.device, dtype=self.dtype)
        labels = torch.full((xb.shape[0], k), -1, device=self.device, dtype=torch.long)
        return scores, labels

    def range_search(self, xq: torch.Tensor, radius: float):
        _ = self._validate_input(xq)
        return ()

    def reset(self) -> None:
        self.buffer = torch.zeros((0, self.d), dtype=self.dtype, device=self.device)
        self._ntotal = 0
        self._is_trained = False


def test_index_base_properties_and_add():
    index = DummyIndex(d=8)
    assert index.d == 8
    assert index.metric == "l2"
    index.train(torch.zeros((4, 8)))
    assert index.is_trained is True
    index.add(torch.ones((2, 8)))
    assert index.ntotal == 2


def test_device_transfer_returns_copy():
    index = DummyIndex(d=4)
    clone = index.to("cpu")
    assert clone is not index
    assert clone.device.type == "cpu"
    assert clone.buffer.device.type == "cpu"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_transfer_when_available():
    index = DummyIndex(d=4)
    cuda_index = index.cuda()
    assert cuda_index.device.type == "cuda"
    assert cuda_index.buffer.device.type == "cuda"


def test_validate_input_checks_dimension():
    index = DummyIndex(d=4)
    with pytest.raises(ValueError):
        index.add(torch.randn(3, 5))


def test_rocm_raises_without_support(monkeypatch):
    index = DummyIndex(d=4)
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    with pytest.raises(RuntimeError):
        index.rocm()
