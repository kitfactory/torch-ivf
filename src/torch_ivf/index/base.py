from __future__ import annotations

import copy
from abc import ABC, abstractmethod
from typing import Iterable, Literal, Sequence

import torch

MetricType = Literal["l2", "ip"]


class IndexBase(ABC):
    """Shared functionality for IVF/Flat indexes."""

    _VALID_METRICS: tuple[MetricType, ...] = ("l2", "ip")

    def __init__(
        self,
        d: int,
        *,
        metric: MetricType = "l2",
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        if d <= 0:
            raise ValueError("dimension `d` must be a positive integer.")
        if metric not in self._VALID_METRICS:
            raise ValueError(f"metric must be one of {self._VALID_METRICS}, got {metric!r}")

        self._d = int(d)
        self._metric: MetricType = metric
        self._dtype = dtype
        self._device = self._resolve_device(device)
        self._ntotal = 0
        self._is_trained = False

    # --------------------------------------------------------------------- #
    # Required interface
    # --------------------------------------------------------------------- #
    @abstractmethod
    def train(self, xb: torch.Tensor) -> None:  # pragma: no cover - abstract method
        raise NotImplementedError

    @abstractmethod
    def add(self, xb: torch.Tensor) -> None:  # pragma: no cover - abstract method
        raise NotImplementedError

    @abstractmethod
    def add_with_ids(self, xb: torch.Tensor, ids: torch.Tensor) -> None:  # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def search(self, xq: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:  # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def range_search(self, xq: torch.Tensor, radius: float) -> Sequence[torch.Tensor]:  # pragma: no cover
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:  # pragma: no cover - abstract method
        raise NotImplementedError

    # --------------------------------------------------------------------- #
    # Public helpers
    # --------------------------------------------------------------------- #
    @property
    def d(self) -> int:
        return self._d

    @property
    def metric(self) -> MetricType:
        return self._metric

    @property
    def dtype(self) -> torch.dtype:
        return self._dtype

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def ntotal(self) -> int:
        return self._ntotal

    @property
    def is_trained(self) -> bool:
        return self._is_trained

    def to(self, device: torch.device | str | None) -> "IndexBase":
        """Return a copy of the index on the requested device."""
        if device is None:
            return self
        target = self._resolve_device(device)
        cloned = copy.deepcopy(self)
        cloned._apply_to_tensors(lambda tensor: tensor.to(target))
        cloned._device = target
        return cloned

    def cpu(self) -> "IndexBase":
        return self.to(torch.device("cpu"))

    def cuda(self, device: int | str | torch.device | None = None) -> "IndexBase":
        target = torch.device(device) if device is not None else torch.device("cuda")
        return self.to(target)

    def rocm(self, device: int | str | torch.device | None = None) -> "IndexBase":
        if not torch.version.hip:
            raise RuntimeError("Current PyTorch build does not support ROCm.")
        target = torch.device(device) if device is not None else torch.device("cuda")
        return self.to(target)

    def dml(self, device: int | str | torch.device | None = None) -> "IndexBase":
        target = torch.device(device) if device is not None else torch.device("dml")
        return self.to(target)


    # ------------------------------------------------------------------ #
    # Utilities for subclasses
    # ------------------------------------------------------------------ #
    def _resolve_device(self, device: torch.device | str | None) -> torch.device:
        if device is None:
            # Mirror torch default device (currently CPU) without forcing CUDA init.
            return torch.empty(0).device
        return torch.device(device)

    def _tensor_attributes(self) -> Iterable[str]:
        """Override to return tensor attribute names to be moved in `to()`."""
        return ()


    def _apply_to_tensors(self, fn) -> None:
        for name in self._tensor_attributes():
            tensor = getattr(self, name, None)
            if tensor is None:
                continue
            setattr(self, name, fn(tensor))

    def _validate_input(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            raise TypeError("Expected torch.Tensor input.")
        if x.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got shape {tuple(x.shape)}.")
        if x.shape[1] != self._d:
            raise ValueError(f"Tensor dimension mismatch: expected {self._d}, got {x.shape[1]}.")
        if x.dtype not in {torch.float16, torch.float32, torch.bfloat16}:
            raise ValueError("Tensor dtype must be float16/float32/bfloat16.")
        return x.to(self._device)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(d={self._d}, metric={self._metric}, "
            f"device='{self._device}', dtype={self._dtype}, ntotal={self._ntotal}, "
            f"is_trained={self._is_trained})"
        )
