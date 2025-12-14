from __future__ import annotations

import torch

from .base import IndexBase, MetricType


class IndexFlat(IndexBase):
    """Base implementation shared by IndexFlatL2/IP."""

    def __init__(
        self,
        d: int,
        *,
        metric: MetricType = "l2",
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(d, metric=metric, device=device, dtype=dtype)
        self._xb = torch.empty((0, d), dtype=self.dtype, device=self.device)
        self._xb_norms = torch.empty(0, dtype=self.dtype, device=self.device)
        self._ids = torch.empty(0, dtype=torch.long, device=self.device)
        self._is_trained = True  # Flat indexes do not require training.
        self._search_mode = "matrix"
        self._auto_matrix_bytes_threshold = 256 * 1024 * 1024

    # ------------------------------------------------------------------ #
    # IndexBase requirements
    # ------------------------------------------------------------------ #
    def train(self, xb: torch.Tensor) -> None:
        self._validate_input(xb)
        self._is_trained = True

    def add(self, xb: torch.Tensor) -> None:
        xb = self._validate_input(xb)
        if xb.shape[0] == 0:
            return
        new_ids = torch.arange(
            self._ntotal,
            self._ntotal + xb.shape[0],
            dtype=torch.long,
            device=self.device,
        )
        self._append(xb, new_ids)

    def add_with_ids(self, xb: torch.Tensor, ids: torch.Tensor) -> None:
        xb = self._validate_input(xb)
        ids = self._validate_ids(ids, xb.shape[0])
        self._append(xb, ids.to(self.device))

    @property
    def search_mode(self) -> str:
        return self._search_mode

    @search_mode.setter
    def search_mode(self, value: str) -> None:
        if value not in {"matrix", "chunked", "auto"}:
            raise ValueError("search_mode must be 'matrix', 'chunked', or 'auto'.")
        self._search_mode = value

    @property
    def auto_matrix_bytes_threshold(self) -> int:
        return int(self._auto_matrix_bytes_threshold)

    @auto_matrix_bytes_threshold.setter
    def auto_matrix_bytes_threshold(self, value: int) -> None:
        value_i = int(value)
        if value_i <= 0:
            raise ValueError("auto_matrix_bytes_threshold must be > 0.")
        self._auto_matrix_bytes_threshold = value_i

    def search(self, xq: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        xq = self._validate_input(xq)
        if k <= 0:
            raise ValueError("k must be a positive integer.")
        if self._ntotal == 0:
            dists = torch.full((xq.shape[0], k), float("inf"), dtype=self.dtype, device=self.device)
            labels = torch.full((xq.shape[0], k), -1, dtype=torch.long, device=self.device)
            if self.metric == "ip":
                dists.fill_(float("-inf"))
            return dists, labels

        mode = self._search_mode
        if mode == "auto":
            element_size = torch.empty((), dtype=self.dtype, device="cpu").element_size()
            est_bytes = int(xq.shape[0]) * int(self._ntotal) * int(element_size)
            mode = "chunked" if est_bytes > int(self._auto_matrix_bytes_threshold) else "matrix"

        if mode == "chunked":
            return self._search_chunked(xq, k)

        scores = self._compute_scores(xq, self._xb)
        if self.metric == "l2":
            dists, idx = torch.topk(scores, min(k, scores.shape[1]), largest=False, dim=1)
        else:  # IP
            dists, idx = torch.topk(scores, min(k, scores.shape[1]), largest=True, dim=1)

        labels = self._ids[idx]
        if idx.shape[1] < k:
            pad_cols = k - idx.shape[1]
            pad_dist = torch.full(
                (scores.shape[0], pad_cols),
                float("inf") if self.metric == "l2" else float("-inf"),
                dtype=self.dtype,
                device=self.device,
            )
            pad_labels = torch.full(
                (scores.shape[0], pad_cols),
                -1,
                dtype=torch.long,
                device=self.device,
            )
            dists = torch.cat([dists, pad_dist], dim=1)
            labels = torch.cat([labels, pad_labels], dim=1)
        return dists, labels

    def _search_chunked(self, xq: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        nq = xq.shape[0]
        nb = int(self._ntotal)
        largest = self.metric == "ip"
        fill = float("-inf") if largest else float("inf")

        best_scores = torch.full((nq, k), fill, dtype=self.dtype, device=self.device)
        best_idx = torch.full((nq, k), -1, dtype=torch.long, device=self.device)
        if nb == 0:
            labels = torch.full((nq, k), -1, dtype=torch.long, device=self.device)
            if self.metric == "l2":
                best_scores.fill_(float("inf"))
            return best_scores, labels

        q = xq.to(self.dtype).contiguous()
        q2 = (q * q).sum(dim=1) if self.metric == "l2" else None

        element_size = torch.empty((), dtype=self.dtype, device="cpu").element_size()
        budget_bytes = 256 * 1024 * 1024 if self.device.type == "cuda" else 512 * 1024 * 1024
        chunk = max(1, int(budget_bytes // max(1, nq * element_size)))
        chunk = min(chunk, nb)

        for start in range(0, nb, chunk):
            end = min(nb, start + chunk)
            xb_chunk = self._xb[start:end]
            prod = torch.matmul(q, xb_chunk.transpose(0, 1))
            if self.metric == "l2":
                x2 = self._xb_norms[start:end]
                dist = q2.unsqueeze(1) + x2.unsqueeze(0) - (2.0 * prod)
                dist = dist.clamp_min_(0)
                topk = min(k, dist.shape[1])
                cand_scores, cand_j = torch.topk(dist, topk, largest=False, dim=1)
            else:
                topk = min(k, prod.shape[1])
                cand_scores, cand_j = torch.topk(prod, topk, largest=True, dim=1)

            cand_idx = (start + cand_j).to(torch.long)
            if topk < k:
                pad_cols = k - topk
                cand_scores = torch.cat(
                    [cand_scores, torch.full((nq, pad_cols), fill, dtype=self.dtype, device=self.device)],
                    dim=1,
                )
                cand_idx = torch.cat(
                    [cand_idx, torch.full((nq, pad_cols), -1, dtype=torch.long, device=self.device)],
                    dim=1,
                )

            best_scores, best_idx = self._merge_topk(best_scores, best_idx, cand_scores, cand_idx, k, largest=largest)

        labels = self._ids.index_select(0, best_idx.clamp_min(0).reshape(-1)).reshape(best_idx.shape)
        labels = torch.where(best_idx < 0, torch.full_like(labels, -1), labels)
        return best_scores, labels

    def range_search(self, xq: torch.Tensor, radius: float):
        xq = self._validate_input(xq)
        nq = xq.shape[0]
        if self._ntotal == 0:
            lims = torch.zeros(nq + 1, dtype=torch.long, device=self.device)
            return lims, torch.empty(0, dtype=self.dtype, device=self.device), torch.empty(
                0, dtype=torch.long, device=self.device
            )

        scores = self._compute_scores(xq, self._xb)
        if self.metric == "l2":
            mask = scores <= radius
        else:
            mask = scores >= radius

        vals = scores[mask]
        repeated_ids = self._ids.unsqueeze(0).expand(nq, -1)
        labels = repeated_ids[mask]
        counts = mask.sum(dim=1, dtype=torch.long)
        lims = torch.zeros(nq + 1, dtype=torch.long, device=self.device)
        lims[1:] = torch.cumsum(counts, dim=0)
        return lims, vals, labels

    def _merge_topk(
        self,
        best_scores: torch.Tensor,
        best_idx: torch.Tensor,
        cand_scores: torch.Tensor,
        cand_idx: torch.Tensor,
        k: int,
        *,
        largest: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        merged_scores = torch.cat([best_scores, cand_scores], dim=1)
        merged_idx = torch.cat([best_idx, cand_idx], dim=1)
        new_scores, pos = torch.topk(merged_scores, k, largest=largest, dim=1)
        new_idx = torch.gather(merged_idx, 1, pos)
        return new_scores, new_idx

    def reset(self) -> None:
        self._xb = torch.empty((0, self.d), dtype=self.dtype, device=self.device)
        self._xb_norms = torch.empty(0, dtype=self.dtype, device=self.device)
        self._ids = torch.empty(0, dtype=torch.long, device=self.device)
        self._ntotal = 0

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _tensor_attributes(self):
        return ("_xb", "_xb_norms", "_ids")

    def _append(self, xb: torch.Tensor, ids: torch.Tensor) -> None:
        self._xb = torch.cat([self._xb, xb], dim=0)
        if self.metric == "l2":
            norms = (xb * xb).sum(dim=1)
            self._xb_norms = torch.cat([self._xb_norms, norms], dim=0)
        self._ids = torch.cat([self._ids, ids.to(torch.long)], dim=0)
        self._ntotal = self._xb.shape[0]

    def _compute_scores(self, xq: torch.Tensor, xb: torch.Tensor) -> torch.Tensor:
        if xb.shape[0] == 0:
            return torch.empty(xq.shape[0], 0, dtype=self.dtype, device=self.device)
        if self.metric == "l2":
            xq_cast = xq.to(self.dtype)
            xb_cast = xb.to(self.dtype)
            xq_norm = (xq_cast * xq_cast).sum(dim=1, keepdim=True)
            xb_norm = self._xb_norms.unsqueeze(0) if self._xb_norms.numel() else (xb_cast * xb_cast).sum(dim=1).unsqueeze(0)
            dist = xq_norm + xb_norm - (2.0 * (xq_cast @ xb_cast.t()))
            return dist.clamp_min_(0)
        xq_cast = xq.to(self.dtype)
        xb_cast = xb.to(self.dtype)
        return xq_cast @ xb_cast.t()

    def _validate_ids(self, ids: torch.Tensor, expected: int) -> torch.Tensor:
        if not isinstance(ids, torch.Tensor):
            raise TypeError("ids must be a torch.Tensor of dtype long.")
        if ids.ndim != 1 or ids.shape[0] != expected:
            raise ValueError(f"ids must be 1-D with length {expected}.")
        if ids.dtype not in {torch.int64, torch.long}:
            raise ValueError("ids dtype must be torch.long.")
        return ids



class IndexFlatL2(IndexFlat):
    def __init__(
        self,
        d: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(d, metric="l2", device=device, dtype=dtype)


class IndexFlatIP(IndexFlat):
    def __init__(
        self,
        d: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(d, metric="ip", device=device, dtype=dtype)
