from __future__ import annotations

import heapq
from typing import Optional

import torch

from ..nn import kmeans
from .base import IndexBase, MetricType


class _Workspace:
    def __init__(self) -> None:
        self._buffers: dict[str, torch.Tensor] = {}
        self.capacity: dict[str, int] = {}

    def __deepcopy__(self, memo) -> "_Workspace":
        return _Workspace()

    def ensure(
        self, name: str, shape: tuple[int, ...], *, dtype: torch.dtype, device: torch.device
    ) -> torch.Tensor:
        required = 1
        for dim in shape:
            required *= int(dim)
        if required <= 0:
            return torch.empty(shape, dtype=dtype, device=device)

        buf = self._buffers.get(name)
        if buf is None or buf.device != device or buf.dtype != dtype:
            buf = None

        if buf is None or buf.numel() < required:
            current = int(buf.numel()) if buf is not None else 0
            grow = int(current * 1.5) if current > 0 else 0
            new_size = max(required, grow)
            buf = torch.empty(new_size, dtype=dtype, device=device)
            self._buffers[name] = buf
            self.capacity[name] = int(buf.numel())

        view = buf[:required].view(shape)
        return view


class IndexIVFFlat(IndexBase):
    """PyTorch implementation of IVF-Flat."""

    def __init__(
        self,
        d: int,
        *,
        metric: MetricType = "l2",
        nlist: Optional[int] = None,
        nprobe: Optional[int] = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(d, metric=metric, device=device, dtype=dtype)
        self._nlist = int(nlist) if nlist is not None else 1024
        if self._nlist <= 0:
            raise ValueError("nlist must be positive.")
        self._centroids_t: torch.Tensor | None = None
        self._centroid_norm2: torch.Tensor | None = None
        self._list_sizes: torch.Tensor | None = None
        self._list_sizes_cpu: list[int] | None = None
        self._effective_max_codes_cache: int | None = None
        self._workspace = _Workspace()
        self._nprobe = 1
        self.nprobe = nprobe or min(8, self._nlist)
        self._max_codes = 0
        self._search_mode = "matrix"
        self._auto_search_avg_group_threshold = 8.0
        self._csr_small_batch_avg_group_threshold = 64.0
        self._approximate_mode = False
        self._reset_storage()

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def nlist(self) -> int:
        return self._nlist

    @property
    def nprobe(self) -> int:
        return self._nprobe

    @nprobe.setter
    def nprobe(self, value: int) -> None:
        if value <= 0:
            raise ValueError("nprobe must be positive.")
        self._nprobe = min(value, self._nlist)
        self._invalidate_max_codes_cache()

    @property
    def max_codes(self) -> int:
        return self._max_codes

    @max_codes.setter
    def max_codes(self, value: int) -> None:
        if value < 0:
            raise ValueError("max_codes must be >= 0.")
        self._max_codes = int(value)
        self._invalidate_max_codes_cache()

    @property
    def search_mode(self) -> str:
        return self._search_mode

    @search_mode.setter
    def search_mode(self, value: str) -> None:
        if value not in {"matrix", "csr", "auto"}:
            raise ValueError("search_mode must be 'matrix', 'csr', or 'auto'.")
        self._search_mode = value

    @property
    def auto_search_avg_group_threshold(self) -> float:
        return self._auto_search_avg_group_threshold

    @auto_search_avg_group_threshold.setter
    def auto_search_avg_group_threshold(self, value: float) -> None:
        value_f = float(value)
        if not (value_f > 0):
            raise ValueError("auto_search_avg_group_threshold must be > 0.")
        self._auto_search_avg_group_threshold = value_f

    @property
    def csr_small_batch_avg_group_threshold(self) -> float:
        return self._csr_small_batch_avg_group_threshold

    @csr_small_batch_avg_group_threshold.setter
    def csr_small_batch_avg_group_threshold(self, value: float) -> None:
        value_f = float(value)
        if not (value_f > 0):
            raise ValueError("csr_small_batch_avg_group_threshold must be > 0.")
        self._csr_small_batch_avg_group_threshold = value_f

    @property
    def approximate_mode(self) -> bool:
        return self._approximate_mode

    @approximate_mode.setter
    def approximate_mode(self, value: bool) -> None:
        self._approximate_mode = bool(value)

    # ------------------------------------------------------------------ #
    # Core API
    # ------------------------------------------------------------------ #
    def train(
        self,
        xb: torch.Tensor,
        *,
        max_iter: int = 25,
        batch_size: Optional[int] = None,
        tol: float = 1e-3,
        generator: Optional[torch.Generator] = None,
        verbose: bool = False,
    ) -> None:
        xb = self._validate_input(xb)
        if xb.shape[0] < self._nlist:
            raise ValueError("Need at least nlist samples to train k-means.")

        result = kmeans(
            xb,
            self._nlist,
            metric=self.metric,
            max_iter=max_iter,
            batch_size=batch_size,
            tol=tol,
            generator=generator,
            verbose=verbose,
        )
        self._centroids = result.centroids.to(self.device)
        self._list_offsets = torch.zeros(self._nlist + 1, dtype=torch.long, device=self.device)
        self._list_offsets_cpu = [0] * (self._nlist + 1)
        self._packed_embeddings = torch.empty((0, self.d), dtype=self.dtype, device=self.device)
        self._packed_norms = torch.empty(0, dtype=self.dtype, device=self.device)
        self._list_ids = torch.empty(0, dtype=torch.long, device=self.device)
        self._is_trained = True
        self._ntotal = 0
        self._invalidate_search_cache()

    def add(self, xb: torch.Tensor) -> None:
        self._ensure_trained()
        xb = self._validate_input(xb)
        if xb.shape[0] == 0:
            return
        ids = torch.arange(
            self._ntotal,
            self._ntotal + xb.shape[0],
            device=self.device,
            dtype=torch.long,
        )
        self._assign_and_append(xb, ids)

    def add_with_ids(self, xb: torch.Tensor, ids: torch.Tensor) -> None:
        self._ensure_trained()
        xb = self._validate_input(xb)
        if ids.ndim != 1 or ids.shape[0] != xb.shape[0]:
            raise ValueError("ids must be 1-D with the same length as xb.")
        if ids.dtype not in {torch.long, torch.int64}:
            raise ValueError("ids must be torch.long.")
        self._assign_and_append(xb, ids.to(self.device))

    def search(self, xq: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_ready_for_search(k)
        xq = self._validate_input(xq)
        if xq.shape[0] == 0:
            return (
                torch.empty(0, k, dtype=self.dtype, device=self.device),
                torch.empty(0, k, dtype=torch.long, device=self.device),
            )

        effective_max_codes = self._effective_max_codes()

        if self._search_mode == "csr":
            return self._search_csr_online(xq, k, max_codes=effective_max_codes)

        if self._search_mode == "auto" and xq.device.type == "cuda":
            probe = min(self._nprobe, self._nlist)
            avg_group_size = (xq.shape[0] * probe) / max(1, self._nlist)
            if avg_group_size >= self._auto_search_avg_group_threshold:
                return self._search_csr_online(xq, k, max_codes=effective_max_codes)

        top_lists = self._top_probed_lists(xq)
        query_candidate_counts = self._estimate_candidates_per_query(top_lists, max_codes=effective_max_codes)
        query_candidate_counts_cpu = query_candidate_counts.to("cpu")
        chunks = self._iter_query_chunks(query_candidate_counts)
        dists = torch.empty((xq.shape[0], k), dtype=self.dtype, device=self.device)
        labels = torch.empty((xq.shape[0], k), dtype=torch.long, device=self.device)

        for start, end in chunks:
            chunk_q = xq[start:end]
            chunk_lists = top_lists[start:end]
            max_candidates = int(query_candidate_counts_cpu[start:end].max().item()) if end > start else 0
            pos_1d = (
                torch.arange(max_candidates, dtype=torch.long, device=self.device)
                if max_candidates > 0
                else torch.empty(0, dtype=torch.long, device=self.device)
            )
            index_matrix, query_counts = self._build_candidate_index_matrix_from_lists(
                chunk_lists,
                max_candidates=max_candidates,
                pos_1d=pos_1d,
                max_codes=effective_max_codes,
            )
            chunk_dists, chunk_labels = self._search_from_index_matrix(chunk_q, index_matrix, query_counts, k)
            dists[start:end] = chunk_dists
            labels[start:end] = chunk_labels

        return dists, labels

    def range_search(self, xq: torch.Tensor, radius: float):
        self._ensure_trained()
        xq = self._validate_input(xq)
        nq = xq.shape[0]
        lims = torch.zeros(nq + 1, dtype=torch.long, device=self.device)
        values_list = []
        ids_list = []

        if nq == 0:
            return lims, torch.empty(0, dtype=self.dtype, device=self.device), torch.empty(
                0, dtype=torch.long, device=self.device
            )

        effective_max_codes = self._effective_max_codes()
        top_lists = self._top_probed_lists(xq)
        query_candidate_counts = self._estimate_candidates_per_query(top_lists, max_codes=effective_max_codes)
        query_candidate_counts_cpu = query_candidate_counts.to("cpu")
        chunks = self._iter_query_chunks(query_candidate_counts)
        current_total = 0
        for start, end in chunks:
            chunk_q = xq[start:end]
            chunk_lists = top_lists[start:end]
            max_candidates = int(query_candidate_counts_cpu[start:end].max().item()) if end > start else 0
            pos_1d = (
                torch.arange(max_candidates, dtype=torch.long, device=self.device)
                if max_candidates > 0
                else torch.empty(0, dtype=torch.long, device=self.device)
            )
            index_matrix, query_counts = self._build_candidate_index_matrix_from_lists(
                chunk_lists,
                max_candidates=max_candidates,
                pos_1d=pos_1d,
                max_codes=effective_max_codes,
            )
            chunk_lims, chunk_vals, chunk_ids = self._range_from_index_matrix(
                chunk_q, index_matrix, query_counts, radius
            )

            counts = chunk_lims[1:] - chunk_lims[:-1]
            if counts.numel():
                lims[start + 1 : end + 1] = current_total + torch.cumsum(counts, dim=0)
            else:
                lims[start + 1 : end + 1] = current_total
            current_total += int(chunk_lims[-1].item())
            if chunk_vals.numel():
                values_list.append(chunk_vals)
                ids_list.append(chunk_ids)

        lims[-1] = current_total
        values = torch.cat(values_list) if values_list else torch.empty(0, dtype=self.dtype, device=self.device)
        ids = torch.cat(ids_list) if ids_list else torch.empty(0, dtype=torch.long, device=self.device)
        return lims, values, ids

    def reset(self) -> None:
        self._reset_storage()
        self._ntotal = 0
        self._is_trained = False
        self._invalidate_search_cache()

    # ------------------------------------------------------------------ #
    # Serialization
    # ------------------------------------------------------------------ #
    def state_dict(self) -> dict[str, torch.Tensor | int | str]:
        return {
            "d": self.d,
            "metric": self.metric,
            "nlist": self._nlist,
            "nprobe": self._nprobe,
            "max_codes": self._max_codes,
            "centroids": self._centroids,
            "list_offsets": self._list_offsets,
            "packed_embeddings": self._packed_embeddings,
            "packed_norms": self._packed_norms,
            "list_ids": self._list_ids,
            "dtype": self.dtype,
        }

    def load_state_dict(self, state: dict) -> None:
        required = {"centroids", "list_offsets", "packed_embeddings", "list_ids"}
        missing = required - state.keys()
        if missing:
            raise KeyError(f"Missing keys in state_dict: {missing}")
        self._centroids = state["centroids"].to(self.device)
        self._list_offsets = state["list_offsets"].to(self.device)
        self._list_offsets_cpu = self._list_offsets.to("cpu").tolist()
        self._packed_embeddings = state["packed_embeddings"].to(self.device)
        packed_norms = state.get("packed_norms")
        if isinstance(packed_norms, torch.Tensor):
            self._packed_norms = packed_norms.to(self.device)
        else:
            self._packed_norms = (self._packed_embeddings * self._packed_embeddings).sum(dim=1)
        self._list_ids = state["list_ids"].to(self.device)
        self._nlist = int(state.get("nlist", self._nlist))
        self.nprobe = int(state.get("nprobe", self._nprobe))
        self.max_codes = int(state.get("max_codes", self._max_codes))
        self._is_trained = self._centroids.numel() > 0
        self._ntotal = self._packed_embeddings.shape[0]
        self._invalidate_search_cache()

    def save(self, path: str) -> None:
        torch.save(self.state_dict(), path)

    @classmethod
    def load(cls, path: str, map_location: Optional[torch.device | str] = None) -> "IndexIVFFlat":
        state = torch.load(path, map_location=map_location)
        index = cls(
            d=int(state["d"]),
            metric=state.get("metric", "l2"),
            nlist=int(state.get("nlist", 1)),
            nprobe=int(state.get("nprobe", 1)),
        )
        index.load_state_dict(state)
        return index

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _ensure_trained(self) -> None:
        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding vectors.")

    def _ensure_ready_for_search(self, k: int) -> None:
        self._ensure_trained()
        if k <= 0:
            raise ValueError("k must be positive.")

    def _invalidate_search_cache(self) -> None:
        self._centroids_t = None
        self._centroid_norm2 = None
        self._list_sizes = None
        self._list_sizes_cpu = None
        self._effective_max_codes_cache = None

    def _invalidate_max_codes_cache(self) -> None:
        self._effective_max_codes_cache = None

    def _ensure_search_cache(self) -> None:
        if self._centroids_t is None or self._centroid_norm2 is None:
            if self._centroids.numel() == 0:
                self._centroids_t = None
                self._centroid_norm2 = None
            else:
                centroids_cast = self._centroids.to(self.dtype)
                self._centroids_t = centroids_cast.transpose(0, 1).contiguous()
                self._centroid_norm2 = (centroids_cast * centroids_cast).sum(dim=1)

        if self._list_sizes is None:
            if self._list_offsets.numel() == 0:
                self._list_sizes = torch.zeros(0, dtype=torch.long, device=self.device)
            else:
                self._list_sizes = self._list_offsets[1:] - self._list_offsets[:-1]

        if self._list_sizes_cpu is None:
            if len(self._list_offsets_cpu) == self._nlist + 1:
                offsets_cpu = self._list_offsets_cpu
            else:
                offsets_cpu = self._list_offsets.to("cpu").tolist()
            self._list_sizes_cpu = [
                int(offsets_cpu[i + 1]) - int(offsets_cpu[i]) for i in range(self._nlist)
            ]

    def _get_list_sizes(self) -> torch.Tensor:
        sizes = self._list_sizes
        if sizes is None or sizes.device != self.device:
            if self._list_offsets.numel() == 0:
                sizes = torch.zeros(0, dtype=torch.long, device=self.device)
            else:
                sizes = self._list_offsets[1:] - self._list_offsets[:-1]
            self._list_sizes = sizes
        return sizes

    def _reset_storage(self) -> None:
        self._centroids = torch.empty((0, self.d), dtype=self.dtype, device=self.device)
        self._packed_embeddings = torch.empty((0, self.d), dtype=self.dtype, device=self.device)
        self._packed_norms = torch.empty(0, dtype=self.dtype, device=self.device)
        self._list_ids = torch.empty(0, dtype=torch.long, device=self.device)
        self._list_offsets = torch.zeros(self._nlist + 1, dtype=torch.long, device=self.device)
        self._list_offsets_cpu = [0] * (self._nlist + 1)
        self._invalidate_search_cache()

    def _assign_and_append(self, xb: torch.Tensor, ids: torch.Tensor) -> None:
        assign = self._assign_centroids(xb)
        if self._ntotal == 0:
            order = torch.argsort(assign)
            assign_sorted = assign[order]
            xb_sorted = xb[order]
            ids_sorted = ids[order]
            self._packed_embeddings = xb_sorted
            self._packed_norms = (xb_sorted * xb_sorted).sum(dim=1)
            self._list_ids = ids_sorted.to(torch.long)
            counts = torch.bincount(assign_sorted, minlength=self._nlist)
            offsets = torch.zeros(self._nlist + 1, dtype=torch.long, device=self.device)
            offsets[1:] = torch.cumsum(counts, dim=0)
            self._list_offsets = offsets
            self._list_offsets_cpu = offsets.to("cpu").tolist()
            self._ntotal = xb.shape[0]
            self._invalidate_search_cache()
            return

        order = torch.argsort(assign)
        assign_sorted = assign[order]
        xb_sorted = xb[order]
        ids_sorted = ids[order].to(torch.long)
        norms_sorted = (xb_sorted * xb_sorted).sum(dim=1)
        counts = torch.bincount(assign_sorted, minlength=self._nlist)

        old_offsets = self._list_offsets
        old_sizes = (old_offsets[1:] - old_offsets[:-1]).to(torch.long)
        new_sizes = old_sizes + counts
        new_offsets = torch.zeros(self._nlist + 1, dtype=torch.long, device=self.device)
        new_offsets[1:] = torch.cumsum(new_sizes, dim=0)
        total = self._ntotal + xb.shape[0]

        # Compute start offsets for each list inside the sorted new batch.
        new_prefix = torch.zeros(self._nlist + 1, dtype=torch.long, device=self.device)
        new_prefix[1:] = torch.cumsum(counts, dim=0)

        # Avoid per-list device sync by materializing small index arrays on CPU.
        old_offsets_cpu = old_offsets.to("cpu").tolist()
        new_offsets_cpu = new_offsets.to("cpu").tolist()
        counts_cpu = counts.to("cpu").tolist()
        new_prefix_cpu = new_prefix.to("cpu").tolist()

        out_embeddings = torch.empty((total, self.d), dtype=self.dtype, device=self.device)
        out_norms = torch.empty(total, dtype=self.dtype, device=self.device)
        out_ids = torch.empty(total, dtype=torch.long, device=self.device)

        for list_id in range(self._nlist):
            old_start = int(old_offsets_cpu[list_id])
            old_end = int(old_offsets_cpu[list_id + 1])
            old_len = old_end - old_start

            new_count = int(counts_cpu[list_id])
            new_start = int(new_prefix_cpu[list_id])
            new_end = int(new_prefix_cpu[list_id + 1])

            out_start = int(new_offsets_cpu[list_id])
            pos = out_start
            if old_len:
                out_embeddings[pos : pos + old_len] = self._packed_embeddings[old_start:old_end]
                out_norms[pos : pos + old_len] = self._packed_norms[old_start:old_end]
                out_ids[pos : pos + old_len] = self._list_ids[old_start:old_end]
                pos += old_len

            if new_count:
                out_embeddings[pos : pos + new_count] = xb_sorted[new_start:new_end]
                out_norms[pos : pos + new_count] = norms_sorted[new_start:new_end]
                out_ids[pos : pos + new_count] = ids_sorted[new_start:new_end]

        self._packed_embeddings = out_embeddings
        self._packed_norms = out_norms
        self._list_ids = out_ids
        self._list_offsets = new_offsets
        self._list_offsets_cpu = new_offsets_cpu
        self._ntotal = total
        self._invalidate_search_cache()

    def _assign_centroids(self, xb: torch.Tensor) -> torch.Tensor:
        if self._centroids.shape[0] == 0:
            raise RuntimeError("Centroids are empty; train the index first.")
        if self.device.type == "cpu":
            batch = 65536
        else:
            batch = 131072
        batch = min(batch, xb.shape[0])
        out = torch.empty((xb.shape[0],), dtype=torch.long, device=self.device)
        for start in range(0, xb.shape[0], batch):
            end = min(xb.shape[0], start + batch)
            scores = self._pairwise_centroids(xb[start:end])
            if self.metric == "l2":
                out[start:end] = torch.argmin(scores, dim=1)
            else:
                out[start:end] = torch.argmax(scores, dim=1)
        return out

    def _suggest_chunk_size(self, total_queries: int) -> int:
        if total_queries <= 0:
            return 0
        if self.device.type == "cpu":
            return min(total_queries, 8)
        avg_list = max(1, (self._ntotal + max(1, self._nlist) - 1) // max(1, self._nlist))
        denom = max(1, self._nprobe * avg_list)
        budget = self._candidate_budget()
        chunk = max(1, budget // denom)
        return min(total_queries, chunk)

    def _top_probed_lists(self, xq: torch.Tensor) -> torch.Tensor:
        if xq.shape[0] == 0 or self._centroids.shape[0] == 0:
            return torch.empty((xq.shape[0], 0), dtype=torch.long, device=self.device)
        centroid_scores = self._pairwise_centroids(xq)
        probe = min(self._nprobe, centroid_scores.shape[1])
        if probe <= 0:
            return torch.empty((xq.shape[0], 0), dtype=torch.long, device=self.device)
        largest = self.metric == "ip"
        _, top_lists = torch.topk(centroid_scores, probe, largest=largest, dim=1)
        return top_lists

    def _effective_max_codes(self) -> int:
        max_codes = int(self._max_codes)
        if max_codes <= 0:
            return 0

        cached = self._effective_max_codes_cache
        if cached is not None:
            return cached

        probe = min(self._nprobe, self._nlist)
        if probe <= 0:
            return 0

        self._ensure_search_cache()
        sizes = self._list_sizes_cpu or []
        if probe >= self._nlist:
            max_possible = sum(sizes)
        else:
            max_possible = sum(heapq.nlargest(probe, sizes))

        effective = 0 if max_codes >= max_possible else max_codes
        self._effective_max_codes_cache = effective
        return effective

    def _estimate_candidates_per_query(self, top_lists: torch.Tensor, *, max_codes: int | None = None) -> torch.Tensor:
        if top_lists.numel() == 0:
            return torch.zeros(top_lists.shape[0], dtype=torch.long, device=self.device)
        sizes = self._get_list_sizes()
        per_probe = sizes[top_lists]
        counts = per_probe.sum(dim=1)
        max_codes_i = self._max_codes if max_codes is None else int(max_codes)
        if max_codes_i:
            counts = torch.minimum(counts, torch.full_like(counts, max_codes_i))
        return counts

    def _build_candidate_index_matrix_from_lists(
        self, top_lists: torch.Tensor, *, max_candidates: int, pos_1d: torch.Tensor, max_codes: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = top_lists.shape[0]
        if chunk == 0 or top_lists.numel() == 0:
            return (
                torch.empty((chunk, 0), dtype=torch.long, device=self.device),
                torch.zeros(chunk, dtype=torch.long, device=self.device),
            )

        if max_candidates <= 0:
            return (
                torch.empty((chunk, 0), dtype=torch.long, device=self.device),
                torch.zeros(chunk, dtype=torch.long, device=self.device),
            )

        probe = top_lists.shape[1]
        flat_lists = top_lists.reshape(-1)
        starts = self._list_offsets[flat_lists]
        ends = self._list_offsets[flat_lists + 1]
        sizes = (ends - starts).reshape(chunk, probe)
        max_codes_i = self._max_codes if max_codes is None else int(max_codes)
        if max_codes_i:
            budget = torch.full((chunk, 1), max_codes_i, dtype=torch.long, device=self.device)
            prev_cum = torch.cumsum(sizes, dim=1) - sizes
            remaining = (budget - prev_cum).clamp_min(0)
            sizes = torch.minimum(sizes, remaining)
        query_counts = sizes.sum(dim=1)
        starts2d = starts.reshape(chunk, probe)
        cum = torch.cumsum(sizes, dim=1)
        prev_cum = cum - sizes

        pos = pos_1d.unsqueeze(0).expand(chunk, -1).contiguous()
        probe_idx = torch.searchsorted(cum, pos, right=True)
        valid = pos < query_counts.unsqueeze(1)
        probe_idx = probe_idx.clamp_max(probe - 1)
        within = pos - torch.gather(prev_cum, 1, probe_idx)
        indices = torch.gather(starts2d, 1, probe_idx) + within
        index_matrix = torch.where(valid, indices, torch.full_like(indices, -1))
        return index_matrix, query_counts

    def _build_tasks_from_lists(
        self, top_lists: torch.Tensor, *, max_codes: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if top_lists.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            return empty, empty, empty

        b, probe = top_lists.shape
        sizes = self._get_list_sizes()[top_lists]

        max_codes_i = self._max_codes if max_codes is None else int(max_codes)
        if max_codes_i:
            cum = torch.cumsum(sizes, dim=1)
            keep = cum <= max_codes_i
            keep[:, 0] = True
        else:
            keep = torch.ones_like(top_lists, dtype=torch.bool)

        q_idx = torch.arange(b, dtype=torch.long, device=self.device).unsqueeze(1).expand(-1, probe)
        tasks_l = top_lists[keep].to(torch.long)
        tasks_q = q_idx[keep]
        if tasks_l.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            return empty, empty, empty

        perm = torch.argsort(tasks_l)
        tasks_l = tasks_l[perm]
        tasks_q = tasks_q[perm]
        unique_l, counts = torch.unique_consecutive(tasks_l, return_counts=True)
        starts = torch.cumsum(counts, dim=0) - counts
        ends = starts + counts
        groups = torch.stack([unique_l, starts, ends], dim=1)
        return tasks_q, tasks_l, groups

    def _build_tasks_from_lists_with_probe(
        self, top_lists: torch.Tensor, *, max_codes: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if top_lists.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            return empty, empty, empty, empty

        b, probe = top_lists.shape
        sizes = self._get_list_sizes()[top_lists]

        max_codes_i = self._max_codes if max_codes is None else int(max_codes)
        if max_codes_i:
            cum = torch.cumsum(sizes, dim=1)
            keep = cum <= max_codes_i
            keep[:, 0] = True
        else:
            keep = torch.ones_like(top_lists, dtype=torch.bool)

        q_idx = torch.arange(b, dtype=torch.long, device=self.device).unsqueeze(1).expand(-1, probe)
        p_idx = torch.arange(probe, dtype=torch.long, device=self.device).unsqueeze(0).expand(b, -1)
        tasks_l = top_lists[keep].to(torch.long)
        tasks_q = q_idx[keep]
        tasks_p = p_idx[keep]
        if tasks_l.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            return empty, empty, empty, empty

        perm = torch.argsort(tasks_l)
        tasks_l = tasks_l[perm]
        tasks_q = tasks_q[perm]
        tasks_p = tasks_p[perm]
        unique_l, counts = torch.unique_consecutive(tasks_l, return_counts=True)
        starts = torch.cumsum(counts, dim=0) - counts
        ends = starts + counts
        groups = torch.stack([unique_l, starts, ends], dim=1)
        return tasks_q, tasks_l, tasks_p, groups

    def _merge_topk(
        self,
        best_scores: torch.Tensor,
        best_idx: torch.Tensor,
        query_ids: torch.Tensor,
        cand_scores: torch.Tensor,
        cand_idx: torch.Tensor,
        k: int,
        *,
        largest: bool,
    ) -> None:
        current_scores = best_scores.index_select(0, query_ids)
        current_idx = best_idx.index_select(0, query_ids)
        merged_scores = torch.cat([current_scores, cand_scores], dim=1)
        merged_idx = torch.cat([current_idx, cand_idx], dim=1)
        new_scores, pos = torch.topk(merged_scores, k, largest=largest, dim=1)
        new_idx = torch.gather(merged_idx, 1, pos)
        best_scores.index_copy_(0, query_ids, new_scores)
        best_idx.index_copy_(0, query_ids, new_idx)

    def _search_csr_online(self, xq: torch.Tensor, k: int, *, max_codes: int) -> tuple[torch.Tensor, torch.Tensor]:
        top_lists = self._top_probed_lists(xq)
        chunks = self._iter_query_chunks_csr(xq.shape[0])

        dists = torch.empty((xq.shape[0], k), dtype=self.dtype, device=self.device)
        labels = torch.empty((xq.shape[0], k), dtype=torch.long, device=self.device)

        for start, end in chunks:
            chunk_q = xq[start:end]
            chunk_lists = top_lists[start:end]
            chunk_dists, chunk_labels = self._search_csr_online_chunk(chunk_q, chunk_lists, k, max_codes=max_codes)
            dists[start:end] = chunk_dists
            labels[start:end] = chunk_labels
        return dists, labels

    def _search_csr_online_chunk(
        self, xq_chunk: torch.Tensor, top_lists: torch.Tensor, k: int, *, max_codes: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        chunk_size = xq_chunk.shape[0]
        largest = self.metric == "ip"
        fill = float("inf") if not largest else float("-inf")
        best_scores = self._workspace.ensure(
            "csr_best_scores", (chunk_size, k), dtype=self.dtype, device=self.device
        )
        best_scores.fill_(fill)
        best_packed = self._workspace.ensure(
            "csr_best_packed", (chunk_size, k), dtype=torch.long, device=self.device
        )
        best_packed.fill_(-1)
        if chunk_size == 0 or top_lists.numel() == 0:
            return best_scores, best_packed

        q = xq_chunk if xq_chunk.dtype == self.dtype else xq_chunk.to(self.dtype)
        q = q.contiguous()
        q2 = (q * q).sum(dim=1) if self.metric == "l2" else None

        avg_group_size = (chunk_size * top_lists.shape[1]) / max(1, self._nlist)
        if avg_group_size < self._csr_small_batch_avg_group_threshold:
            query_counts = self._estimate_candidates_per_query(top_lists, max_codes=max_codes)
            max_candidates = int(query_counts.max().item()) if query_counts.numel() else 0
            if max_candidates > 0 and (chunk_size * max_candidates) <= self._candidate_budget():
                pos_1d = torch.arange(max_candidates, dtype=torch.long, device=self.device)
                index_matrix, query_counts = self._build_candidate_index_matrix_from_lists(
                    top_lists,
                    max_candidates=max_candidates,
                    pos_1d=pos_1d,
                    max_codes=max_codes,
                )
                return self._search_from_index_matrix(q, index_matrix, query_counts, k)
            return self._search_csr_buffered_chunk(q, q2, top_lists, k, max_codes=max_codes)

        tasks_q, _, groups = self._build_tasks_from_lists(top_lists, max_codes=max_codes)
        if groups.numel() == 0:
            return best_scores, best_packed

        q_tasks = q.index_select(0, tasks_q)
        q2_tasks = q2.index_select(0, tasks_q) if q2 is not None else None

        groups_cpu = groups.to("cpu").tolist()
        for l, start, end in groups_cpu:
            a = int(self._list_offsets_cpu[int(l)])
            b = int(self._list_offsets_cpu[int(l) + 1])
            if b <= a:
                continue
            query_ids = tasks_q[int(start) : int(end)]
            if query_ids.numel() == 0:
                continue

            qg = q_tasks[int(start) : int(end)]
            if q2 is not None:
                q2g = q2_tasks[int(start) : int(end)].unsqueeze(1)
            else:
                q2g = None

            vec_chunk = self._csr_vec_chunk(buffered=False)
            if (b - a) <= vec_chunk:
                x = self._packed_embeddings[a:b]
                prod = torch.matmul(qg, x.transpose(0, 1))
                if self.metric == "l2":
                    x2 = self._packed_norms[a:b]
                    dist = q2g + x2.unsqueeze(0) - (2.0 * prod)
                    dist = dist.clamp_min_(0)
                    topk = min(k, dist.shape[1])
                    cand_scores, cand_j = torch.topk(dist, topk, largest=False, dim=1)
                else:
                    topk = min(k, prod.shape[1])
                    cand_scores, cand_j = torch.topk(prod, topk, largest=True, dim=1)
                cand_packed = (a + cand_j).to(torch.long)
                if topk < k:
                    pad_cols = k - topk
                    cand_scores = torch.cat(
                        [cand_scores, torch.full((qg.shape[0], pad_cols), fill, dtype=self.dtype, device=self.device)],
                        dim=1,
                    )
                    cand_packed = torch.cat(
                        [cand_packed, torch.full((qg.shape[0], pad_cols), -1, dtype=torch.long, device=self.device)],
                        dim=1,
                    )
                self._merge_topk(best_scores, best_packed, query_ids, cand_scores, cand_packed, k, largest=largest)
            else:
                local_best_scores = self._workspace.ensure(
                    "csr_local_best_scores", (qg.shape[0], k), dtype=self.dtype, device=self.device
                )
                local_best_scores.fill_(fill)
                local_best_packed = self._workspace.ensure(
                    "csr_local_best_packed", (qg.shape[0], k), dtype=torch.long, device=self.device
                )
                local_best_packed.fill_(-1)
                local_query_ids = torch.arange(qg.shape[0], dtype=torch.long, device=self.device)
                for p in range(a, b, vec_chunk):
                    pe = min(b, p + vec_chunk)
                    x = self._packed_embeddings[p:pe]
                    if x.numel() == 0:
                        continue
                    prod = torch.matmul(qg, x.transpose(0, 1))
                    if self.metric == "l2":
                        x2 = self._packed_norms[p:pe]
                        dist = q2g + x2.unsqueeze(0) - (2.0 * prod)
                        dist = dist.clamp_min_(0)
                        topk = min(k, dist.shape[1])
                        cand_scores, cand_j = torch.topk(dist, topk, largest=False, dim=1)
                    else:
                        topk = min(k, prod.shape[1])
                        cand_scores, cand_j = torch.topk(prod, topk, largest=True, dim=1)

                    cand_packed = (p + cand_j).to(torch.long)
                    if topk < k:
                        pad_cols = k - topk
                        cand_scores = torch.cat(
                            [cand_scores, torch.full((qg.shape[0], pad_cols), fill, dtype=self.dtype, device=self.device)],
                            dim=1,
                        )
                        cand_packed = torch.cat(
                            [cand_packed, torch.full((qg.shape[0], pad_cols), -1, dtype=torch.long, device=self.device)],
                            dim=1,
                        )
                    self._merge_topk(
                        local_best_scores, local_best_packed, local_query_ids, cand_scores, cand_packed, k, largest=largest
                    )

                self._merge_topk(best_scores, best_packed, query_ids, local_best_scores, local_best_packed, k, largest=largest)

        out_ids = self._list_ids.index_select(0, best_packed.clamp_min(0).reshape(-1)).reshape(best_packed.shape)
        out_ids = torch.where(best_packed < 0, torch.full_like(out_ids, -1), out_ids)
        return best_scores, out_ids

    def _search_csr_buffered_chunk(
        self,
        q: torch.Tensor,
        q2: torch.Tensor | None,
        top_lists: torch.Tensor,
        k: int,
        *,
        max_codes: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        chunk_size = q.shape[0]
        probe = top_lists.shape[1]
        largest = self.metric == "ip"
        fill = float("inf") if not largest else float("-inf")
        if chunk_size == 0 or probe == 0:
            best_scores = self._workspace.ensure(
                "csr_buf_best_scores", (chunk_size, k), dtype=self.dtype, device=self.device
            )
            best_scores.fill_(fill)
            best_ids = self._workspace.ensure(
                "csr_buf_best_ids", (chunk_size, k), dtype=torch.long, device=self.device
            )
            best_ids.fill_(-1)
            return best_scores, best_ids

        tasks_q, _, tasks_p, groups = self._build_tasks_from_lists_with_probe(top_lists, max_codes=max_codes)
        if groups.numel() == 0:
            best_scores = self._workspace.ensure(
                "csr_buf_best_scores", (chunk_size, k), dtype=self.dtype, device=self.device
            )
            best_scores.fill_(fill)
            best_ids = self._workspace.ensure(
                "csr_buf_best_ids", (chunk_size, k), dtype=torch.long, device=self.device
            )
            best_ids.fill_(-1)
            return best_scores, best_ids

        t = tasks_q.numel()
        task_scores = self._workspace.ensure("csr_task_scores", (t, k), dtype=self.dtype, device=self.device)
        task_scores.fill_(fill)
        task_packed = self._workspace.ensure("csr_task_packed", (t, k), dtype=torch.long, device=self.device)
        task_packed.fill_(-1)

        q_tasks = q.index_select(0, tasks_q)
        q2_tasks = q2.index_select(0, tasks_q) if q2 is not None else None

        groups_cpu = groups.to("cpu").tolist()
        vec_chunk = self._csr_vec_chunk(buffered=True)
        for l, start, end in groups_cpu:
            a = int(self._list_offsets_cpu[int(l)])
            b = int(self._list_offsets_cpu[int(l) + 1])
            if b <= a:
                continue
            s = int(start)
            e = int(end)
            if e <= s:
                continue

            qg = q_tasks[s:e]
            if q2 is not None:
                q2g = q2_tasks[s:e].unsqueeze(1)
            else:
                q2g = None

            if (b - a) <= vec_chunk:
                x = self._packed_embeddings[a:b]
                prod = torch.matmul(qg, x.transpose(0, 1))
                if self.metric == "l2":
                    x2 = self._packed_norms[a:b]
                    dist = q2g + x2.unsqueeze(0) - (2.0 * prod)
                    dist = dist.clamp_min_(0)
                    topk = min(k, dist.shape[1])
                    cand_scores, cand_j = torch.topk(dist, topk, largest=False, dim=1)
                else:
                    topk = min(k, prod.shape[1])
                    cand_scores, cand_j = torch.topk(prod, topk, largest=True, dim=1)
                cand_packed = (a + cand_j).to(torch.long)
                if topk < k:
                    pad_cols = k - topk
                    cand_scores = torch.cat(
                        [cand_scores, torch.full((qg.shape[0], pad_cols), fill, dtype=self.dtype, device=self.device)],
                        dim=1,
                    )
                    cand_packed = torch.cat(
                        [cand_packed, torch.full((qg.shape[0], pad_cols), -1, dtype=torch.long, device=self.device)],
                        dim=1,
                    )
                task_scores[s:e] = cand_scores
                task_packed[s:e] = cand_packed
            else:
                local_best_scores = self._workspace.ensure(
                    "csr_buf_local_best_scores", (qg.shape[0], k), dtype=self.dtype, device=self.device
                )
                local_best_scores.fill_(fill)
                local_best_packed = self._workspace.ensure(
                    "csr_buf_local_best_packed", (qg.shape[0], k), dtype=torch.long, device=self.device
                )
                local_best_packed.fill_(-1)
                local_query_ids = torch.arange(qg.shape[0], dtype=torch.long, device=self.device)
                for p in range(a, b, vec_chunk):
                    pe = min(b, p + vec_chunk)
                    x = self._packed_embeddings[p:pe]
                    if x.numel() == 0:
                        continue
                    prod = torch.matmul(qg, x.transpose(0, 1))
                    if self.metric == "l2":
                        x2 = self._packed_norms[p:pe]
                        dist = q2g + x2.unsqueeze(0) - (2.0 * prod)
                        dist = dist.clamp_min_(0)
                        topk = min(k, dist.shape[1])
                        cand_scores, cand_j = torch.topk(dist, topk, largest=False, dim=1)
                    else:
                        topk = min(k, prod.shape[1])
                        cand_scores, cand_j = torch.topk(prod, topk, largest=True, dim=1)

                    cand_packed = (p + cand_j).to(torch.long)
                    if topk < k:
                        pad_cols = k - topk
                        cand_scores = torch.cat(
                            [cand_scores, torch.full((qg.shape[0], pad_cols), fill, dtype=self.dtype, device=self.device)],
                            dim=1,
                        )
                        cand_packed = torch.cat(
                            [cand_packed, torch.full((qg.shape[0], pad_cols), -1, dtype=torch.long, device=self.device)],
                            dim=1,
                        )
                    self._merge_topk(
                        local_best_scores,
                        local_best_packed,
                        local_query_ids,
                        cand_scores,
                        cand_packed,
                        k,
                        largest=largest,
                    )
                task_scores[s:e] = local_best_scores
                task_packed[s:e] = local_best_packed

        buf_scores = self._workspace.ensure(
            "csr_buf_scores", (chunk_size * probe, k), dtype=self.dtype, device=self.device
        )
        buf_scores.fill_(fill)
        buf_packed = self._workspace.ensure(
            "csr_buf_packed", (chunk_size * probe, k), dtype=torch.long, device=self.device
        )
        buf_packed.fill_(-1)
        linear = tasks_q * probe + tasks_p
        buf_scores.index_copy_(0, linear, task_scores)
        buf_packed.index_copy_(0, linear, task_packed)

        flat_scores = buf_scores.view(chunk_size, probe * k)
        flat_packed = buf_packed.view(chunk_size, probe * k)
        best_scores, pos = torch.topk(flat_scores, k, largest=largest, dim=1)
        best_packed = torch.gather(flat_packed, 1, pos)
        out_ids = self._list_ids.index_select(0, best_packed.clamp_min(0).reshape(-1)).reshape(best_packed.shape)
        out_ids = torch.where(best_packed < 0, torch.full_like(out_ids, -1), out_ids)
        return best_scores, out_ids

    def _csr_task_budget(self) -> int:
        if self.device.type == "cpu":
            return 20_000
        return 200_000

    def _bytes_per_vector(self) -> int:
        elem_size = torch.tensor([], dtype=self.dtype).element_size()
        return max(1, int(self.d) * int(elem_size))

    def _csr_vec_chunk(self, *, buffered: bool) -> int:
        if self.device.type == "cpu":
            target_bytes = 8 * 1024 * 1024 if buffered else 4 * 1024 * 1024
        else:
            target_bytes = 32 * 1024 * 1024 if buffered else 16 * 1024 * 1024
        bytes_per_vec = self._bytes_per_vector()
        return max(1, int(target_bytes // bytes_per_vec))

    def _iter_query_chunks_csr(self, nq: int) -> list[tuple[int, int]]:
        if nq <= 0:
            return [(0, 0)]
        budget = self._csr_task_budget()
        probe = max(1, min(self._nprobe, self._nlist))
        chunk = max(1, budget // probe)
        chunk = min(chunk, nq)
        return [(i, min(nq, i + chunk)) for i in range(0, nq, chunk)]

    def _candidate_budget(self) -> int:
        bytes_per_vec = self._bytes_per_vector()
        if self.device.type == "cpu":
            target_bytes = 512 * 1024 * 1024
        else:
            target_bytes = 256 * 1024 * 1024
            if self.device.type == "cuda" and torch.cuda.is_available() and hasattr(torch.cuda, "mem_get_info"):
                try:
                    free_mem, _ = torch.cuda.mem_get_info(self.device)
                    target_bytes = min(target_bytes, int(free_mem * 0.1))
                except RuntimeError:
                    pass
        target_bytes = max(target_bytes, bytes_per_vec)
        return max(1, int(target_bytes // bytes_per_vec))

    def _candidate_block_size(self, max_candidates: int) -> int:
        if self.device.type == "cpu":
            return max_candidates
        return 4096

    def _iter_query_chunks(self, query_candidate_counts: torch.Tensor) -> list[tuple[int, int]]:
        budget = self._candidate_budget()
        counts = query_candidate_counts.to("cpu").tolist()
        total = len(counts)
        if total == 0:
            return [(0, 0)]

        chunks: list[tuple[int, int]] = []
        start = 0
        running = 0
        for i, c in enumerate(counts):
            c_int = int(c)
            if i == start:
                running = c_int
                continue
            if running + c_int > budget:
                chunks.append((start, i))
                start = i
                running = c_int
            else:
                running += c_int
        chunks.append((start, total))
        return chunks

    def _search_from_candidates(
        self,
        xq_chunk: torch.Tensor,
        cand_vecs: torch.Tensor,
        cand_ids: torch.Tensor,
        cand_query_ids: torch.Tensor,
        query_counts: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        chunk_size = xq_chunk.shape[0]
        largest = self.metric == "ip"
        fill = float("inf") if not largest else float("-inf")

        if cand_vecs.numel() == 0 or int(query_counts.max().item()) == 0:
            dists = torch.full((chunk_size, k), fill, dtype=self.dtype, device=self.device)
            labels = torch.full((chunk_size, k), -1, dtype=torch.long, device=self.device)
            if largest:
                dists.fill_(float("-inf"))
            return dists, labels

        scores = self._compute_candidate_scores(cand_vecs, cand_query_ids, xq_chunk)
        max_candidates = int(query_counts.max().item())

        score_matrix = torch.full(
            (chunk_size, max_candidates),
            fill,
            dtype=self.dtype,
            device=self.device,
        )
        label_matrix = torch.full(
            (chunk_size, max_candidates),
            -1,
            dtype=torch.long,
            device=self.device,
        )

        prefix = torch.cumsum(
            torch.cat(
                [torch.zeros(1, dtype=torch.long, device=self.device), query_counts[:-1]]
            ),
            dim=0,
        )
        positions = torch.arange(cand_vecs.shape[0], dtype=torch.long, device=self.device) - prefix[cand_query_ids]
        score_matrix[cand_query_ids, positions] = scores
        label_matrix[cand_query_ids, positions] = cand_ids

        topk = min(k, score_matrix.shape[1]) if score_matrix.shape[1] > 0 else 0
        if topk > 0:
            dists, idx = torch.topk(score_matrix, topk, largest=largest, dim=1)
            labels = torch.gather(label_matrix, 1, idx)
        else:
            dists = torch.full((chunk_size, 0), fill, dtype=self.dtype, device=self.device)
            labels = torch.empty((chunk_size, 0), dtype=torch.long, device=self.device)

        if topk < k:
            pad_cols = k - topk
            dists = torch.cat(
                [dists, torch.full((chunk_size, pad_cols), fill, dtype=self.dtype, device=self.device)],
                dim=1,
            )
            labels = torch.cat(
                [labels, torch.full((chunk_size, pad_cols), -1, dtype=torch.long, device=self.device)],
                dim=1,
            )

        return dists, labels

    def _search_from_index_matrix(
        self,
        xq_chunk: torch.Tensor,
        index_matrix: torch.Tensor,
        query_counts: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        chunk_size = xq_chunk.shape[0]
        largest = self.metric == "ip"
        fill = float("inf") if not largest else float("-inf")

        dists = torch.full((chunk_size, k), fill, dtype=self.dtype, device=self.device)
        labels = torch.full((chunk_size, k), -1, dtype=torch.long, device=self.device)
        if chunk_size == 0:
            return dists, labels
        if index_matrix.numel() == 0 or index_matrix.shape[1] == 0:
            return dists, labels

        pad_mask = index_matrix < 0
        idx = index_matrix.clamp_min(0)
        max_candidates = int(index_matrix.shape[1])
        if max_candidates <= 0:
            return dists, labels

        q = xq_chunk if xq_chunk.dtype == self.dtype else xq_chunk.to(self.dtype)
        q = q.contiguous()

        # Prefer the full-matrix path for speed; fall back to block streaming when the
        # candidate tensor would be extremely large (avoid peak memory blow-ups).
        elem_size = torch.tensor([], dtype=self.dtype).element_size()
        cand_bytes = int(chunk_size) * int(max_candidates) * int(self.d) * int(elem_size)
        use_streaming = self.device.type != "cpu" and cand_bytes >= 512 * 1024 * 1024

        if not use_streaming:
            idx_flat = idx.reshape(-1)
            cand_vecs = self._packed_embeddings.index_select(0, idx_flat).reshape(chunk_size, max_candidates, self.d)
            dot = torch.bmm(cand_vecs, q.unsqueeze(2)).squeeze(2)
            if self.metric == "l2":
                cand_norms = self._packed_norms.index_select(0, idx_flat).reshape(chunk_size, max_candidates)
                q_norm = (q * q).sum(dim=1, keepdim=True)
                scores = cand_norms + q_norm - (2.0 * dot)
                scores = scores.clamp_min_(0)
                scores.masked_fill_(pad_mask, float("inf"))
                top_vals, top_idx = torch.topk(scores, min(k, scores.shape[1]), largest=False, dim=1)
            else:
                scores = dot
                scores.masked_fill_(pad_mask, float("-inf"))
                top_vals, top_idx = torch.topk(scores, min(k, scores.shape[1]), largest=True, dim=1)

            top_packed_idx = torch.gather(idx, 1, top_idx)
            top_labels = self._list_ids.index_select(0, top_packed_idx.reshape(-1)).reshape(top_packed_idx.shape)
            top_labels = torch.where(top_packed_idx < 0, torch.full_like(top_labels, -1), top_labels)
            if top_idx.shape[1] < k:
                pad_cols = k - top_idx.shape[1]
                top_vals = torch.cat(
                    [top_vals, torch.full((chunk_size, pad_cols), fill, dtype=self.dtype, device=self.device)],
                    dim=1,
                )
                top_labels = torch.cat(
                    [top_labels, torch.full((chunk_size, pad_cols), -1, dtype=torch.long, device=self.device)],
                    dim=1,
                )
            return top_vals, top_labels

        if self.metric == "l2":
            q_norm = (q * q).sum(dim=1, keepdim=True)
        else:
            q_norm = None

        best_scores = torch.full((chunk_size, k), fill, dtype=self.dtype, device=self.device)
        best_packed_idx = torch.full((chunk_size, k), -1, dtype=torch.long, device=self.device)
        block = self._candidate_block_size(max_candidates)
        for col_start in range(0, max_candidates, block):
            col_end = min(max_candidates, col_start + block)
            idx_block = idx[:, col_start:col_end]
            pad_block = pad_mask[:, col_start:col_end]
            idx_flat = idx_block.reshape(-1)
            cand_vecs = self._packed_embeddings.index_select(0, idx_flat).reshape(
                chunk_size, col_end - col_start, self.d
            )
            dot = torch.bmm(cand_vecs, q.unsqueeze(2)).squeeze(2)
            if self.metric == "l2":
                cand_norms = self._packed_norms.index_select(0, idx_flat).reshape(chunk_size, col_end - col_start)
                scores = cand_norms + q_norm - (2.0 * dot)
                scores = scores.clamp_min_(0)
                scores.masked_fill_(pad_block, float("inf"))
                top_vals, top_idx = torch.topk(scores, min(k, scores.shape[1]), largest=False, dim=1)
            else:
                scores = dot
                scores.masked_fill_(pad_block, float("-inf"))
                top_vals, top_idx = torch.topk(scores, min(k, scores.shape[1]), largest=True, dim=1)

            top_packed = torch.gather(idx_block, 1, top_idx)
            merged_scores = torch.cat([best_scores, top_vals], dim=1)
            merged_idx = torch.cat([best_packed_idx, top_packed], dim=1)
            best_scores, best_pos = torch.topk(merged_scores, k, largest=largest, dim=1)
            best_packed_idx = torch.gather(merged_idx, 1, best_pos)

        best_labels = self._list_ids.index_select(0, best_packed_idx.clamp_min(0).reshape(-1)).reshape(
            best_packed_idx.shape
        )
        best_labels = torch.where(best_packed_idx < 0, torch.full_like(best_labels, -1), best_labels)
        return best_scores, best_labels

    def _range_from_candidates(
        self,
        xq_chunk: torch.Tensor,
        cand_vecs: torch.Tensor,
        cand_ids: torch.Tensor,
        cand_query_ids: torch.Tensor,
        query_counts: torch.Tensor,
        radius: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chunk_size = xq_chunk.shape[0]
        lims = torch.zeros(chunk_size + 1, dtype=torch.long, device=self.device)
        if cand_vecs.numel() == 0 or int(query_counts.max().item()) == 0:
            return lims, torch.empty(0, dtype=self.dtype, device=self.device), torch.empty(0, dtype=torch.long, device=self.device)

        scores = self._compute_candidate_scores(cand_vecs, cand_query_ids, xq_chunk)
        if self.metric == "l2":
            mask = scores <= radius
        else:
            mask = scores >= radius

        if not mask.any():
            return lims, torch.empty(0, dtype=self.dtype, device=self.device), torch.empty(0, dtype=torch.long, device=self.device)

        selected_scores = scores[mask]
        selected_ids = cand_ids[mask]
        selected_queries = cand_query_ids[mask]
        ones = torch.ones_like(selected_queries, dtype=torch.long)
        hit_counts = torch.zeros(chunk_size, dtype=torch.long, device=self.device)
        hit_counts.scatter_add_(0, selected_queries, ones)
        lims[1:] = torch.cumsum(hit_counts, dim=0)
        return lims, selected_scores, selected_ids

    def _range_from_index_matrix(
        self,
        xq_chunk: torch.Tensor,
        index_matrix: torch.Tensor,
        query_counts: torch.Tensor,
        radius: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        chunk_size = xq_chunk.shape[0]
        lims = torch.zeros(chunk_size + 1, dtype=torch.long, device=self.device)
        if chunk_size == 0:
            return lims, torch.empty(0, dtype=self.dtype, device=self.device), torch.empty(
                0, dtype=torch.long, device=self.device
            )
        if index_matrix.numel() == 0 or index_matrix.shape[1] == 0:
            return lims, torch.empty(0, dtype=self.dtype, device=self.device), torch.empty(
                0, dtype=torch.long, device=self.device
            )

        pad_mask = index_matrix < 0
        idx = index_matrix.clamp_min(0)
        max_candidates = int(index_matrix.shape[1])
        if max_candidates <= 0:
            return lims, torch.empty(0, dtype=self.dtype, device=self.device), torch.empty(
                0, dtype=torch.long, device=self.device
            )

        q = xq_chunk if xq_chunk.dtype == self.dtype else xq_chunk.to(self.dtype)
        q = q.contiguous()
        if self.metric == "l2":
            q_norm = (q * q).sum(dim=1, keepdim=True)
        else:
            q_norm = None

        hit_counts = torch.zeros(chunk_size, dtype=torch.long, device=self.device)
        values_list: list[torch.Tensor] = []
        packed_list: list[torch.Tensor] = []
        block = self._candidate_block_size(max_candidates)
        for col_start in range(0, max_candidates, block):
            col_end = min(max_candidates, col_start + block)
            idx_block = idx[:, col_start:col_end]
            pad_block = pad_mask[:, col_start:col_end]
            idx_flat = idx_block.reshape(-1)
            cand_vecs = self._packed_embeddings.index_select(0, idx_flat).reshape(
                chunk_size, col_end - col_start, self.d
            )
            dot = torch.bmm(cand_vecs, q.unsqueeze(2)).squeeze(2)
            if self.metric == "l2":
                cand_norms = self._packed_norms.index_select(0, idx_flat).reshape(chunk_size, col_end - col_start)
                scores = (cand_norms + q_norm - (2.0 * dot)).clamp_min_(0)
                hit = scores <= radius
            else:
                scores = dot
                hit = scores >= radius

            valid = hit & (~pad_block)
            hit_counts += valid.sum(dim=1, dtype=torch.long)
            if valid.any():
                values_list.append(scores[valid])
                packed_list.append(idx_block[valid])

        lims[1:] = torch.cumsum(hit_counts, dim=0)
        if not values_list:
            return lims, torch.empty(0, dtype=self.dtype, device=self.device), torch.empty(
                0, dtype=torch.long, device=self.device
            )
        values = torch.cat(values_list)
        packed_idx = torch.cat(packed_list)
        ids = self._list_ids.index_select(0, packed_idx.reshape(-1)).reshape(packed_idx.shape)
        return lims, values, ids

    def _collect_candidate_vectors(
        self, xq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if xq.shape[0] == 0 or self._centroids.shape[0] == 0:
            empty_vecs = torch.empty((0, self.d), dtype=self.dtype, device=self.device)
            empty_ids = torch.empty(0, dtype=torch.long, device=self.device)
            empty_queries = torch.empty(0, dtype=torch.long, device=self.device)
            counts = torch.zeros(xq.shape[0], dtype=torch.long, device=self.device)
            return empty_vecs, empty_ids, empty_queries, counts

        centroid_scores = self._pairwise_centroids(xq)
        probe = min(self._nprobe, centroid_scores.shape[1])
        if probe == 0:
            empty_vecs = torch.empty((0, self.d), dtype=self.dtype, device=self.device)
            empty_ids = torch.empty(0, dtype=torch.long, device=self.device)
            empty_queries = torch.empty(0, dtype=torch.long, device=self.device)
            counts = torch.zeros(xq.shape[0], dtype=torch.long, device=self.device)
            return empty_vecs, empty_ids, empty_queries, counts

        largest = self.metric == "ip"
        _, top_lists = torch.topk(centroid_scores, probe, largest=largest, dim=1)

        flat_lists = top_lists.reshape(-1)
        starts = self._list_offsets[flat_lists]
        ends = self._list_offsets[flat_lists + 1]
        sizes = ends - starts
        nonzero = sizes > 0
        if not nonzero.any():
            empty_vecs = torch.empty((0, self.d), dtype=self.dtype, device=self.device)
            empty_ids = torch.empty(0, dtype=torch.long, device=self.device)
            empty_queries = torch.empty(0, dtype=torch.long, device=self.device)
            counts = torch.zeros(xq.shape[0], dtype=torch.long, device=self.device)
            return empty_vecs, empty_ids, empty_queries, counts

        nz_sizes = sizes[nonzero]
        total = int(nz_sizes.sum().item())
        if total == 0:
            empty_vecs = torch.empty((0, self.d), dtype=self.dtype, device=self.device)
            empty_ids = torch.empty(0, dtype=torch.long, device=self.device)
            empty_queries = torch.empty(0, dtype=torch.long, device=self.device)
            counts = torch.zeros(xq.shape[0], dtype=torch.long, device=self.device)
            return empty_vecs, empty_ids, empty_queries, counts

        nz_starts = starts[nonzero]
        probe_actual = top_lists.shape[1]
        query_ids = (
            torch.arange(xq.shape[0], device=self.device)
            .unsqueeze(1)
            .expand(-1, probe_actual)
            .reshape(-1)
        )
        nz_query_ids = query_ids[nonzero]

        offsets = torch.cumsum(
            torch.cat(
                [torch.zeros(1, dtype=torch.long, device=self.device), nz_sizes[:-1]]
            ),
            dim=0,
        )
        repeated_offsets = torch.repeat_interleave(offsets, nz_sizes)
        repeated_starts = torch.repeat_interleave(nz_starts, nz_sizes)

        arange_total = torch.arange(total, dtype=torch.long, device=self.device)
        candidate_indices = repeated_starts + arange_total - repeated_offsets

        cand_vecs = self._packed_embeddings[candidate_indices]
        cand_ids = self._list_ids[candidate_indices]
        cand_query_ids = torch.repeat_interleave(nz_query_ids, nz_sizes)

        query_counts = torch.zeros(xq.shape[0], dtype=torch.long, device=self.device)
        if cand_query_ids.numel() > 0:
            ones = torch.ones_like(cand_query_ids, dtype=torch.long)
            query_counts.scatter_add_(0, cand_query_ids, ones)

        return cand_vecs, cand_ids, cand_query_ids, query_counts

    def _collect_candidate_vectors_from_lists(
        self, top_lists: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        chunk = top_lists.shape[0]
        if chunk == 0 or top_lists.numel() == 0:
            return (
                torch.empty((0, self.d), dtype=self.dtype, device=self.device),
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.empty(0, dtype=self.dtype, device=self.device),
                torch.zeros(chunk, dtype=torch.long, device=self.device),
            )

        probe = top_lists.shape[1]
        flat_lists = top_lists.reshape(-1)
        starts = self._list_offsets[flat_lists]
        ends = self._list_offsets[flat_lists + 1]
        sizes = (ends - starts).reshape(chunk, probe)
        if self._max_codes:
            budget = torch.full((chunk, 1), self._max_codes, dtype=torch.long, device=self.device)
            prev_cum = torch.cumsum(sizes, dim=1) - sizes
            remaining = (budget - prev_cum).clamp_min(0)
            sizes = torch.minimum(sizes, remaining)
        query_counts = sizes.sum(dim=1)

        flat_sizes = sizes.reshape(-1)
        nonzero = flat_sizes > 0
        if not nonzero.any():
            return (
                torch.empty((0, self.d), dtype=self.dtype, device=self.device),
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.empty(0, dtype=self.dtype, device=self.device),
                query_counts,
            )

        nz_sizes = flat_sizes[nonzero]
        total = int(nz_sizes.sum().item())
        if total == 0:
            return (
                torch.empty((0, self.d), dtype=self.dtype, device=self.device),
                torch.empty(0, dtype=torch.long, device=self.device),
                torch.empty(0, dtype=self.dtype, device=self.device),
                query_counts,
            )

        nz_starts = starts[nonzero]
        query_ids = (
            torch.arange(chunk, device=self.device)
            .unsqueeze(1)
            .expand(-1, probe)
            .reshape(-1)
        )
        nz_query_ids = query_ids[nonzero]

        offsets = torch.cumsum(
            torch.cat([torch.zeros(1, dtype=torch.long, device=self.device), nz_sizes[:-1]]),
            dim=0,
        )
        repeated_offsets = torch.repeat_interleave(offsets, nz_sizes)
        repeated_starts = torch.repeat_interleave(nz_starts, nz_sizes)
        arange_total = torch.arange(total, dtype=torch.long, device=self.device)
        candidate_indices = repeated_starts + arange_total - repeated_offsets

        cand_vecs = self._packed_embeddings[candidate_indices]
        cand_ids = self._list_ids[candidate_indices]
        cand_norms = self._packed_norms[candidate_indices]
        return cand_vecs, cand_ids, cand_norms, query_counts

    def _compute_candidate_scores(
        self, cand_vecs: torch.Tensor, cand_query_ids: torch.Tensor, xq: torch.Tensor
    ) -> torch.Tensor:
        query_vecs = xq[cand_query_ids]
        if self.metric == "l2":
            diff = query_vecs - cand_vecs
            return (diff * diff).sum(dim=1)
        return (query_vecs * cand_vecs).sum(dim=1)

    def _pairwise_centroids(self, x: torch.Tensor) -> torch.Tensor:
        if self._centroids.shape[0] == 0:
            return torch.empty(x.shape[0], 0, dtype=self.dtype, device=self.device)
        self._ensure_search_cache()
        centroids_t = self._centroids_t
        if centroids_t is None:
            return torch.empty(x.shape[0], 0, dtype=self.dtype, device=self.device)
        x_cast = x.to(self.dtype)
        if self.metric == "l2":
            x_norm = (x_cast * x_cast).sum(dim=1, keepdim=True)
            y_norm = self._centroid_norm2
            if y_norm is None:
                return torch.empty(x.shape[0], 0, dtype=self.dtype, device=self.device)
            dist = x_norm + y_norm.unsqueeze(0) - (2.0 * (x_cast @ centroids_t))
            return dist.clamp_min_(0)
        return x_cast @ centroids_t

    def _pairwise(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if y.shape[0] == 0:
            return torch.empty(x.shape[0], 0, dtype=self.dtype, device=self.device)
        x_cast = x.to(self.dtype)
        y_cast = y.to(self.dtype)
        if self.metric == "l2":
            x_norm = (x_cast * x_cast).sum(dim=1, keepdim=True)
            y_norm = (y_cast * y_cast).sum(dim=1).unsqueeze(0)
            dist = x_norm + y_norm - (2.0 * (x_cast @ y_cast.t()))
            return dist.clamp_min_(0)
        return x_cast @ y_cast.t()

    def _tensor_attributes(self):
        return (
            "_centroids",
            "_packed_embeddings",
            "_packed_norms",
            "_list_ids",
            "_list_offsets",
            "_centroids_t",
            "_centroid_norm2",
            "_list_sizes",
        )
