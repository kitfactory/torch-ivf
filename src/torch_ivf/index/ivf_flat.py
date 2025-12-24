from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Literal, Optional

import torch

from ..nn import kmeans
from .base import IndexBase, MetricType


@dataclass(frozen=True)
class SearchParams:
    profile: Literal["exact", "speed", "approx"] = "speed"
    safe_pruning: bool = True
    approximate: bool = False
    nprobe: int | None = None
    max_codes: int | None = None
    candidate_budget: int | None = None
    budget_strategy: Literal["uniform", "distance_weighted"] = "distance_weighted"
    list_ordering: Literal["none", "residual_norm_asc", "proj_desc"] | None = None
    rebuild_policy: Literal["manual", "auto_threshold"] = "manual"
    rebuild_threshold_adds: int = 0
    dynamic_nprobe: bool = False
    min_codes_per_list: int = 0
    max_codes_cap_per_list: int = 0
    strict_budget: bool = False
    use_per_list_sizes: bool = False
    debug_stats: bool = False


@dataclass(frozen=True)
class _ResolvedSearchConfig:
    profile: str
    safe_pruning: bool
    approximate: bool
    nprobe: int
    max_codes: int
    candidate_budget: int | None
    budget_strategy: str
    list_ordering: str | None
    rebuild_policy: str
    rebuild_threshold_adds: int
    dynamic_nprobe: bool
    min_codes_per_list: int
    max_codes_cap_per_list: int
    strict_budget: bool
    use_per_list_sizes: bool
    debug_stats: bool


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
        self._list_ordering: str | None = None
        self._rebuild_policy = "manual"
        self._rebuild_threshold_adds = 0
        self._adds_since_rebuild = 0
        self._last_search_stats: dict[str, float | int | str] | None = None
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

    @property
    def last_search_stats(self) -> dict[str, float | int | str] | None:
        return self._last_search_stats

    def to(self, device: torch.device | str | None) -> "IndexIVFFlat":
        if device is None:
            return self
        cloned = super().to(device)
        cloned._invalidate_search_cache()
        return cloned

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
        self._list_ordering = None
        self._adds_since_rebuild = 0
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
        self._adds_since_rebuild += int(xb.shape[0])

    def add_with_ids(self, xb: torch.Tensor, ids: torch.Tensor) -> None:
        self._ensure_trained()
        xb = self._validate_input(xb)
        if ids.ndim != 1 or ids.shape[0] != xb.shape[0]:
            raise ValueError("ids must be 1-D with the same length as xb.")
        if ids.dtype not in {torch.long, torch.int64}:
            raise ValueError("ids must be torch.long.")
        self._assign_and_append(xb, ids.to(self.device))
        self._adds_since_rebuild += int(xb.shape[0])

    def search(
        self, xq: torch.Tensor, k: int, *, params: SearchParams | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self._ensure_ready_for_search(k)
        xq = self._validate_input(xq)
        if xq.shape[0] == 0:
            return (
                torch.empty(0, k, dtype=self.dtype, device=self.device),
                torch.empty(0, k, dtype=torch.long, device=self.device),
            )

        config = self._resolve_search_params(params)
        if config.list_ordering is not None:
            self._maybe_rebuild_lists(config)

        top_lists: torch.Tensor
        per_list_sizes: torch.Tensor | None = None
        effective_max_codes = self._effective_max_codes_for(config.nprobe, config.max_codes)
        debug_stats: dict[str, float | int | str] | None = None
        if config.debug_stats:
            debug_stats = {
                "nq": int(xq.shape[0]),
                "nprobe_user": int(config.nprobe),
                "candidate_budget": int(config.candidate_budget or 0),
                "approximate": int(config.approximate),
                "use_per_list_sizes": int(config.use_per_list_sizes),
            }
        else:
            self._last_search_stats = None

        if config.approximate and config.candidate_budget is not None:
            if config.use_per_list_sizes:
                top_lists, top_scores = self._top_probed_lists_with_scores(xq, nprobe=config.nprobe)
                per_list_sizes = self._allocate_candidate_sizes(top_lists, top_scores, config)
                effective_max_codes = 0
                if debug_stats is not None:
                    per_list_cap = self._budget_to_max_codes(
                        config.candidate_budget, config.nprobe, per_list=True
                    )
                    debug_stats["budget_path"] = "per_list"
                    debug_stats["per_list_cap"] = int(per_list_cap)
            else:
                top_lists = self._top_probed_lists(xq, nprobe=config.nprobe)
                max_codes_budget = self._budget_to_max_codes(
                    config.candidate_budget, config.nprobe, per_list=False
                )
                max_codes_eff = self._min_positive(config.max_codes, max_codes_budget)
                effective_max_codes = self._effective_max_codes_for(config.nprobe, max_codes_eff)
                if debug_stats is not None:
                    debug_stats["budget_path"] = "max_codes"
                    debug_stats["max_codes_from_budget"] = int(max_codes_budget)
        else:
            top_lists = self._top_probed_lists(xq, nprobe=config.nprobe)

        if self._search_mode == "csr":
            results = self._search_csr_online(
                xq,
                k,
                max_codes=effective_max_codes,
                nprobe=config.nprobe,
                top_lists=top_lists,
                per_list_sizes=per_list_sizes,
                debug_stats=debug_stats,
            )
            if debug_stats is not None:
                debug_stats["search_path"] = "csr"
                debug_stats["nprobe_eff"] = int(config.nprobe)
                debug_stats["max_codes_eff"] = int(effective_max_codes)
                self._last_search_stats = debug_stats
            return results

        if self._search_mode == "auto" and xq.device.type == "cuda":
            probe = min(config.nprobe, self._nlist)
            avg_group_size = (xq.shape[0] * probe) / max(1, self._nlist)
            auto_threshold = self._auto_search_avg_group_threshold * (self._nlist / 512)
            if avg_group_size >= auto_threshold:
                results = self._search_csr_online(
                    xq,
                    k,
                    max_codes=effective_max_codes,
                    nprobe=config.nprobe,
                    top_lists=top_lists,
                    per_list_sizes=per_list_sizes,
                    debug_stats=debug_stats,
                )
                if debug_stats is not None:
                    debug_stats["search_path"] = "csr"
                    debug_stats["nprobe_eff"] = int(config.nprobe)
                    debug_stats["max_codes_eff"] = int(effective_max_codes)
                    self._last_search_stats = debug_stats
                return results

        query_candidate_counts = self._estimate_candidates_per_query(
            top_lists, max_codes=effective_max_codes, per_list_sizes=per_list_sizes
        )
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
            chunk_sizes = per_list_sizes[start:end] if per_list_sizes is not None else None
            index_matrix, query_counts = self._build_candidate_index_matrix_from_lists(
                chunk_lists,
                max_candidates=max_candidates,
                pos_1d=pos_1d,
                max_codes=effective_max_codes,
                per_list_sizes=chunk_sizes,
            )
            chunk_dists, chunk_labels = self._search_from_index_matrix(chunk_q, index_matrix, query_counts, k)
            dists[start:end] = chunk_dists
            labels[start:end] = chunk_labels

        if debug_stats is not None:
            debug_stats["search_path"] = "matrix"
            debug_stats["nprobe_eff"] = int(config.nprobe)
            debug_stats["max_codes_eff"] = int(effective_max_codes)
            debug_stats["chunks"] = int(len(chunks))
            debug_stats["total_candidates_est"] = int(query_candidate_counts.sum().item())
            self._last_search_stats = debug_stats
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
        self._list_ordering = None
        self._adds_since_rebuild = 0
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
        self._list_ordering = None
        self._adds_since_rebuild = 0
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

    def _resolve_search_params(self, params: SearchParams | None) -> _ResolvedSearchConfig:
        if params is None:
            profile = "speed"
            safe_pruning = True
            approximate = bool(self._approximate_mode)
            nprobe = self._nprobe
            max_codes = self._max_codes
            candidate_budget = None
            budget_strategy = "distance_weighted"
            list_ordering = None
            rebuild_policy = "manual"
            rebuild_threshold_adds = 0
            dynamic_nprobe = False
            min_codes_per_list = 0
            max_codes_cap_per_list = 0
            strict_budget = False
            use_per_list_sizes = False
            debug_stats = False
        else:
            profile = params.profile
            safe_pruning = bool(params.safe_pruning)
            approximate = bool(params.approximate)
            nprobe = self._nprobe if params.nprobe is None else int(params.nprobe)
            max_codes = self._max_codes if params.max_codes is None else int(params.max_codes)
            candidate_budget = params.candidate_budget
            budget_strategy = params.budget_strategy
            list_ordering = params.list_ordering
            rebuild_policy = params.rebuild_policy
            rebuild_threshold_adds = params.rebuild_threshold_adds
            dynamic_nprobe = bool(params.dynamic_nprobe)
            min_codes_per_list = int(params.min_codes_per_list)
            max_codes_cap_per_list = int(params.max_codes_cap_per_list)
            strict_budget = bool(params.strict_budget)
            use_per_list_sizes = bool(params.use_per_list_sizes)
            debug_stats = bool(params.debug_stats)

        if profile not in {"exact", "speed", "approx"}:
            raise ValueError("profile must be 'exact', 'speed', or 'approx'.")
        if nprobe <= 0:
            raise ValueError("nprobe must be >= 1.")
        if max_codes < 0:
            raise ValueError("max_codes must be >= 0.")
        if candidate_budget is not None:
            candidate_budget = int(candidate_budget)
        if candidate_budget is not None and candidate_budget < 0:
            raise ValueError("candidate_budget must be >= 0.")
        if budget_strategy not in {"uniform", "distance_weighted"}:
            raise ValueError("budget_strategy must be 'uniform' or 'distance_weighted'.")
        if list_ordering not in {None, "none", "residual_norm_asc", "proj_desc"}:
            raise ValueError("list_ordering must be None or one of 'none', 'residual_norm_asc', 'proj_desc'.")
        if rebuild_policy not in {"manual", "auto_threshold"}:
            raise ValueError("rebuild_policy must be 'manual' or 'auto_threshold'.")
        if rebuild_threshold_adds < 0:
            raise ValueError("rebuild_threshold_adds must be >= 0.")
        if min_codes_per_list < 0:
            raise ValueError("min_codes_per_list must be >= 0.")
        if max_codes_cap_per_list < 0:
            raise ValueError("max_codes_cap_per_list must be >= 0.")

        if candidate_budget is not None and candidate_budget == 0:
            candidate_budget = None

        if list_ordering == "none":
            list_ordering = None

        nprobe = min(nprobe, self._nlist)

        if approximate and candidate_budget is not None and self.metric != "l2":
            raise NotImplementedError("Approximate mode is only supported for L2 in Phase P1.")
        if list_ordering is not None and self.metric != "l2":
            raise NotImplementedError("list_ordering is only supported for L2 in Phase P1.")
        if list_ordering == "proj_desc" and self.metric == "l2":
            raise ValueError("proj_desc is only supported for IP in Phase 2.")
        if self.metric != "l2":
            safe_pruning = False

        if candidate_budget is not None and not approximate:
            raise ValueError("candidate_budget requires approximate=True.")

        if dynamic_nprobe and budget_strategy != "distance_weighted":
            dynamic_nprobe = False
        if dynamic_nprobe and candidate_budget is None:
            dynamic_nprobe = False

        if use_per_list_sizes and candidate_budget is None:
            raise ValueError("use_per_list_sizes requires candidate_budget.")
        if (strict_budget or min_codes_per_list > 0 or max_codes_cap_per_list > 0 or dynamic_nprobe) and not use_per_list_sizes:
            raise ValueError("per-list budgeting requires use_per_list_sizes=True.")

        if approximate and list_ordering is None and self.metric == "l2":
            list_ordering = "residual_norm_asc"

        return _ResolvedSearchConfig(
            profile=profile,
            safe_pruning=safe_pruning,
            approximate=approximate,
            nprobe=nprobe,
            max_codes=max_codes,
            candidate_budget=candidate_budget,
            budget_strategy=budget_strategy,
            list_ordering=list_ordering,
            rebuild_policy=rebuild_policy,
            rebuild_threshold_adds=rebuild_threshold_adds,
            dynamic_nprobe=dynamic_nprobe,
            min_codes_per_list=min_codes_per_list,
            max_codes_cap_per_list=max_codes_cap_per_list,
            strict_budget=strict_budget,
            use_per_list_sizes=use_per_list_sizes,
            debug_stats=debug_stats,
        )

    def _maybe_rebuild_lists(self, config: _ResolvedSearchConfig) -> None:
        if config.list_ordering is None:
            return
        if config.rebuild_policy == "auto_threshold":
            threshold = int(config.rebuild_threshold_adds)
            if self._list_ordering != config.list_ordering:
                self.rebuild_lists(ordering=config.list_ordering)
                return
            if threshold > 0 and self._adds_since_rebuild >= threshold:
                self.rebuild_lists(ordering=config.list_ordering)

    def rebuild_lists(self, *, ordering: str | None = None) -> None:
        if ordering is None:
            ordering = self._list_ordering or "residual_norm_asc"
        if ordering not in {"residual_norm_asc"}:
            raise ValueError("ordering must be 'residual_norm_asc'.")
        if self.metric != "l2":
            raise NotImplementedError("rebuild_lists is only supported for L2 in Phase P1.")
        if self._ntotal == 0:
            self._list_ordering = ordering
            self._adds_since_rebuild = 0
            return

        offsets_cpu = self._list_offsets.to("cpu").tolist()
        for list_id in range(self._nlist):
            a = int(offsets_cpu[list_id])
            b = int(offsets_cpu[list_id + 1])
            if b - a <= 1:
                continue
            x = self._packed_embeddings[a:b]
            c = self._centroids[list_id].to(self.dtype)
            diff = x - c
            key = (diff * diff).sum(dim=1)
            order = torch.argsort(key)
            self._packed_embeddings[a:b] = x.index_select(0, order)
            self._packed_norms[a:b] = self._packed_norms[a:b].index_select(0, order)
            self._list_ids[a:b] = self._list_ids[a:b].index_select(0, order)

        self._list_ordering = ordering
        self._adds_since_rebuild = 0

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

    def _top_probed_lists(self, xq: torch.Tensor, *, nprobe: int | None = None) -> torch.Tensor:
        if xq.shape[0] == 0 or self._centroids.shape[0] == 0:
            return torch.empty((xq.shape[0], 0), dtype=torch.long, device=self.device)
        centroid_scores = self._pairwise_centroids(xq)
        probe_in = self._nprobe if nprobe is None else int(nprobe)
        probe = min(probe_in, centroid_scores.shape[1])
        if probe <= 0:
            return torch.empty((xq.shape[0], 0), dtype=torch.long, device=self.device)
        largest = self.metric == "ip"
        _, top_lists = torch.topk(centroid_scores, probe, largest=largest, dim=1)
        return top_lists

    def _top_probed_lists_with_scores(
        self, xq: torch.Tensor, *, nprobe: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if xq.shape[0] == 0 or self._centroids.shape[0] == 0:
            empty = torch.empty((xq.shape[0], 0), dtype=torch.long, device=self.device)
            return empty, torch.empty((xq.shape[0], 0), dtype=self.dtype, device=self.device)
        centroid_scores = self._pairwise_centroids(xq)
        probe = min(int(nprobe), centroid_scores.shape[1])
        if probe <= 0:
            empty = torch.empty((xq.shape[0], 0), dtype=torch.long, device=self.device)
            return empty, torch.empty((xq.shape[0], 0), dtype=self.dtype, device=self.device)
        largest = self.metric == "ip"
        top_scores, top_lists = torch.topk(centroid_scores, probe, largest=largest, dim=1)
        return top_lists, top_scores

    def _effective_max_codes(self) -> int:
        return self._effective_max_codes_for(self._nprobe, self._max_codes)

    def _effective_max_codes_for(self, nprobe: int, max_codes: int) -> int:
        max_codes_i = int(max_codes)
        if max_codes_i <= 0:
            return 0

        use_cache = nprobe == self._nprobe and max_codes_i == self._max_codes
        cached = self._effective_max_codes_cache if use_cache else None
        if cached is not None:
            return cached

        probe = min(int(nprobe), self._nlist)
        if probe <= 0:
            return 0

        self._ensure_search_cache()
        sizes = self._list_sizes_cpu or []
        if probe >= self._nlist:
            max_possible = sum(sizes)
        else:
            max_possible = sum(heapq.nlargest(probe, sizes))

        effective = 0 if max_codes_i >= max_possible else max_codes_i
        if use_cache:
            self._effective_max_codes_cache = effective
        return effective

    @staticmethod
    def _budget_to_max_codes(candidate_budget: int, nprobe: int, *, per_list: bool) -> int:
        budget = int(candidate_budget)
        if budget <= 0:
            return 0
        if not per_list:
            return budget
        probe = max(1, int(nprobe))
        return (budget + probe - 1) // probe

    @staticmethod
    def _min_positive(a: int, b: int) -> int:
        if a <= 0:
            return b
        if b <= 0:
            return a
        return min(a, b)

    def _distance_weights(self, distances: torch.Tensor) -> torch.Tensor:
        if distances.numel() == 0:
            return distances
        dist = distances.to(torch.float32)
        d_min = dist.min(dim=1, keepdim=True).values
        d_med = dist.median(dim=1, keepdim=True).values
        denom = (d_med - d_min).clamp_min(1e-6)
        z = (dist - d_min) / denom
        z = z.clamp(0.0, 8.0)
        return torch.exp(-3.0 * z)

    def _enforce_strict_budget(
        self, alloc: torch.Tensor, *, budget: int, min_codes_per_list: int
    ) -> torch.Tensor:
        if alloc.numel() == 0:
            return alloc
        if min_codes_per_list <= 0:
            totals = alloc.sum(dim=1, keepdim=True).clamp_min(1)
            scale = torch.minimum(torch.ones_like(totals, dtype=torch.float32), budget / totals.to(torch.float32))
            scaled = torch.floor(alloc.to(torch.float32) * scale).to(torch.long)
            return scaled

        alloc_cpu = alloc.to("cpu").tolist()
        for i, row in enumerate(alloc_cpu):
            total = int(sum(row))
            if total <= budget:
                continue
            excess = total - budget
            while excess > 0:
                reducible = [max(0, v - min_codes_per_list) for v in row]
                reducible_total = sum(reducible)
                if reducible_total <= 0:
                    break
                for j, cap in enumerate(reducible):
                    if cap <= 0:
                        continue
                    take = max(1, int(excess * cap / reducible_total))
                    take = min(take, cap, excess)
                    row[j] -= take
                    excess -= take
                    if excess <= 0:
                        break
            alloc_cpu[i] = row
        return torch.tensor(alloc_cpu, dtype=torch.long, device=self.device)

    def _allocate_candidate_sizes(
        self, top_lists: torch.Tensor, top_scores: torch.Tensor, config: _ResolvedSearchConfig
    ) -> torch.Tensor:
        if top_lists.numel() == 0:
            return torch.zeros_like(top_lists, dtype=torch.long)
        if config.candidate_budget is None:
            return torch.zeros_like(top_lists, dtype=torch.long)

        list_sizes = self._get_list_sizes()
        sizes = list_sizes[top_lists]
        nprobe_user = max(1, min(config.nprobe, self._nlist))
        max_codes_from_budget = (config.candidate_budget + nprobe_user - 1) // nprobe_user
        max_codes_eff = self._min_positive(config.max_codes, max_codes_from_budget)
        if max_codes_eff > 0:
            sizes = torch.minimum(sizes, torch.full_like(sizes, max_codes_eff))
        if config.max_codes_cap_per_list > 0:
            sizes = torch.minimum(
                sizes, torch.full_like(sizes, int(config.max_codes_cap_per_list))
            )

        min_codes = int(config.min_codes_per_list)
        if config.budget_strategy == "uniform":
            base = (config.candidate_budget + nprobe_user - 1) // nprobe_user
            alloc = torch.full_like(sizes, int(base))
            alloc = torch.minimum(alloc, sizes)
            if min_codes > 0:
                min_alloc = torch.minimum(sizes, torch.full_like(sizes, min_codes))
                alloc = torch.maximum(alloc, min_alloc)
        else:
            weights = self._distance_weights(top_scores)
            if weights.numel() == 0:
                alloc = torch.zeros_like(sizes)
            else:
                weight_sum = weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
                alloc = torch.round(
                    weights * (float(config.candidate_budget) / weight_sum)
                ).to(torch.long)
                alloc = torch.minimum(alloc, sizes)
                if min_codes > 0:
                    min_alloc = torch.minimum(sizes, torch.full_like(sizes, min_codes))
                    alloc = torch.maximum(alloc, min_alloc)

            if config.dynamic_nprobe and weights.numel() > 0:
                order = torch.argsort(weights, dim=1, descending=True)
                weights_sorted = torch.gather(weights, 1, order)
                caps_sorted = torch.gather(sizes, 1, order)
                if min_codes > 0:
                    min_alloc = torch.minimum(caps_sorted, torch.full_like(caps_sorted, min_codes))
                    cum_min = torch.cumsum(min_alloc, dim=1)
                    active = cum_min <= config.candidate_budget
                    if config.candidate_budget > 0:
                        active[:, 0] = True
                    min_alloc = torch.where(active, min_alloc, torch.zeros_like(min_alloc))
                    if config.candidate_budget > 0:
                        min_alloc[:, 0] = torch.minimum(
                            min_alloc[:, 0], torch.full_like(min_alloc[:, 0], config.candidate_budget)
                        )
                    remaining = config.candidate_budget - min_alloc.sum(dim=1)
                else:
                    active = torch.ones_like(caps_sorted, dtype=torch.bool)
                    min_alloc = torch.zeros_like(caps_sorted)
                    remaining = torch.full(
                        (weights.shape[0],), config.candidate_budget, dtype=torch.long, device=self.device
                    )
                remaining = remaining.clamp_min(0)
                w_active = torch.where(active, weights_sorted, torch.zeros_like(weights_sorted))
                w_sum = w_active.sum(dim=1, keepdim=True)
                denom = w_sum.clamp_min(1e-6)
                extra = torch.round(remaining.to(torch.float32).unsqueeze(1) * w_active / denom).to(torch.long)
                extra = torch.where(w_sum > 0, extra, torch.zeros_like(extra))
                alloc_sorted = torch.minimum(min_alloc + extra, caps_sorted)
                alloc = torch.zeros_like(alloc_sorted)
                alloc.scatter_(1, order, alloc_sorted)

        if config.strict_budget:
            alloc = self._enforce_strict_budget(
                alloc, budget=int(config.candidate_budget), min_codes_per_list=min_codes
            )

        if config.candidate_budget > 0:
            total = alloc.sum(dim=1)
            empty = total == 0
            if empty.any():
                first_caps = sizes[empty, 0]
                alloc[empty, 0] = torch.minimum(
                    torch.ones_like(first_caps, dtype=torch.long), first_caps
                )

        return alloc

    def _estimate_candidates_per_query(
        self,
        top_lists: torch.Tensor,
        *,
        max_codes: int | None = None,
        per_list_sizes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if top_lists.numel() == 0:
            return torch.zeros(top_lists.shape[0], dtype=torch.long, device=self.device)
        if per_list_sizes is not None:
            return per_list_sizes.sum(dim=1)
        sizes = self._get_list_sizes()
        per_probe = sizes[top_lists]
        counts = per_probe.sum(dim=1)
        max_codes_i = self._max_codes if max_codes is None else int(max_codes)
        if max_codes_i:
            counts = torch.minimum(counts, torch.full_like(counts, max_codes_i))
        return counts

    def _sum_read_codes_for_chunk(
        self,
        top_lists: torch.Tensor,
        *,
        max_codes: int,
        per_list_sizes: torch.Tensor | None,
    ) -> int:
        if top_lists.numel() == 0:
            return 0
        sizes = self._get_list_sizes()[top_lists]
        if per_list_sizes is not None:
            sizes = torch.minimum(sizes, per_list_sizes)
        else:
            max_codes_i = int(max_codes)
            if max_codes_i > 0:
                cum = torch.cumsum(sizes, dim=1)
                keep = cum <= max_codes_i
                keep[:, 0] = True
                sizes = sizes * keep
        return int(sizes.sum().item())

    def _update_csr_debug_stats(
        self,
        debug_stats: dict[str, float | int | str] | None,
        *,
        top_lists: torch.Tensor,
        max_codes: int,
        per_list_sizes: torch.Tensor | None,
        tasks_total: int | None,
        groups: torch.Tensor | None,
    ) -> None:
        if debug_stats is None:
            return
        sum_read = self._sum_read_codes_for_chunk(
            top_lists, max_codes=max_codes, per_list_sizes=per_list_sizes
        )
        debug_stats["sum_read_codes"] = int(debug_stats.get("sum_read_codes", 0)) + int(sum_read)

        if top_lists.numel():
            list_sizes = self._get_list_sizes()[top_lists]
            if per_list_sizes is not None:
                list_sizes = torch.minimum(list_sizes, per_list_sizes)
            max_list_len = int(list_sizes.max().item()) if list_sizes.numel() else 0
            debug_stats["max_list_len"] = max(
                int(debug_stats.get("max_list_len", 0)), int(max_list_len)
            )

        if tasks_total is not None:
            debug_stats["tasks_total"] = int(debug_stats.get("tasks_total", 0)) + int(tasks_total)

        if groups is not None and groups.numel():
            group_sizes = groups[:, 2] - groups[:, 1]
            max_group = int(group_sizes.max().item()) if group_sizes.numel() else 0
            debug_stats["max_group_size"] = max(
                int(debug_stats.get("max_group_size", 0)), int(max_group)
            )
            debug_stats["unique_lists_in_chunk"] = max(
                int(debug_stats.get("unique_lists_in_chunk", 0)), int(groups.shape[0])
            )

    def _group_task_max_sizes(
        self, task_sizes: torch.Tensor, groups: torch.Tensor
    ) -> torch.Tensor:
        if groups.numel() == 0:
            return torch.zeros((0,), dtype=torch.long, device=self.device)
        if task_sizes.numel() == 0:
            return torch.zeros((groups.shape[0],), dtype=torch.long, device=self.device)
        counts = (groups[:, 2] - groups[:, 1]).to(torch.long)
        group_ids = torch.repeat_interleave(
            torch.arange(groups.shape[0], device=self.device), counts
        )
        if group_ids.numel() == 0:
            return torch.zeros((groups.shape[0],), dtype=torch.long, device=self.device)
        group_max = torch.zeros(
            (groups.shape[0],), dtype=task_sizes.dtype, device=self.device
        )
        group_max.scatter_reduce_(0, group_ids, task_sizes, reduce="amax", include_self=False)
        return group_max

    def _csr_groups_cpu(
        self,
        groups: torch.Tensor,
        task_sizes: torch.Tensor | None,
    ) -> list[list[int]]:
        if groups.numel() == 0:
            return []
        if task_sizes is None:
            pad = torch.zeros((groups.shape[0], 1), dtype=groups.dtype, device=groups.device)
            return torch.cat([groups, pad], dim=1).to("cpu").tolist()
        group_max = self._group_task_max_sizes(task_sizes, groups)
        return torch.cat([groups, group_max.unsqueeze(1)], dim=1).to("cpu").tolist()

    def _build_candidate_index_matrix_from_lists(
        self,
        top_lists: torch.Tensor,
        *,
        max_candidates: int,
        pos_1d: torch.Tensor,
        max_codes: int | None = None,
        per_list_sizes: torch.Tensor | None = None,
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
        if per_list_sizes is not None:
            sizes = torch.minimum(sizes, per_list_sizes)
        else:
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
        self,
        top_lists: torch.Tensor,
        *,
        max_codes: int | None = None,
        per_list_sizes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if top_lists.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            return empty, empty, empty

        b, probe = top_lists.shape
        sizes = self._get_list_sizes()[top_lists]
        if per_list_sizes is not None:
            sizes = torch.minimum(sizes, per_list_sizes)
            keep = sizes > 0
        else:
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
        self,
        top_lists: torch.Tensor,
        *,
        max_codes: int | None = None,
        per_list_sizes: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if top_lists.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=self.device)
            return empty, empty, empty, empty

        b, probe = top_lists.shape
        sizes = self._get_list_sizes()[top_lists]
        if per_list_sizes is not None:
            sizes = torch.minimum(sizes, per_list_sizes)
            keep = sizes > 0
        else:
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

    def _search_csr_online(
        self,
        xq: torch.Tensor,
        k: int,
        *,
        max_codes: int,
        nprobe: int,
        top_lists: torch.Tensor | None = None,
        per_list_sizes: torch.Tensor | None = None,
        debug_stats: dict[str, float | int | str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if top_lists is None:
            top_lists = self._top_probed_lists(xq, nprobe=nprobe)
        chunks = self._iter_query_chunks_csr(xq.shape[0], nprobe=nprobe)
        if debug_stats is not None:
            debug_stats["chunks"] = int(len(chunks))
            counts = self._estimate_candidates_per_query(
                top_lists, max_codes=max_codes, per_list_sizes=per_list_sizes
            )
            debug_stats["total_candidates_est"] = int(counts.sum().item())

        dists = torch.empty((xq.shape[0], k), dtype=self.dtype, device=self.device)
        labels = torch.empty((xq.shape[0], k), dtype=torch.long, device=self.device)

        for start, end in chunks:
            chunk_q = xq[start:end]
            chunk_lists = top_lists[start:end]
            chunk_sizes = per_list_sizes[start:end] if per_list_sizes is not None else None
            chunk_dists, chunk_labels = self._search_csr_online_chunk(
                chunk_q,
                chunk_lists,
                k,
                max_codes=max_codes,
                per_list_sizes=chunk_sizes,
                debug_stats=debug_stats,
            )
            dists[start:end] = chunk_dists
            labels[start:end] = chunk_labels
        if debug_stats is not None and debug_stats.get("blocked_pad_den"):
            pad_den = float(debug_stats.get("blocked_pad_den", 0))
            pad_num = float(debug_stats.get("blocked_pad_num", 0))
            debug_stats["blocked_pad_ratio"] = pad_num / pad_den if pad_den > 0 else 0.0
            blocks = float(debug_stats.get("blocked_blocks", 0))
            if blocks > 0:
                debug_stats["blocked_gmax_avg"] = float(debug_stats.get("blocked_gmax_sum", 0)) / blocks
                debug_stats["blocked_lmax_avg"] = float(debug_stats.get("blocked_lmax_sum", 0)) / blocks
        return dists, labels

    def _search_csr_online_chunk(
        self,
        xq_chunk: torch.Tensor,
        top_lists: torch.Tensor,
        k: int,
        *,
        max_codes: int,
        per_list_sizes: torch.Tensor | None = None,
        debug_stats: dict[str, float | int | str] | None = None,
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
            query_counts = self._estimate_candidates_per_query(
                top_lists, max_codes=max_codes, per_list_sizes=per_list_sizes
            )
            query_counts_cpu = query_counts.to("cpu")
            max_candidates = int(query_counts_cpu.max().item()) if query_counts_cpu.numel() else 0
            if max_candidates > 0 and (chunk_size * max_candidates) <= self._candidate_budget():
                pos_1d = torch.arange(max_candidates, dtype=torch.long, device=self.device)
                index_matrix, query_counts = self._build_candidate_index_matrix_from_lists(
                    top_lists,
                    max_candidates=max_candidates,
                    pos_1d=pos_1d,
                    max_codes=max_codes,
                    per_list_sizes=per_list_sizes,
                )
                if debug_stats is not None:
                    debug_stats["matrix_fallback"] = int(debug_stats.get("matrix_fallback", 0)) + 1
                    self._update_csr_debug_stats(
                        debug_stats,
                        top_lists=top_lists,
                        max_codes=max_codes,
                        per_list_sizes=per_list_sizes,
                        tasks_total=None,
                        groups=None,
                    )
                return self._search_from_index_matrix(q, index_matrix, query_counts, k)
            return self._search_csr_buffered_chunk(
                q,
                q2,
                top_lists,
                k,
                max_codes=max_codes,
                per_list_sizes=per_list_sizes,
                debug_stats=debug_stats,
            )

        if per_list_sizes is None:
            tasks_q, _, groups = self._build_tasks_from_lists(top_lists, max_codes=max_codes)
            tasks_p = None
            task_sizes = None
        else:
            tasks_q, _, tasks_p, groups = self._build_tasks_from_lists_with_probe(
                top_lists, max_codes=max_codes, per_list_sizes=per_list_sizes
            )
            task_sizes = per_list_sizes[tasks_q, tasks_p] if tasks_q.numel() else None
        if groups.numel() == 0:
            return best_scores, best_packed

        q_tasks = q.index_select(0, tasks_q)
        q2_tasks = q2.index_select(0, tasks_q) if q2 is not None else None

        if debug_stats is not None:
            self._update_csr_debug_stats(
                debug_stats,
                top_lists=top_lists,
                max_codes=max_codes,
                per_list_sizes=per_list_sizes,
                tasks_total=int(tasks_q.numel()),
                groups=groups,
            )
            debug_stats["list_groups"] = int(debug_stats.get("list_groups", 0)) + int(groups.shape[0])
        groups_cpu = self._csr_groups_cpu(groups, task_sizes)
        if (
            per_list_sizes is not None
            and self.metric == "l2"
            and self._search_csr_blocked_groups(
                q_tasks,
                q2_tasks,
                tasks_q,
                task_sizes,
                groups_cpu,
                k,
                best_scores=best_scores,
                best_packed=best_packed,
                fill=fill,
                debug_stats=debug_stats,
            )
        ):
            out_ids = self._list_ids.index_select(0, best_packed.clamp_min(0).reshape(-1)).reshape(best_packed.shape)
            out_ids = torch.where(best_packed < 0, torch.full_like(out_ids, -1), out_ids)
            return best_scores, out_ids
        for l, start, end, max_len in groups_cpu:
            a = int(self._list_offsets_cpu[int(l)])
            b = int(self._list_offsets_cpu[int(l) + 1])
            if b <= a:
                continue
            query_ids = tasks_q[int(start) : int(end)]
            if query_ids.numel() == 0:
                continue
            if task_sizes is not None:
                group_sizes = task_sizes[int(start) : int(end)]
                if group_sizes.numel() == 0:
                    continue
                if max_len <= 0:
                    continue
                list_len = b - a
                if max_len < list_len:
                    b = a + max_len
                else:
                    max_len = list_len
            else:
                group_sizes = None
                max_len = b - a

            qg = q_tasks[int(start) : int(end)]
            if q2 is not None:
                q2g = q2_tasks[int(start) : int(end)].unsqueeze(1)
            else:
                q2g = None

            vec_chunk = self._csr_vec_chunk(buffered=False)
            if (b - a) <= vec_chunk:
                x = self._packed_embeddings[a:b]
                if debug_stats is not None:
                    debug_stats["matmul_calls"] = int(debug_stats.get("matmul_calls", 0)) + 1
                    debug_stats["topk_calls"] = int(debug_stats.get("topk_calls", 0)) + 1
                    debug_stats["matmul_total_rows"] = int(debug_stats.get("matmul_total_rows", 0)) + int(
                        qg.shape[0]
                    )
                    debug_stats["matmul_total_cols"] = int(debug_stats.get("matmul_total_cols", 0)) + int(
                        x.shape[0]
                    )
                prod = torch.matmul(qg, x.transpose(0, 1))
                if self.metric == "l2":
                    x2 = self._packed_norms[a:b]
                    dist = q2g + x2.unsqueeze(0) - (2.0 * prod)
                    dist = dist.clamp_min_(0)
                    if group_sizes is not None:
                        pos = torch.arange(max_len, device=self.device)
                        mask = pos.unsqueeze(0) >= group_sizes.unsqueeze(1)
                        dist.masked_fill_(mask, float("inf"))
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
                    if debug_stats is not None:
                        debug_stats["matmul_calls"] = int(debug_stats.get("matmul_calls", 0)) + 1
                        debug_stats["topk_calls"] = int(debug_stats.get("topk_calls", 0)) + 1
                        debug_stats["matmul_total_rows"] = int(debug_stats.get("matmul_total_rows", 0)) + int(
                            qg.shape[0]
                        )
                        debug_stats["matmul_total_cols"] = int(debug_stats.get("matmul_total_cols", 0)) + int(
                            x.shape[0]
                        )
                    prod = torch.matmul(qg, x.transpose(0, 1))
                    if self.metric == "l2":
                        x2 = self._packed_norms[p:pe]
                        dist = q2g + x2.unsqueeze(0) - (2.0 * prod)
                        dist = dist.clamp_min_(0)
                        if group_sizes is not None:
                            pos = torch.arange(p - a, pe - a, device=self.device)
                            mask = pos.unsqueeze(0) >= group_sizes.unsqueeze(1)
                            dist.masked_fill_(mask, float("inf"))
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

    def _search_csr_blocked_groups(
        self,
        q_tasks: torch.Tensor,
        q2_tasks: torch.Tensor | None,
        tasks_q: torch.Tensor,
        task_sizes: torch.Tensor | None,
        groups_cpu: list[list[int]],
        k: int,
        *,
        best_scores: torch.Tensor,
        best_packed: torch.Tensor,
        fill: float,
        debug_stats: dict[str, float | int | str] | None,
    ) -> bool:
        if q2_tasks is None or task_sizes is None:
            return False
        block_size = self._csr_list_block_size()
        if block_size <= 1 or not groups_cpu:
            return False
        pad_ratio_limit = float(self._csr_block_pad_ratio_limit())
        max_block_elements = int(self._csr_block_max_elements())
        cost_ratio_limit = float(self._csr_block_cost_ratio_limit())

        vec_chunk = self._csr_vec_chunk(buffered=False)
        offsets_cpu = (
            self._list_offsets_cpu
            if len(self._list_offsets_cpu) == self._nlist + 1
            else self._list_offsets.to("cpu").tolist()
        )
        device = self.device
        d = self.d
        largest = self.metric == "ip"
        group_entries: list[tuple[int, int, int, int, int, int, int]] = []
        for l, start, end, max_len in groups_cpu:
            s = int(start)
            e = int(end)
            if e <= s:
                continue
            list_id = int(l)
            a = int(offsets_cpu[list_id])
            b = int(offsets_cpu[list_id + 1])
            if b <= a:
                continue
            gsize = e - s
            if gsize <= 0:
                continue
            list_len = b - a
            if max_len > 0 and max_len < list_len:
                list_len = int(max_len)
                b = a + list_len
            if list_len <= 0:
                continue
            if list_len > vec_chunk:
                return False
            group_entries.append((gsize, list_len, list_id, s, e, a, b))

        if not group_entries:
            return False

        group_entries.sort(key=lambda v: (v[0], v[1]))

        candidate_sizes: list[int] = []
        for size in (block_size, 16, 8, 4):
            if size > 1 and size <= block_size and size not in candidate_sizes:
                candidate_sizes.append(size)
        chosen_block = 1
        for size in candidate_sizes:
            if size <= 1:
                continue
            ok = True
            for i in range(0, len(group_entries), size):
                block = group_entries[i : i + size]
                g_sizes = [info[0] for info in block]
                l_sizes = [info[1] for info in block]
                gmax = max(g_sizes)
                lmax = max(l_sizes)
                cost_block = len(block) * gmax * lmax
                cost_list = sum(gs * ls for gs, ls in zip(g_sizes, l_sizes))
                if cost_list <= 0:
                    continue
                if max_block_elements > 0 and cost_block > max_block_elements:
                    ok = False
                    break
                if cost_block > cost_list * cost_ratio_limit:
                    ok = False
                    break
            if ok:
                chosen_block = size
                break

        if chosen_block <= 1:
            if debug_stats is not None:
                debug_stats["list_block_size"] = int(chosen_block)
            return False

        if debug_stats is not None:
            debug_stats["list_block_size"] = int(chosen_block)
            debug_stats["blocked_pad_ratio_limit"] = float(pad_ratio_limit)

        for block_start in range(0, len(group_entries), chosen_block):
            block = group_entries[block_start : block_start + chosen_block]
            g_sizes = [info[0] for info in block]
            l_sizes = [info[1] for info in block]
            gmax = max(g_sizes)
            lmax = max(l_sizes)
            bcount = len(block)
            pad_ratio = 0.0
            pad_den = sum(gs * ls for gs, ls in zip(g_sizes, l_sizes))
            if pad_den > 0:
                pad_ratio = (bcount * gmax * lmax) / pad_den

            q_pad = self._workspace.ensure(
                "csr_block_q", (bcount, gmax, d), dtype=self.dtype, device=device
            )
            q2_pad = self._workspace.ensure(
                "csr_block_q2", (bcount, gmax), dtype=self.dtype, device=device
            )
            x_pad = self._workspace.ensure(
                "csr_block_x", (bcount, lmax, d), dtype=self.dtype, device=device
            )
            x2_pad = self._workspace.ensure(
                "csr_block_x2", (bcount, lmax), dtype=self.dtype, device=device
            )
            sizes_pad = self._workspace.ensure(
                "csr_block_sizes", (bcount, gmax), dtype=torch.long, device=device
            )
            sizes_pad.zero_()

            query_ids_list: list[torch.Tensor] = []
            offsets_list: list[int] = []
            for i, (gsize, list_len, list_id, s, e, a, b) in enumerate(block):
                q_slice = q_tasks[s:e]
                q_pad[i, :gsize] = q_slice
                q2_pad[i, :gsize] = q2_tasks[s:e]
                x_slice = self._packed_embeddings[a:b]
                x_pad[i, :list_len] = x_slice
                x2_pad[i, :list_len] = self._packed_norms[a:b]
                sizes_pad[i, :gsize] = task_sizes[s:e]
                query_ids_list.append(tasks_q[s:e])
                offsets_list.append(a)

            lens_buf = self._workspace.ensure(
                "csr_block_lens", (bcount,), dtype=torch.long, device=device
            )
            gsize_buf = self._workspace.ensure(
                "csr_block_gsizes", (bcount,), dtype=torch.long, device=device
            )
            offsets_buf = self._workspace.ensure(
                "csr_block_offsets", (bcount,), dtype=torch.long, device=device
            )
            lens_cpu = torch.as_tensor(l_sizes, dtype=torch.long)
            gsize_cpu = torch.as_tensor(g_sizes, dtype=torch.long)
            offsets_cpu_tensor = torch.as_tensor(offsets_list, dtype=torch.long)
            lens_buf[:bcount].copy_(lens_cpu)
            gsize_buf[:bcount].copy_(gsize_cpu)
            offsets_buf[:bcount].copy_(offsets_cpu_tensor)

            if debug_stats is not None:
                prev_ratio = float(debug_stats.get("blocked_pad_ratio_max", 0.0))
                if pad_ratio > prev_ratio:
                    debug_stats["blocked_pad_ratio_max"] = float(pad_ratio)
                debug_stats["blocked_used_blocks"] = int(debug_stats.get("blocked_used_blocks", 0)) + 1
                debug_stats["matmul_calls"] = int(debug_stats.get("matmul_calls", 0)) + 1
                debug_stats["topk_calls"] = int(debug_stats.get("topk_calls", 0)) + 1
                debug_stats["list_blocks"] = int(debug_stats.get("list_blocks", 0)) + 1
                debug_stats["blocked_blocks"] = int(debug_stats.get("blocked_blocks", 0)) + 1
                debug_stats["blocked_lists"] = int(debug_stats.get("blocked_lists", 0)) + int(bcount)
                debug_stats["blocked_pad_num"] = int(debug_stats.get("blocked_pad_num", 0)) + int(
                    bcount * gmax * lmax
                )
                debug_stats["blocked_pad_den"] = int(debug_stats.get("blocked_pad_den", 0)) + int(
                    sum(gs * ls for gs, ls in zip(g_sizes, l_sizes))
                )
                debug_stats["blocked_gmax_sum"] = int(debug_stats.get("blocked_gmax_sum", 0)) + int(gmax)
                debug_stats["blocked_lmax_sum"] = int(debug_stats.get("blocked_lmax_sum", 0)) + int(lmax)
                debug_stats["matmul_total_rows"] = int(debug_stats.get("matmul_total_rows", 0)) + int(
                    sum(g_sizes)
                )
                debug_stats["matmul_total_cols"] = int(debug_stats.get("matmul_total_cols", 0)) + int(
                    sum(l_sizes)
                )

            prod = torch.bmm(q_pad, x_pad.transpose(1, 2))
            dist = q2_pad.unsqueeze(2) + x2_pad.unsqueeze(1) - (2.0 * prod)
            dist = dist.clamp_min_(0)

            list_pos = self._workspace.ensure(
                "csr_block_list_pos", (lmax,), dtype=torch.long, device=device
            )
            torch.arange(lmax, device=device, out=list_pos)
            list_mask = list_pos.unsqueeze(0) >= lens_buf[:bcount].unsqueeze(1)
            dist.masked_fill_(list_mask.unsqueeze(1), float("inf"))

            query_pos = self._workspace.ensure(
                "csr_block_query_pos", (gmax,), dtype=torch.long, device=device
            )
            torch.arange(gmax, device=device, out=query_pos)
            query_mask = query_pos.unsqueeze(0) >= gsize_buf[:bcount].unsqueeze(1)
            dist.masked_fill_(query_mask.unsqueeze(2), float("inf"))

            group_mask = list_pos.view(1, 1, lmax) >= sizes_pad.unsqueeze(2)
            dist.masked_fill_(group_mask, float("inf"))

            topk = min(k, lmax)
            cand_scores, cand_j = torch.topk(dist, topk, largest=largest, dim=2)
            offsets = offsets_buf[:bcount].view(bcount, 1, 1)
            cand_packed = cand_j + offsets

            for i, q_ids in enumerate(query_ids_list):
                gsize = int(q_ids.numel())
                if gsize <= 0:
                    continue
                self._merge_topk(
                    best_scores,
                    best_packed,
                    q_ids,
                    cand_scores[i, :gsize],
                    cand_packed[i, :gsize],
                    k,
                    largest=largest,
                )
                if debug_stats is not None:
                    debug_stats["blocked_merge_calls"] = int(debug_stats.get("blocked_merge_calls", 0)) + 1
        return True

    def _search_csr_buffered_chunk(
        self,
        q: torch.Tensor,
        q2: torch.Tensor | None,
        top_lists: torch.Tensor,
        k: int,
        *,
        max_codes: int,
        per_list_sizes: torch.Tensor | None = None,
        debug_stats: dict[str, float | int | str] | None = None,
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

        tasks_q, _, tasks_p, groups = self._build_tasks_from_lists_with_probe(
            top_lists, max_codes=max_codes, per_list_sizes=per_list_sizes
        )
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
        task_sizes = per_list_sizes[tasks_q, tasks_p] if per_list_sizes is not None else None

        if debug_stats is not None:
            self._update_csr_debug_stats(
                debug_stats,
                top_lists=top_lists,
                max_codes=max_codes,
                per_list_sizes=per_list_sizes,
                tasks_total=int(tasks_q.numel()),
                groups=groups,
            )
        groups_cpu = self._csr_groups_cpu(groups, task_sizes)
        vec_chunk = self._csr_vec_chunk(buffered=True)
        for l, start, end, max_len in groups_cpu:
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
            if task_sizes is not None:
                group_sizes = task_sizes[s:e]
                if group_sizes.numel() == 0:
                    continue
                if max_len <= 0:
                    continue
                list_len = b - a
                if max_len < list_len:
                    b = a + max_len
                else:
                    max_len = list_len
            else:
                group_sizes = None
                max_len = b - a

            if (b - a) <= vec_chunk:
                x = self._packed_embeddings[a:b]
                if debug_stats is not None:
                    debug_stats["matmul_calls"] = int(debug_stats.get("matmul_calls", 0)) + 1
                    debug_stats["topk_calls"] = int(debug_stats.get("topk_calls", 0)) + 1
                    debug_stats["matmul_total_rows"] = int(debug_stats.get("matmul_total_rows", 0)) + int(
                        qg.shape[0]
                    )
                    debug_stats["matmul_total_cols"] = int(debug_stats.get("matmul_total_cols", 0)) + int(
                        x.shape[0]
                    )
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
                    if debug_stats is not None:
                        debug_stats["matmul_calls"] = int(debug_stats.get("matmul_calls", 0)) + 1
                        debug_stats["topk_calls"] = int(debug_stats.get("topk_calls", 0)) + 1
                        debug_stats["matmul_total_rows"] = int(debug_stats.get("matmul_total_rows", 0)) + int(
                            qg.shape[0]
                        )
                        debug_stats["matmul_total_cols"] = int(debug_stats.get("matmul_total_cols", 0)) + int(
                            x.shape[0]
                        )
                    prod = torch.matmul(qg, x.transpose(0, 1))
                    if self.metric == "l2":
                        x2 = self._packed_norms[p:pe]
                        dist = q2g + x2.unsqueeze(0) - (2.0 * prod)
                        dist = dist.clamp_min_(0)
                        if group_sizes is not None:
                            pos = torch.arange(p - a, pe - a, device=self.device)
                            mask = pos.unsqueeze(0) >= group_sizes.unsqueeze(1)
                            dist.masked_fill_(mask, float("inf"))
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

    def _csr_list_block_size(self) -> int:
        if self.device.type == "cpu":
            return 1
        return 16

    def _csr_block_pad_ratio_limit(self) -> float:
        if self.device.type == "cpu":
            return 1.0
        return 1.15

    def _csr_block_cost_ratio_limit(self) -> float:
        if self.device.type == "cpu":
            return 1.0
        return 2.0

    def _csr_block_max_elements(self) -> int:
        elem_size = torch.tensor([], dtype=self.dtype).element_size()
        if self.device.type == "cpu":
            target_bytes = 16 * 1024 * 1024
        else:
            target_bytes = 64 * 1024 * 1024
        return max(1, int(target_bytes // max(1, elem_size)))

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

    def _iter_query_chunks_csr(self, nq: int, *, nprobe: int) -> list[tuple[int, int]]:
        if nq <= 0:
            return [(0, 0)]
        budget = self._csr_task_budget()
        probe = max(1, min(int(nprobe), self._nlist))
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
        )
