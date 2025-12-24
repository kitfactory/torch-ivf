# torch-ivf Architecture

## 目的
性能改善（Phase P1: Eager-only）に必要な構造を明文化し、キャッシュ/ワークスペース/検索経路の責務と依存方向を固定する。

## レイヤー構造と依存方向
- Interface 層: CLI・ベンチ・サンプル（`scripts/`, `examples/`）。
- Application 層: Index 公開 API（`IndexBase`, `IndexFlat*`, `IndexIVFFlat`）。
- Domain 層: IVF 検索パイプライン、k-means、候補収集、topk merge。
- Infrastructure 層: PyTorch 演算・シリアライズ・ユーティリティ（`torch.*`, `utils/`）。

依存方向: Interface → Application → Domain → Infrastructure のみ。逆依存は作らない。

## 主要コンポーネント責務
- `IndexIVFFlat`: 公開 API、検索モード切り替え、キャッシュ/ワークスペースのライフサイクル管理。
- `SearchParams`（公開構造）: search の profile/近似設定を保持し、内部設定へ変換する。
- `SearchCache`（内部構造）: 派生テンソルを保持し、無効化条件を明示。
- `Workspace`（内部構造）: CSR 検索の一時テンソルを再利用し、容量とデバイス/型の整合性を管理。
- `CandidateBudgeter`（内部ロジック）: 近似モード時の候補予算配分（uniform/distance_weighted）。
- `SafePruner`（内部ロジック）: L2 の安全剪定（list radius による list 単位スキップ）。
- `ListOrdering`（内部ロジック）: list 内の並べ替え（residual_norm_asc / proj_desc）と再構築ポリシー（manual/auto_threshold）。
- `SubclusterBounds`（内部構造）: list 内 subcluster の bound を保持し、Exact のまま部分スキップを行う。
- `AnchorPrefilter`（内部構造）: list anchor によるプレフィルタで予算配分対象を絞る。

## 主要 I/F（最小粒度）
### IndexIVFFlat
```python
class IndexIVFFlat:
    def search(self, xq: torch.Tensor, k: int, *, params: "SearchParams | None" = None) -> tuple[torch.Tensor, torch.Tensor]: ...
    def train(self, xb: torch.Tensor) -> None: ...
    def add(self, xb: torch.Tensor) -> None: ...
    def reset(self) -> None: ...
    @property
    def last_search_stats(self) -> dict | None: ...
    def to(self, device: torch.device | str) -> "IndexIVFFlat": ...
```

### SearchParams（公開）
```python
class SearchParams:
    profile: str  # exact / speed / approx
    safe_pruning: bool
    approximate: bool
    nprobe: int | None
    max_codes: int | None
    candidate_budget: int | None
    budget_strategy: str  # uniform / distance_weighted
    list_ordering: str | None  # none / residual_norm_asc / proj_desc
    rebuild_policy: str  # manual / auto_threshold
    rebuild_threshold_adds: int
    dynamic_nprobe: bool
    min_codes_per_list: int
    max_codes_cap_per_list: int
    strict_budget: bool
    use_per_list_sizes: bool
    debug_stats: bool
```

解決順序: profile の既定値 → Index 設定 → 明示 SearchParams。`dynamic_nprobe` は `budget_strategy="distance_weighted"` のみ有効で、`nprobe_user` を上限として減少のみ許可する。

### キャッシュ I/F（内部）
```python
class _SearchCache:
    centroids_T: torch.Tensor | None
    centroid_norm2: torch.Tensor | None
    list_sizes: torch.Tensor | None
    effective_max_codes: int | None

class IndexIVFFlat:
    def _invalidate_search_cache(self) -> None: ...
    def _ensure_search_cache(self) -> None: ...
```

### ワークスペース I/F（内部）
```python
class _Workspace:
    device: torch.device | None
    dtype: torch.dtype | None
    capacity: dict[str, int]

    def ensure(self, name: str, shape: tuple[int, ...], dtype: torch.dtype, device: torch.device) -> torch.Tensor: ...
```

### 近似モード I/F（内部）
```python
class _ApproximateConfig:
    enabled: bool
    candidate_budget: int | None
    budget_strategy: str  # uniform / distance_weighted
    min_codes_per_list: int
    max_codes_cap_per_list: int
    strict_budget: bool
    debug_stats: bool
    list_ordering: str | None  # none / residual_norm_asc / proj_desc
    rebuild_policy: str  # manual / auto_threshold
    rebuild_threshold_adds: int
    dynamic_nprobe: bool
```

### Subcluster bounds I/F（内部）
```python
class _SubclusterBounds:
    sub_centroids: torch.Tensor
    sub_radii: torch.Tensor
    sub_offsets: torch.Tensor  # list -> subcluster ranges
```

### Anchor prefilter I/F（内部）
```python
class _AnchorPrefilter:
    anchors: torch.Tensor  # [nlist, n_anchor, d]
```

## 非機能ポリシー
- **互換性**: 既存 API・挙動（検索結果）を維持する。
- **Eager-only**: `torch.compile` や独自カーネルに依存しない。
- **メモリ安全性**: 既存の chunking/streaming による OOM 回避を壊さない。
- **P1 対象**: P1 は L2 のみを対象とし、IP は Phase 2 で対応する。

## ログ/エラー方針
- 例外は `ValueError` / `RuntimeError` を継続使用する。
- ログを追加する場合は最小限とし、形式は `[torch-ivf][P1] message` を採用する。
