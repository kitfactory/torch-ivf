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
- `SearchCache`（内部構造）: 派生テンソルを保持し、無効化条件を明示。
- `Workspace`（内部構造）: CSR 検索の一時テンソルを再利用し、容量とデバイス/型の整合性を管理。

## 主要 I/F（最小粒度）
### IndexIVFFlat
```python
class IndexIVFFlat:
    def search(self, xq: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]: ...
    def train(self, xb: torch.Tensor) -> None: ...
    def add(self, xb: torch.Tensor) -> None: ...
    def reset(self) -> None: ...
    def to(self, device: torch.device | str) -> "IndexIVFFlat": ...
```

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

## 非機能ポリシー
- **互換性**: 既存 API・挙動（検索結果）を維持する。
- **Eager-only**: `torch.compile` や独自カーネルに依存しない。
- **メモリ安全性**: 既存の chunking/streaming による OOM 回避を壊さない。

## ログ/エラー方針
- 例外は `ValueError` / `RuntimeError` を継続使用する。
- ログを追加する場合は最小限とし、形式は `[torch-ivf][P1] message` を採用する。
