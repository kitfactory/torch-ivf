# torch-ivf Specification

## 1. Overview
- **目的**: PyTorch だけで `IndexFlat*` / `IndexIVFFlat` 相当の IVF を実現し、CPU / CUDA / ROCm / DirectML を同一 API で扱う。特に Windows + ROCm (PyTorch 公式ビルド) での GPU 検索を最優先する。
- **主なユーザー**: Faiss を利用しているが Windows/ROCm で GPU を動かしたい開発者、PyTorch ベースの特徴量検索を統一したい MLOps チーム。
- **約束**: `import faiss` を `import torch_ivf as faiss` にほぼ置き換えるだけで移行できる互換インターフェースを提供する。

## 2. Scope
### 2.1 Included
1. L2 / Inner Product 対応の `IndexFlatL2`, `IndexFlatIP`, `IndexIVFFlat`.
2. PyTorch Tensor ベースの `train`, `add`, `add_with_ids`, `search`, `range_search`.
3. CPU / CUDA / ROCm / DirectML へのデバイス移動 (`to`, `cpu`, `cuda`, `rocm`, `dml`).
4. PyTorch `state_dict` 形式でのシリアライズ (`save`, `load`, `from_state_dict`).
5. `scripts/benchmark.py` / `scripts/benchmark_faiss_cpu.py` によるベンチマーク計測と JSON 収集。

### 2.2 Excluded (v0)
- PQ / IVFPQ / IVF-HNSW など高機能なインデックス。
- Faiss C++ バイナリ互換 (将来の検討事項)。
- マルチノード分散、オンライン更新 (MVP 外)。

## 3. Terminology
- `xb`: base vectors (float16/float32/bfloat16 Tensor).
- `xq`: query vectors.
- `quantizer`: coarse centroid を保持する `IndexFlat*`.
- `list_offsets`: inverted list の prefix sum。`packed_embeddings` と組み合わせて candidate を取り出す。

## 4. Module Layout & API
```
torch_ivf/
 ├─ __init__.py
 ├─ index/
 │   ├─ base.py        # IndexBase とデバイス共通ロジック
 │   ├─ flat.py        # IndexFlatL2 / IndexFlatIP
 │   └─ ivf_flat.py    # IndexIVFFlat
 ├─ nn/kmeans.py       # mini-batch k-means
 ├─ utils/
 │   ├─ serialization.py
 │   └─ tensor_ops.py
 └─ typing.py
```

| クラス | 役割 | Faiss 対応 |
|--------|------|------------|
| `IndexFlatL2` | L2 距離の完全探索 | `faiss.IndexFlatL2` |
| `IndexFlatIP` | 内積スコアの完全探索 | `faiss.IndexFlatIP` |
| `IndexIVFFlat` | IVF + Flat の組み合わせ | `faiss.IndexIVFFlat` |

共通コンストラクタ:
- `d: int` (ベクトル次元, 必須)
- `metric: Literal["l2", "ip"]` (既定 `"l2"`)
- `device: torch.device | str | None` (None の場合は PyTorch 既定)
- `dtype: torch.dtype` (既定 `torch.float32`)

共通属性: `d`, `ntotal`, `nlist`, `nprobe`, `max_codes`, `device`, `is_trained`.

## 5. Behavioral Requirements

### 5.1 Data validation
- `xb`, `xq` の dtype は `float16`, `float32`, `bfloat16` を許可。内部 centroid は常に float32 で保持。
- `tensor.shape[1] == d` を強制。違反時は `ValueError`。
- 入力 Tensor は `contiguous()` に変換し、必要に応じて `to(device)` する。`pin_memory` は呼び出し側の責務。

### 5.2 Device semantics
- コンストラクタで device を指定しない場合は PyTorch 既定 device を利用。
- `IndexBase.to(device)` は内部 Tensor（centroids、packed embeddings、ids、offsets）を再配置した新インスタンスを返す。
- `cuda` / `rocm` / `dml` メソッドも `to(torch.device("cuda"))` 等の薄いラッパー。
- 1 つのインスタンスは基本的に 1 デバイス専用とし、`train/add/search` を跨いで頻繁に移動しない設計とする。

### 5.3 Training
- `IndexIVFFlat.train(xb)` は mini-batch k-means (最大 20 outer iters, tol=1e-3) を利用。
- `nlist` の既定は `min(1024, max(1, int(sqrt(ntotal))))`。ユーザー指定を優先。
- 収束後に `is_trained=True`。未学習のまま `add` / `search` を呼ぶと `RuntimeError`。

### 5.4 Add / Add with IDs
- `add` は自動 ID (`torch.arange(self.ntotal, ...)`) を払い出す。
- `add_with_ids` は `int64` Tensor を要求。重複 ID は Faiss と同様に許容。
- ベッチサイズの大きい追加を推奨。DataLoader から渡す場合は `pin_memory=True` + `to(device, non_blocking=True)` を推奨。

### 5.5 Search (batched pipeline)
1. まずクエリ全体に対して coarse 検索を 1 回だけ実行する（小さな行列を何度も作らない）。  
   - `_top_probed_lists(xq)` が `centroid_scores = pairwise(xq, centroids)` → `torch.topk(..., nprobe)` を行い、`top_lists: (nq, nprobe)` を返す。
2. inverted list の実サイズから “実際の候補数” を見積もってチャンク分割する。  
   - `list_sizes = list_offsets[1:] - list_offsets[:-1]` を使い、`candidates_per_query = list_sizes[top_lists].sum(dim=1)` を計算する。
   - `_iter_query_chunks(candidates_per_query)` が候補数合計が予算（CPU=400k / GPU=250k candidates）を超えないように `(start, end)` のチャンク列を作る。
3. 各チャンクで `_collect_candidate_vectors_from_lists(top_lists_chunk)` を呼び出し、候補ベクトル・ID・（L2 用）事前計算済みノルムをまとめて gather する。  
   - `repeat_interleave` と prefix sum を使って、候補を **クエリ順に連結**した `(total_candidates, d)` テンソルにする（巨大な `scatter_add_` でスコア行列を構築しない）。
4. スコア計算は “クエリの複製” を避け、L2 ではキャッシュしたノルムを使う。  
   - L2: `||x - q||^2 = ||x||^2 + ||q||^2 - 2 (x @ q)`（`||x||^2` は `packed_norms` を使用）  
   - IP: `x @ q`
5. 各クエリで `torch.topk(k)` を取り、不足分は距離 `inf/-inf` と ID `-1` で埋める。
- `range_search` も同じ gather 結果を使い、半径条件でフィルタした件数を `lims` に反映する（可変長出力）。
- `max_codes > 0` の場合は、probe 順に inverted list を走査し、クエリあたりのスキャン件数が `max_codes` を超えないように候補数を切り詰める（Faiss の `max_codes` に相当）。
- AMP/Autocast 中でも挙動が変わらないよう、入力 dtype に合わせて計算、内部の centroid や統計は float32 で保持。
- `torch.compile` に依存せず PyTorch eager 実装のみで動くことを保証する。

#### 5.5.1 性能ボトルネックの分解（設計メモ）
本プロジェクトの主戦場は「距離計算」ではなく、以下の 2 つに分解できる。

- **(A) 候補参照（index/gather）の重さ**  
  `aten::index` / `index_select` が支配的になる場合、アクセスがランダム/非連続になり、GPU（特に iGPU / UMA）でメモリ帯域が詰まりやすい。
- **(B) topk の重さ**  
  候補数が大きいほど、内部で「選別＋並べ替え＋読み書き」が増える。`k=20` でも候補が数万〜数十万だと重い。

よって次の設計目標は以下に分解する。

- **A: gather を slice に寄せる**（DB を list 順に再配置し、検索時は `db_reordered[begin:end]` で読む）
- **B: topk を一発巨大 topk からオンライン topk へ**（list/chunk ごとに小さな topk を取り、merge を繰り返す）

#### 5.5.2 `max_codes` は近似ノブ（設計メモ）
`max_codes` は「1クエリあたりに読み（スキャン）に行くベクトル数を削る近似ノブ」であり、recall が落ちるのは正常。

大工事で狙う改善は 2 種類に整理できる。

1. **近似は維持（max_codes を使う）**が、パディング/ランダムアクセスを消して「同じ max_codes で速くする」
2. **近似を減らす（max_codes を上げる）**代わりに、候補表現と選択方法を変えて「recall を保ったまま速度を戻す」

### 5.6 Flat Search（実用的な全探索）
`IndexFlat*` は “正解生成（GT）” や “IVF の回帰（nprobe=nlist）” の土台になるため、`nb` が大きいケースでも OOM せず安定して動くことを優先する。

#### 5.6.1 Flat の search_mode（matrix / chunked / auto）
- `IndexFlat.search_mode` は `"matrix" | "chunked" | "auto"`。
- `"matrix"`: `scores = Q @ X.T` / `dist = q2 + x2 - 2 Q@X.T` を **全件行列**で作り `topk`（最速になりやすいが OOM しやすい）。
- `"chunked"`: `X` をチャンク分割して距離→local topk→merge（OOM 回避、GPU/ROCm で安定）。
- `"auto"`: `nq * nb * sizeof(dtype)` が閾値を超えるときだけ `"chunked"` に落とす（tiny-batch では `"matrix"` を優先）。

#### 5.6.2 L2/IP 共通骨格（chunk→local topk→merge）
- IP: `scores = Q @ X.T`、各チャンクで `topk(k, largest=True)`。
- L2: `||q-x||^2 = q2 + x2 - 2 (Q @ X.T)`、各チャンクで `topk(k, largest=False)`。
- L2 の `x2=||x||^2` は `add/add_with_ids` 時に前計算し、検索時の再計算を避ける。

#### 5.6.3 受け入れ基準（Flat）
- `chunked` / `auto` の `search` が faiss-cpu `IndexFlatL2/IP` と一致（CPU の単体テストで担保）。
- `auto` は tiny-batch では `"matrix"`、大規模では `"chunked"` を選べる（閾値を調整可能）。

#### 5.5.3 大工事の方向性（vNext）
IVFFlat で「probed lists の中で厳密に top-k」を求める限り、原理的には **その list 内の全ベクトル距離**を読む必要がある。
したがって「無駄読みを減らす」の主成分は以下である。

- パディング（架空候補）をなくす
- gather を slice に寄せる
- topk のやり方を変えてメモリアクセスを減らす（オンライン/階層化）

提案する vNext の骨格（CSR/slice + online topk）:

1. `top_lists: (nq, nprobe)` を得る
2. list を連続配置して **slice** で読む（候補参照の gather を消す）
3. list（または chunk）単位で距離→`topk(k)` を取り、`best_k` をオンライン merge（`2k→k`）する

##### 設計目標
- **A: 候補参照（index/gather）を消す**  
  `index_matrix → gather` を廃止し、list を連続配置して slice で読む（`db_reordered[a:b]`）。
- **B: topk を “巨大 1 回” から “小さい繰り返し” へ**  
  `topk(B, max_candidates)` を廃止し、list/ベクトルチャンクごとの `topk(k)` と merge を回す。

##### データレイアウト（Index 構築後に保持）
必須:
- `centroids: FloatTensor [nlist, d]`
- `db_reordered: FloatTensor [nb, d]`（list ごとに連続配置）
- `db_norm2: FloatTensor [nb]`（L2 の `||x||^2` 事前計算）
- `ids_reordered: IntTensor [nb]`（元ID）
- `offsets: IntTensor [nlist+1]`（CSR の prefix sum。`offsets[l]:offsets[l+1]` が list l）

任意（coarse を GEMM 形に寄せる場合に有用）:
- `centroid_norm2: FloatTensor [nlist]`

※ 現行実装の対応:  
`db_reordered == packed_embeddings` / `db_norm2 == packed_norms` / `ids_reordered == list_ids` / `offsets == list_offsets`。

※ 実装スイッチ:  
`IndexIVFFlat.search_mode` で検索パスを切り替える（既定: `"matrix"`）。

- `"matrix"`: 既存の固定形状 `index_matrix` + batched gather + 1 回の巨大 `topk`。
- `"csr"`: CSR/slice + online topk（vNext）。
- `"auto"`: GPU（ROCm/CUDA）のみ自動分岐。`avg_group = (nq * nprobe / nlist)` を用い、`avg_group >= auto_search_avg_group_threshold` のとき `"csr"`、それ以外は `"matrix"`。

補足:
- `auto_search_avg_group_threshold` は既定 `8.0`（`nlist=512, nprobe=32` なら `nq>=128` で `"csr"` を選ぶ）。
- CPU では `"auto"` は `"matrix"` と同等に扱う（安定性優先）。

#### 5.5.4 小バッチ最適化（vNext.1）
CSR/slice + online topk は大きい `nq` で GPU に乗りやすい一方、`nq` が小さいケース（例: `nq<=512`）では **list 単位の処理が細かすぎて kernel launch が支配的**になりやすい。
典型的には以下の形で遅くなる。

- 1 list あたりの `GEMM` が小さすぎる（`g×d` と `L×d` が小さく、起動コストの比率が上がる）
- `topk/merge` が “小さいテンソルに対して多数回” 発生する
- Python ループや同期点（host↔device）が相対的に目立つ

このため vNext.1 では、小バッチ時に限り **起動回数を減らす**方向の最適化を追加する。
設計の原則は「大バッチ（例: `nq=19600`）の高速パスを変えず、条件付きで小バッチだけ別ルートを使う」。

評価の前提（“曲線”で境界を見る）:
- `nq=1/8/32/128/512/2048/19600` を最低セットとして計測し、「どこから伸び始めるか」の境界を同定する。

分岐条件（`nq` 固定ではなく汎用指標）:
- CSR の実効的な “1 list あたりのクエリ数” は概ね `avg_group_size = (B * nprobe / nlist)` で説明できる。
- vNext.1 の分岐は `avg_group_size < threshold` のときだけ“小バッチ用ルート”を有効化する（他パラメータでも破綻しにくい）。

候補となる実装方針（いずれも eager / PyTorch の範囲で完結）:

1. **案A（安全）: “同じ起動回数で速くする”**
   - list ループ内でのテンソル生成を削減（再利用、in-place、不要な `index_select` の削減）
   - merge の回数/形状を安定化（`k` が小さい前提の `2k→k` を維持しつつ、無駄な pad を避ける）
2. **案A'（推奨）: online merge の回数削減（小バッチ用バッファリング）**
   - list ごとの `local_topk(k)` は維持しつつ、`best` への `cat+topk` merge を毎回行わない
   - `buf[B, nprobe, k]` に “local topk の結果” を書き溜め、最後に `buf.view(B, nprobe*k)` に対して `topk(k)` を 1 回だけ実行する
   - メモリは `B*nprobe*k` 増えるため、小バッチ条件（`avg_group_size < threshold` 等）でのみ有効化する
3. **案B（次段）: list-block batching**
   - 複数 list を “ブロック” にまとめ、`GEMM/topk` をブロック単位で実行して起動回数を減らす（より大きい改修）
   - 小バッチのみ有効化し、`nq` が十分大きい場合は既存 CSR パスを維持する

受け入れ基準（定量）:
- 小バッチ（例: `nq=512`）で `csr` の search_ms が改善する
- 大バッチ（例: `nq=19600`）で `csr` の search_ms が悪化しない（少なくとも QPS -5% 以内）

実装メモ（起動回数削減）:
- `tasks_q`（list で sort 済み）に対して `q_tasks = Q[tasks_q]` を 1 回だけ作り、各 list の処理は `q_tasks[start:end]` の slice で取り出す（list ごとの `index_select` を避ける）。

（案A'）バッファリングの擬似コード:
```python
# 小バッチ専用（例: avg_group_size < threshold）
# 前提: top_lists は [B, nprobe] の prefix 適用済み

buf_scores = full((B, nprobe, k), fill)         # float
buf_packed = full((B, nprobe, k), -1)           # int64 (packed index)

for probe_j in range(nprobe):
    # list_id_per_q: [B]
    l = top_lists[:, probe_j]
    # group by list_id and process as usual, but write results into buf[:, probe_j, :]
    # (実装は tasks/group を使い回す)
    for each list group:
        cand_scores, cand_packed = local_topk_for_that_list(group_queries, list_slice)
        buf_scores[group_queries, probe_j, :] = cand_scores
        buf_packed[group_queries, probe_j, :] = cand_packed

# 最後に 1 回だけ topk
flat_scores = buf_scores.view(B, nprobe * k)
flat_packed = buf_packed.view(B, nprobe * k)
best_scores, pos = topk(flat_scores, k)
best_packed = gather(flat_packed, pos)
```

##### Search パイプライン（クエリ `Q: [B, d]`）
**Step 1) coarse（probe list を決める）**  
L2 は cdist ではなく GEMM 形で作る（ROCm では rocBLAS に寄せる）。

`||q - c||^2 = ||q||^2 + ||c||^2 - 2 (q · c)`

出力:
- `list_ids: IntTensor [B, nprobe]`（近い順に上位 nprobe）
- `list_dist: FloatTensor [B, nprobe]`（任意・デバッグ用）

**Step 2) max_codes に基づく prefix 選択（読む list を絞る）**  
Faiss 互換寄せの素直な定義（推奨）:
- list を近い順に足し上げて、次の list を足すと `max_codes` を超えるならそこで止める（prefix）。
- 0 件回避で先頭 list は必ず読む。

備考:
- 「最後の list を途中まで読む（partial scan）」は意味が変わるため、vNext では **オプション扱い**（既定は list 境界で止める）。

**Step 3) タスク化（query×list のペア列）**  
`(query_id, list_id)` のタスク列 `T` を作り、`list_id` で sort して list 単位にまとめて処理できるようにする。

**Step 4) list 単位に処理（slice→matmul→local topk→merge）**  
同じ list を要求するクエリ集合 `G` をまとめて処理する。
- list の候補ベクトルは slice で連続取得（gather を避ける）
- 距離は `Q[G] @ X.T`（GEMM）で作る
- `local_topk(k)` を取って `best_k` にオンライン merge

##### 擬似コード（実装の骨格）
**タスク作成（GPU 上で完結する版）**
```python
def build_tasks(list_ids, offsets, max_codes: int):
    # list_ids: [B, nprobe] int64
    B, nprobe = list_ids.shape

    # sizes: [B, nprobe]
    sizes = offsets[list_ids + 1] - offsets[list_ids]

    if max_codes > 0:
        cum = sizes.cumsum(dim=1)          # [B, nprobe]
        keep = cum <= max_codes            # prefix condition
        keep[:, 0] = True                  # at least one list
    else:
        keep = torch.ones_like(list_ids, dtype=torch.bool)

    tasks_l = list_ids[keep]               # [T]
    q_idx = torch.arange(B, device=list_ids.device).unsqueeze(1).expand(-1, nprobe)
    tasks_q = q_idx[keep]                  # [T]

    # group by list (sort)
    perm = tasks_l.argsort()
    return tasks_q[perm], tasks_l[perm]
```

**オンライン merge（k が小さい前提で `2k→k`）**
```python
def merge_topk(best_d, best_i, G, cand_d, cand_i, k):
    # best_d/best_i: [B, k]
    # G: [g] query indices
    # cand_d/cand_i: [g, k]
    md = torch.cat([best_d[G], cand_d], dim=1)  # [g, 2k]
    mi = torch.cat([best_i[G], cand_i], dim=1)  # [g, 2k]
    new_d, sel = md.topk(k, largest=False, dim=1)
    best_d[G] = new_d
    best_i[G] = mi.gather(1, sel)
```

**search 本体（list 単位に slice→GEMM→local topk→merge）**
```python
@torch.no_grad()
def search_ivf_csr(
    Q, k, nprobe, max_codes,
    centroids, centroid_norm2,
    db_reordered, db_norm2, ids_reordered, offsets,
    vec_chunk=4096,
):
    # Q: [B, d]
    B, d = Q.shape
    device = Q.device

    # --- coarse ---
    q2 = (Q * Q).sum(dim=1)                          # [B]
    prod = Q @ centroids.T                           # [B, nlist]
    dist_c = q2[:, None] + centroid_norm2[None, :] - 2.0 * prod
    _, list_ids = dist_c.topk(nprobe, largest=False, dim=1)         # [B, nprobe]

    # --- tasks ---
    tasks_q, tasks_l = build_tasks(list_ids, offsets, max_codes)    # [T], [T] sorted by list
    T = tasks_l.numel()

    # --- init best ---
    best_d = torch.full((B, k), float("inf"), device=device)
    best_i = torch.full((B, k), -1, device=device, dtype=torch.long)  # reordered idx

    # --- group boundaries (tasks_l is sorted) ---
    change = torch.ones((T,), device=device, dtype=torch.bool)
    change[1:] = tasks_l[1:] != tasks_l[:-1]
    starts = torch.nonzero(change).flatten()
    ends = torch.cat([starts[1:], torch.tensor([T], device=device)])

    for s, e in zip(starts.tolist(), ends.tolist()):
        l = int(tasks_l[s].item())
        G = tasks_q[s:e]                              # [g]
        a = int(offsets[l].item())
        b = int(offsets[l + 1].item())

        # vector chunking to avoid huge [g, L]
        for p in range(a, b, vec_chunk):
            q = min(p + vec_chunk, b)
            X = db_reordered[p:q]                     # [C, d] slice
            X2 = db_norm2[p:q]                        # [C]
            prod = Q[G] @ X.T                         # [g, C]
            dist = q2[G][:, None] + X2[None, :] - 2.0 * prod
            cand_d, cand_j = dist.topk(k, largest=False, dim=1)      # [g, k]
            cand_i = (p + cand_j).to(torch.long)                     # [g, k] reordered idx
            merge_topk(best_d, best_i, G, cand_d, cand_i, k)

    out_ids = ids_reordered[best_i.clamp_min(0)]
    out_ids = torch.where(best_i < 0, torch.full_like(out_ids, -1), out_ids)
    return best_d, out_ids
```

##### 実装メモ（ハマりやすい所）
- coarse も GEMM 形に（`cdist` は避ける）。ただし `nlist` が小さい場合、支配的になるとは限らない。
- `merge_topk` は scatter 的だが、`k` が小さい前提なので支配になりにくい（主戦場は A/B）。
- list の平均長が小さい（例: `262144/512≈512`）なら、`vec_chunk` は実質 1 回で済むことが多い。
- `tasks_l.argsort()` は `T≈B*nprobe` 程度なので、チャンクを適切に刻めば許容範囲。
- 計測は必ず `torch.cuda.synchronize()`（ROCm でも必要）。

##### 受け入れ条件（docs/plan 用）
- `max_codes=0` で結果の `recall@k` が旧実装（現行パス）と一致する（浮動小数差は許容）。
- profiler 上で `aten::index`（ランダム gather）が支配的でなくなる（slice ベースになっている）。
- `topk` の対象が「巨大 `max_candidates`」から「list/チャンク（~512）＋ `2k→k` merge」に変わっている。
- `max_codes` sweep の曲線形状が維持され、`nq` を上げたときの QPS スケールが改善する。

### 5.6 Serialization
- `state_dict` 形式:  
  `{"d": int, "metric": str, "nlist": int, "nprobe": int, "max_codes": int, "centroids": Tensor, "packed_embeddings": Tensor, "packed_norms": Tensor, "list_ids": Tensor, "list_offsets": Tensor}`。  
  互換性のため `packed_norms` が無い場合はロード時に再計算する。
- `save(path)` / `load(path)` は `torch.save` / `torch.load` の薄いラッパー。
- 今後 `torch_ivf.adapters.faiss` で Faiss 互換フォーマットを追加予定。

### 5.7 Benchmark artifacts
- `scripts/benchmark.py`（torch-ivf）と `scripts/benchmark_faiss_cpu.py`（faiss-cpu）で取得した結果を JSON Lines (`benchmarks/benchmarks.jsonl`) に追記する。
- `scripts/benchmark_sweep_max_codes.py` は `max_codes` の速度/精度（自己比較）を複数点で記録する用途で使う。
- レコードには `library`, `device`, `backend`, `metric`, `dim`, `nb`, `nq`, `nlist`, `nprobe`, `max_codes`, `topk`, `dtype`, `train_ms`, `add_ms`, `search_ms`, `search_ms_min`, `warmup`, `repeat`, `qps`, `torch_version`, `python_version`, `host_cpu`, `host_os`, `timestamp`, `label` を含める。
- レコードには `train_n`（学習点数）も含める（`--train-n` で指定、0 の場合は自動）。
- `scripts/benchmark_sweep_max_codes.py` のレコードは追加で `recall_at_k_vs_unlimited`（同一ライブラリの `max_codes=0` を基準とした recall@k）を含める。
- `scripts/benchmark.py` は CUDA/ROCm デバイスの場合、メモリ統計も含める（CPU の場合は `null`）。
  - `device_total_memory_bytes`
  - `mem_allocated_bytes`, `mem_reserved_bytes`
  - `train_peak_allocated_bytes`, `train_peak_reserved_bytes`
  - `add_peak_allocated_bytes`, `add_peak_reserved_bytes`
  - `search_peak_allocated_bytes`, `search_peak_reserved_bytes`
- README の代表表は `benchmarks.jsonl` の最新レコードに合わせる。

## 6. Performance Targets & Measurement
- 1M (d=128, float32) で CUDA/ROCm 双方 5ms 未満 (10-NN, nprobe=16) を目標。
- CPU では 100ms 以内（Ryzen AI Max+ 395 のような高帯域 CPU を想定）。
- メモリオーバーヘッドは Faiss `IndexIVFFlat` + 10% に抑える。
- 代表構成 (`nb=262144, nq=512, nlist=512, nprobe=32, k=20`) を常設し、CPU / ROCm / faiss-cpu の 3 つを必ず測定する。

## 7. Compile Policy
- `torch.compile` / `Index.compile()` に頼らない。Windows + ROCm でのコンパイル失敗（MSVC Unicode 問題等）を避けるため。
- もし Triton/TorchInductor を導入する場合は「任意の高速化」扱いとし、既存 API を一切変えない。

## 8. Testing Strategy
- `uv run pytest` が唯一のテストエントリーポイント。すべての進捗チェックはこのコマンドが成功してから行う。
- ゴールデンテスト: Faiss (CPU) の `search` 結果と一致することを確認する。
- 各デバイス（CPU / CUDA / ROCm / DML）で `train → add → search` の round-trip を実行するテストを用意する。
- `torch.use_deterministic_algorithms(True)` でも通るよう、乱数に `torch.Generator` + seed を使う。

## 9. Migration Notes
- `faiss.IndexFlatL2(d)` → `torch_ivf.index.IndexFlatL2(d)` のように置き換える。Tensor は PyTorch Tensor に統一。
- `index = index.to(torch.device("cuda:0"))` が `faiss.index_cpu_to_gpu` に相当。
- Faiss のユーティリティ (`faiss.normalize_L2`) は自前で `torch.nn.functional.normalize` を呼ぶ。

## 10. Open Questions
1. Faiss バイナリ互換シリアライズはどの段階で提供するか。
2. Triton / TorchInductor kernel をどこまで標準搭載にするか。
3. DirectML デバイスでの `torch.topk` パフォーマンスをどう補うか。

## 11. Next Steps
1. Batched search のチャンクヒューリスティクスをプロファイルし、入力サイズに応じた動的調整を導入する。
2. `torch_ivf.adapters.faiss` を追加し、`from_faiss` / `to_faiss` を提供する。
3. Triton / TorchInductor ベースのカスタム距離計算カーネルを PoC し、GPU でのレイテンシ改善を図る。
4. 各大きな実装変更後に本仕様と `docs/plan.md` を更新し、ドキュメントと実装を同期させる。
