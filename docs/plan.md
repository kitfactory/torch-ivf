# torch-ivf Plan

## 進め方ルール
- すべてのチェックは `uv run pytest` が成功した状態でのみ付ける。テストが通らない限り次の項目へ進まない。
- ベンチマーク関連タスクは **ROCm(GPU) / faiss-cpu** を優先して `scripts/benchmark*.py` を実行し、`benchmarks/benchmarks.jsonl` と `README.md` に結果を反映してからチェックする（CPU は任意）。
- 依存追加は `uv add --dev pytest faiss-cpu` を必ず使用する（別手段は禁止）。

## 現状（ざっくり）
- ✅ Foundations / IVF Core / Bench infra は成立
- ✅ IVF search は `matrix` → `csr`（slice + online topk）で大幅改善
- ✅ 小バッチ対策（CSR の kernel launch 削減）も導入済み
- ✅ Flat は存在し、OOM 回避の実用化も完了（chunked/auto）

---

## Now（次に潰す：優先順）

### 0) P1 性能改善（Eager-only / Windows+ROCm 維持）
- [x] Task A: キャッシュ層の追加（PERF-1）
  - 成果物: `src/torch_ivf/index/ivf_flat.py`, `tests/test_index_ivf_flat.py`
  - `train/add/reset/max_codes` で `_invalidate_search_cache()` を呼ぶ
  - `centroids_T`, `centroid_norm2`, `list_sizes`, `effective_max_codes` をキャッシュ
- [x] Task B: CSR search の同期削減（PERF-2）
  - 成果物: `src/torch_ivf/index/ivf_flat.py`
  - `.cpu()/.tolist()/.item()` の回数を削減し、必要なら同期 1 回に集約
- [x] Task C: ワークスペース再利用（PERF-3）
  - 成果物: `src/torch_ivf/index/ivf_flat.py`, `tests/test_index_ivf_flat.py`
  - 再利用バッファ（scores/packed/buf/best）を保持し、容量拡張のみ許容
  - `device` / `dtype` 変更時の再確保を明示
- [x] Task D: small-batch 閾値の定数化と説明（PERF-4）
  - 成果物: `src/torch_ivf/index/ivf_flat.py`, `docs/spec.md`（根拠と閾値）
  - `avg_group_size` 閾値を定数化し、ベンチで妥当性を確認
- [x] Task E: テスト追加（キャッシュ invalidate / ワークスペース影響なし）
  - 成果物: `tests/test_index_ivf_flat.py`
- [x] Task F: ベンチ更新（変更前後比較）
  - 成果物: `benchmarks/benchmarks.jsonl`, `README.md`（必要なら）
  - `scripts/benchmark_sweep_nq.py` / `scripts/benchmark.py` を使用
- [x] Task G: 速度優先デフォルトON（PERF-6a）/近似モードOFF（PERF-6b）の境界を実装に反映
  - 成果物: `src/torch_ivf/index/ivf_flat.py`, `docs/spec.md`
  - デフォルト ON は結果不変の最適化のみとし、近似ノブは明示的に有効化した場合のみ適用

### 1) CSR vNext の仕上げ（互換・説明・残タスク）
- [x] **max_codes の prefix ルールを faiss-cpu と合わせる**
  - 既定: 先頭 list は必ず読む / list 境界で止める
  - 成果物: `docs/spec.md`
- [x] 仕様テスト: max_codes の “境界ケース” で採用 list 数が期待どおりになるユニットテストを追加する（例: cum が max_codes-1 / max_codes / max_codes+1）。
  - 成果物: `tests/test_index_ivf_flat.py`
- [x] fast-path: max_codes>0 でも “全クエリが剪定されない” 場合は max_codes=0 と同じルートに落とす（無駄計算を避ける）。
  - 成果物: `src/torch_ivf/index/ivf_flat.py`
- [x] 受け入れ条件: `max_codes=0` で旧パスと `recall@k` が一致（浮動小数差は許容）。
- [x] 受け入れ条件: profiler で `aten::index`（ランダム gather）が支配的でなくなる（slice ベース）。
- [x] ROCm(GPU) / faiss-cpu のベンチ（`nq=19600` と `max_codes` スイープ）を再取得し、`benchmarks/benchmarks.jsonl` / README を更新する。
  - 成果物: `benchmarks/benchmarks.jsonl`, `README.md`

### 2) add の高速化（第3優先）
- [x] 大きい追加バッチ時の CPU/ROCm add_ms をベンチで比較し、特に ROCm 側で改善が出るバッチサイズを詰める。
- [x] `uv run pytest` と CPU/ROCm ベンチの再取得、`benchmarks/benchmarks.jsonl` / README 更新。
  - 成果物: `benchmarks/benchmarks.jsonl`, `README.md`

### 3) ヒューリスティクス見直し（動的 chunk / 動的候補上限）
- [x] 目標: `nb/nq/nlist/nprobe` の変更で極端に遅くならないこと（ベンチで確認）。
- [x] `uv run pytest` と CPU/ROCm ベンチの再取得、`benchmarks/benchmarks.jsonl` / README 更新。
  - 成果物: `benchmarks/benchmarks.jsonl`, `README.md`, `docs/spec.md`（閾値の根拠）

### 4) 小バッチ高速化（CSR search の kernel launch 削減）※必要なら追加の詰め
- [x] 小バッチ（例: `nq=512`）のプロファイルを取得し、「(i) 小さすぎる GEMM」「(ii) topk/merge 起動回数」「(iii) Python ループ/同期点」の支配を分解する。
- [ ] 実装案B（次点）: list-block batching（複数 list をまとめて処理）で GEMM/topk の起動回数を減らす（小バッチのみ）。
- [x] 受け入れ条件（大バッチ保護）: `nq=19600` の `csr` が **現状から劣化しない**（少なくとも QPS -5% 以内）。
  - 成果物: `benchmarks/benchmarks.jsonl`, `README.md`

### 5) 近似/設定強化（PERF-6c / PERF-6b / PERF-6a.1 / PERF-6a.2）
- [ ] SearchParams / profile（PERF-6c）
  - `search(xq, k, *, params=None)` を追加し、後方互換を維持
  - `profile=exact/speed/approx` の既定と優先順位を整理
  - `candidate_budget` 等の入力バリデーションを追加
- [ ] Safe pruning（PERF-6a.1）
  - `list_radius` を保持し、`lb(l)` による list 単位スキップを実装（L2 のみ）
- [ ] Candidate budgeting strategy（PERF-6b.1）
  - `candidate_budget`, `budget_strategy`（uniform/distance_weighted）を実装
  - `min_codes_per_list`, `max_codes_cap_per_list`, `strict_budget`, `dynamic_nprobe` を追加
  - `max_codes`/`nprobe` との優先順位ルールを実装
- [ ] In-list ordering（PERF-6b.2）
  - `list_ordering`（residual_norm_asc / proj_desc）を実装
  - `rebuild_policy`（manual/auto_threshold）と unsorted tail 運用を明確化
- [ ] 品質ゲート（PERF-6b.3）
  - `recall@k` の下限（16k/32k/64k）を定義
  - baseline と fallback の条件を整備
- [ ] anchors プレフィルタ（PERF-6b.4）※任意
  - list ごとに anchor を保持し、見込み list を絞って配分
- [ ] subcluster bound（PERF-6a.2）※任意
  - Exact のまま subcluster 単位でスキップできるようにする

---

## Next（相互運用・オプション高速化）

### Faiss 相互運用
- [ ] `torch_ivf.adapters.faiss` を追加し、`from_faiss` / `to_faiss` を通じて Faiss との相互運用を確立する。

### Optional: 専用カーネル
- [ ] Triton / TorchInductor ベースの専用距離計算カーネルを導入し、GPU での search パスを高速化する（“任意の高速化” として扱い、デフォルトは eager のまま）。

---

## 定着（成果を壊さない）
- [ ] README の “勝ち筋” を固定し、再現性を上げる（速度そのものより安定化を優先）。
  - [x] tiny-batch と throughput の二相を `nq` sweep（表＋図）で明記する。
  - [x] `search_mode=auto` を正式化し、分岐条件（`avg_group = nq*nprobe/nlist`）と既定閾値を明記する。
  - [x] `benchmarks/env.json` を生成できる仕組み（`scripts/dump_env.py`）を追加し、ベンチの前提環境を固定で記録できるようにする。
  - [ ] ベンチ再現手順（最小）を README に 1 か所にまとめる（`dump_env` → `benchmark` → `benchmarks.jsonl` の更新手順）。
  - [ ] “代表値の更新ポリシー” を明文化する（例: `--warmup/--repeat` と `search_ms=median` を固定）。
  - [x] `uv run pytest` が通ることを確認してチェックする。
- [ ] 重要な実装変更ごとに本プランと `docs/spec.md` を更新し、仕様と実装を同期させる。

---

## Done（履歴：必要なら参照）
<details>
<summary>✅ Done: Foundations / Flat baseline / IVF core / Bench infra / Step1-5 / 改善サイクル / CSR vNext（基盤）</summary>

### Foundations
- [x] `uv add --dev pytest faiss-cpu` でテスト依存を導入する。
- [x] `src/torch_ivf/` スケルトンと `__init__.py` を作成する。
- [x] `pyproject.toml` を src レイアウトと `torch_ivf` パッケージ名に合わせて整備する。
- [x] `IndexBase` を実装し、共通属性と抽象メソッドをそろえる。
- [x] `IndexBase.to()/cpu()/cuda()/rocm()/dml()` で内部 Tensor を安全に移動できるようにする。

### Flat Baseline（実装/テスト）
- [x] `IndexFlatL2` / `IndexFlatIP` を実装し、Faiss 互換 API を提供する。
- [x] Faiss とのゴールデン比較テストを追加する (CPU 基準)。
- [x] **Flat を実用化（OOMしない全探索）**（chunked/auto を含む）

### IVF Core
- [x] kmeans / IVFFlat / save/load / range_search 等

### Validation & Benchmarks / Step 1-5 / 改善サイクル / CSR vNext（基盤）
- [x] （旧 plan.md のチェック済み項目一式）

</details>
