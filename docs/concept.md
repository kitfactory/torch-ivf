# torch-ivf Concept

## Motivation
- ユーザーは GPU ベンダーごとに `faiss-gpu` や AMD 専用実装に切り替える必要があり、運用負荷が高い。
- Windows + ROCm 版 PyTorch の組み合わせでは選択肢が限られ、同一コードで GPU IVF を利用できない。
- PyTorch 原生の実装であれば CPU から CUDA / ROCm / DirectML にまたがる幅広いデバイスを単一コードで扱え、依存構築の難易度も下がる。

## ゴール
1. **PyTorch 100% 実装の IVF**: 近似最近傍探索 (ANN) で一般的な Inverted File Index を PyTorch Tensor と Autograd だけで完結させる。
2. **マルチベンダー GPU**: CUDA / ROCm / DirectML / CPU を同 API で利用。特に Windows + ROCm PyTorch での動作を最優先。
3. **シンプルな依存関係**: `pip install torch-ivf` 後に追加のネイティブビルドなしで使える。
4. **統一インターフェース**: FAISS ライクな `index.train`, `index.add`, `index.search` を提供し既存ワークロードを移行しやすくする。

## 非ゴール
- PQ / HNSW など IVF 以外の複雑な ANN 手法の完全再実装。
- デバイス固有最適化 (例: CUDA kernel のみを前提にした warp 最適化)。必要になれば TorchScript / Triton で拡張する。

## コア設計
1. **Coarse Quantizer**  
   - PyTorch 実装の k-means (mini-batch, cosine/L2 切り替え可)。  
   - 学習は `torch.cuda`/`torch.rocm` などデバイスに依存せず同じループを使用。
2. **Inverted Lists**  
   - 各 centroid に対応するベクトル集合を `packed_embeddings` と `list_offsets` に保持。  
   - torch_scatter 相当の更新は `index_add_`, `gather`, `sort` でエミュレートしデバイス互換性を保つ。
3. **Search パス**  
   - クエリはまず coarse で上位 `nprobe` セントロイドを取得 (`topk` + 距離計算)。  
   - 該当 inverted list をバッチで展開し、`torch.cdist` / `einsum` を使って再ランキング。  
   - すべて PyTorch OP なので AMP / Autograd / TorchScript とも親和性あり。
4. **メタデータ互換**  
   - ID, メタ情報は別 Tensor で保持。FAISS の `IndexIVFFlat` に相当する `torch_ivf.IndexIVFFlat` を MVP とし、将来的に `IVFPQ` を追加。

## デバイス戦略
- まず CUDA / ROCm / CPU を CI でカバー。Windows + ROCm はローカル実機テスト、Linux + ROCm は GitHub Actions 上で rocBLAS 付きコンテナを想定。
- DirectML は `torch.set_default_device("dml")` で動作確認するが、パフォーマンス最適化は後続タスク。

## 里程標
1. **MVP**
   - `IndexIVFFlat` クラスと簡易 CLI/Notebook を提供。
   - CIFAR-10 などで embedding を生成し検索できる E2E サンプルを `examples/` に追加。
2. **ベンチマーク**
   - CUDA vs ROCm vs CPU で同一スクリプトを回し、レイテンシ/メモリ/精度を測定。
3. **互換レイヤ** 
   - FAISS 互換のシリアライザ、`from_faiss()` / `to_faiss()` の PoC。
4. **最適化**
   - Triton / TorchInductor を使った距離計算カーネル差し替え。

## 性能改善フェーズ（Eager-only / Windows+ROCm 維持）
本フェーズでは **PyTorch eager + 標準演算のみ** で、小バッチ性能を中心に改善する。大バッチの throughput を悪化させないことを前提とする。

### 機能一覧（Phase: P1）
| Spec ID | 機能 | 目的 | 依存関係 | MVP / フェーズ |
|---|---|---|---|---|
| PERF-0 | P1 共通制約（Eager-only/拡張なし） | 互換性と環境依存の回避 | 全体 | P1 |
| PERF-1 | 派生テンソルのキャッシュ | 再計算を抑え、小バッチ固定費を削減 | `IndexIVFFlat` | P1 |
| PERF-2 | CSR 経路の同期削減 | `.cpu()/.tolist()/.item()` を最小化 | `IndexIVFFlat.search(csr)` | P1 |
| PERF-3 | ワークスペース再利用 | alloc/fill を減らし起動回数を抑制 | `IndexIVFFlat.search(csr)` | P1 |
| PERF-4 | small-batch buffered 改善 | 小バッチの起動回数削減を最優先 | `IndexIVFFlat.search(csr)` | P1 |
| PERF-5 | ベンチ再現性の強化 | 変更前後の比較を容易にする | `scripts/benchmark*.py` | P1 |
| PERF-6a | 速度優先デフォルトON（安全） | 結果不変の範囲で速度を上げる | `IndexIVFFlat.search` | P1 |
| PERF-6b | 近似モード（デフォルトOFF） | recall 低下を許容して高速化 | `IndexIVFFlat.search` | P1 |

### フェーズ方針
- P1 は **API 互換・依存追加なし** を必須とし、Windows + ROCm で壊れないことを最優先する。

## リスクと対策
- **PyTorch 実装の性能不足**: Triton kernel や専用カーネルでホットスポットを最適化できる余地がある。必要に応じてバックエンドを差し替える抽象化を先に整備。
- **ROCm on Windows の成熟度**: 新しい ROCm リリースで API が変わる可能性。CI に nightly ROCm PyTorch を追加し早期検知。
- **巨大データのインメモリ保持**: 分割ロードやメモリマップを提供し、PyTorch の `tensor.share_memory_()` を利用してプロセス間で共有。

## 今後のドキュメント
- `docs/architecture.md`: モジュール図とクラス設計。
- `docs/usage.md`: 典型的なワークフロー (学習 / 追加 / 検索)。
- `README.md`: プロジェクト概要と getting started。

このドキュメントは初期コンセプトをまとめたものなので、実装進捗に合わせて更新してください。
