# docs/OVERVIEW.md（入口 / 運用の正本）

この文書は **プロジェクト運用の正本**です。`AGENTS.md` は最小ルールのみで、詳細はここに集約します。

---

## 現在地（必ず更新）
- 現在フェーズ: P0
- 今回スコープ（1〜5行）:
  - Throughput 領域（例: `nq=19600`）で **faiss-cpu 比 15x** を達成・再現できる設定/手順を固める（代表: `nb=262144, nlist=512, nprobe=32, k=20, max_codes=0`）。
  - torch-ivf 側の `dtype=float16` も含めてベンチを “同一データで” 比較できるようにし、`benchmarks/benchmarks.jsonl` に記録する。
  - 現実装（`src/torch_ivf/index/ivf_flat.py`）と `docs/spec.md` の整合を保つ（仕様の追随）。
- 非ゴール（やらないこと）:
  - Exact float32 のみで 15x を必達にする（現状は float16 を勝ち筋に含める）。
  - Triton/TorchInductor など「専用カーネル導入」を前提にする（任意の最適化として別扱い）。
  - 近似ノブ（recall を落とす候補剪定）を暗黙に有効化する（使う場合は明示し、品質ゲートで管理）。
- 重要リンク:
  - concept: `./concept.md`
  - spec: `./spec.md`
  - architecture: `./architecture.md`
  - plan: `./plan.md`

---

## レビューゲート（必ず止まる）
共通原則：**自己レビュー → 完成と判断できたらユーザー確認 → 合意で次へ**

---

## 更新の安全ルール（判断用）
### 合意不要
- 誤字修正、リンク更新、意味を変えない追記
- plan のチェック更新
- 小さな明確化（既存方針に沿う）

### 提案→合意→適用（必須）
- 大量削除、章構成変更、移動/リネーム
- Spec ID / Error ID の変更
- API/データモデルの形を変える設計変更
- セキュリティ/重大バグ修正で挙動が変わるもの
