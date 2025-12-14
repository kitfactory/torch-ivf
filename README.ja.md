# torch-ivf（日本語）

torch-ivf は、Faiss の `IndexFlat` / `IndexIVFFlat` に近い API を目指しつつ、**PyTorch ネイティブ**で動作する IVF（Inverted File Index）ライブラリです。CPU / CUDA / ROCm / DirectML を同一コードで扱えることを目標にしています。開発は特に Windows + ROCm PyTorch 環境を重視しています。

- English README: `README.md`

## クイックスタート

## インストール（PyTorch は前提）

torch-ivf は PyTorch を強制インストールしません。CUDA/ROCm/DirectML/CPU など、利用環境に合った PyTorch を **先に**入れてから torch-ivf をインストールしてください。

- すでに PyTorch を入れている場合（推奨）:
  ```bash
  pip install torch-ivf
  ```
- CPU で手早く試したい場合（PyTorch も pip で入れる）:
  ```bash
  pip install "torch-ivf[pytorch]"
  ```

1. 合成データのデモを実行:
   ```bash
   python examples/ivf_demo.py --device cpu --verify
   python examples/ivf_demo.py --device cuda --verify
   ```
2. チュートリアル（利用者向け）:
   - `docs/tutorial.ja.md`
   - `docs/tutorial.en.md`

## 重要ポイント（転送オーバーヘッド削減）

- 目的 device 上でテンソルを生成する（`torch.randn(..., device=device)`）。
- `add` / `search` はできるだけ大きいバッチで呼ぶ（数千〜）。
- `index = IndexIVFFlat(...).to(device)` は 1 回だけ行い、内部バッファは同じ device に常駐させる。
- DataLoader 経由なら `pin_memory=True` と `tensor.to(device, non_blocking=True)` を使う。

## ドキュメント

- `docs/concept.md` – 背景と狙い
- `docs/spec.md` – 仕様（API/挙動）
- `docs/plan.md` – 進捗チェックリスト
- `docs/tutorial.ja.md` – チュートリアル（日本語）
- `docs/tutorial.en.md` – Tutorial (English)

## ベンチマーク

- `scripts/benchmark.py`: torch-ivf ベンチ（CPU/ROCm）。JSON を出力し `benchmarks/benchmarks.jsonl` に追記。
- `scripts/benchmark_faiss_cpu.py`: faiss-cpu 参照ベンチ。
- `scripts/benchmark_sweep_nq.py`: `nq` スイープ（小バッチ vs throughput の境界を見る）。
- `scripts/benchmark_sweep_max_codes.py`: `max_codes` スイープ（速度/自己比較 recall のトレードオフを見る）。
- `scripts/dump_env.py`: `benchmarks/env.json` を生成。
- `scripts/profile_ivf_search.py`: `IndexIVFFlat.search` の `torch.profiler` 表を表示。

## 開発（uv）

このリポジトリの開発には `uv` を使います。

```bash
uv sync
uv run pytest
```

