# torch-ivf チュートリアル（利用者向け）

このチュートリアルは「PyTorch を使って特徴量検索をしたい」利用者向けに、`torch-ivf` の基本的な使い方（Flat / IVFFlat）を最短で試せるようにまとめます。

## 0. 前提（PyTorch は先に用意）

`torch-ivf` は PyTorch を強制インストールしません。CUDA/ROCm/DirectML/CPU のどの PyTorch を使うかは利用環境ごとに異なるため、**先に PyTorch をインストール**してから `torch-ivf` を入れてください。

- すでに PyTorch を入れている場合:
  ```bash
  pip install torch-ivf
  ```
- CPU で手早く試したい場合（PyTorch も一緒に入れる）:
  ```bash
  pip install "torch-ivf[pytorch]"
  ```

## 1. まずは動かす（合成データ）

リポジトリの例をそのまま動かせます。

```bash
python examples/ivf_demo.py --device cpu --verify
python examples/ivf_demo.py --device cuda --verify
```

`--verify` は CPU の `IndexFlat*` と結果を比較して簡易チェックします（速度比較ではありません）。

## 2. Flat（全探索）

小規模・正解生成・IVF の検証などには Flat（全探索）が便利です。

メトリクス選択の目安:
- L2: そのままの特徴量で距離（一般的）
- IP: 正規化済み特徴量で cosine 類似（≒内積）として使うことが多い

```python
import torch
from torch_ivf.index import IndexFlatL2

d = 128
xb = torch.randn(10000, d)
xq = torch.randn(32, d)

index = IndexFlatL2(d)
index.add(xb)
D, I = index.search(xq, k=10)
```

## 3. IVFFlat（近似検索）

IVF は `train`（k-means）で coarse centroid を学習し、その後 `add` でベクトルを追加します。

```python
import torch
from torch_ivf.index import IndexIVFFlat

d = 128
device = "cuda" if torch.cuda.is_available() else "cpu"
# ROCm: PyTorch 的には device="cuda" のままで OK（ROCm でも torch.cuda.is_available() が True になる想定）
# DirectML: 環境によって torch-directml 側の device オブジェクトを使うことがあるため、公式手順に従ってください。
# 例（環境による）:
# import torch_directml
# device = torch_directml.device()

xb = torch.randn(200000, d, device=device)
xq = torch.randn(4096, d, device=device)

index = IndexIVFFlat(d=d, nlist=512, nprobe=32, metric="l2").to(device)
index.search_mode = "auto"
index.train(xb[:20480])
index.add(xb)
#
# nprobe は「覗きにいく list の数」です。増やすと recall は上がりやすい一方、遅くなりやすいです。
#
with torch.inference_mode():
    D, I = index.search(xq, k=20)
```

## 4. `search_mode`（matrix / csr / auto）

`IndexIVFFlat.search_mode` は検索パスを選びます。

- `matrix`: 固定形状の候補行列を作り、1回の大きい `topk` を実行
- `csr`: list 連続化 + slice を前提に、list 単位で距離計算し online topk（小さな `topk` の繰り返し）
- `auto`: GPU のときに `nq*nprobe/nlist` が十分大きければ `csr` を選ぶ（小バッチは `matrix` に寄せる）

迷ったら `auto` を使うのが安全です（小バッチ/まとめ投げのどちらでも寄せてくれます）。

## 5. `max_codes`（Faiss 互換の近似ノブ）

`IndexIVFFlat.max_codes` は **クエリあたりに走査する候補数の上限**です（0 は無制限）。
list が偏って候補が膨らむとき、`max_codes` を使うと時間が安定しやすくなりますが、近似が強くなり `recall@k` は下がります。

例:
```python
index.max_codes = 16384
D, I = index.search(xq, k=20)
```

## 6. よくある落とし穴（性能）

- **転送が計測に混ざる**: `xb`/`xq` を毎回 CPU→GPU に `.to()` していると遅くなります。テンソルは最初から目的 device で作るか、1回だけ移して使い回してください。
- **小バッチは CPU/別パスが強いことがある**: `nq` が小さいと GPU の起動コストが支配的になります。`search_mode="auto"` を使うか、クエリをまとめて投げてください。
- **`train_n` が少ない**: `train_n` が小さいと list が偏り、候補数が増えて遅くなりやすいです。目安として `train_n >= 40*nlist` を推奨します。
- **性能比較はスクリプトで行う**: 手計測だと同期/ウォームアップ/統計が揃わずブレます。`scripts/benchmark*.py` を使ってください。
