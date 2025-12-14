# torch-ivf Tutorial (for users)

This tutorial is for users who want to run embedding search with PyTorch. It shows the basic usage of `torch-ivf` (Flat / IVFFlat) in the shortest path.

## 0. Prerequisite (install PyTorch first)

`torch-ivf` does not force-install PyTorch. Since the right PyTorch build depends on your environment (CUDA/ROCm/DirectML/CPU), **install PyTorch first**, then install `torch-ivf`.

- If you already have PyTorch:
  ```bash
  pip install torch-ivf
  ```
- If you want a quick CPU setup (also installs PyTorch via pip):
  ```bash
  pip install "torch-ivf[pytorch]"
  ```

## 1. Run it first (synthetic data)

You can run the repository example as-is:

```bash
python examples/ivf_demo.py --device cpu --verify
python examples/ivf_demo.py --device cuda --verify
```

`--verify` compares results against CPU `IndexFlat*` as a basic correctness check (not a performance check).

## 2. Flat (exhaustive search)

Flat is useful for small datasets, ground-truth generation, and validating IVF settings.

Metric selection (rule of thumb):
- L2: use Euclidean distance on raw embeddings (common default)
- IP: often used as cosine similarity (≈ inner product) with normalized embeddings

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

## 3. IVFFlat (approximate search)

IVF runs `train` (k-means) to learn coarse centroids, then `add` to insert vectors.

```python
import torch
from torch_ivf.index import IndexIVFFlat

d = 128
device = "cuda" if torch.cuda.is_available() else "cpu"
# ROCm: from PyTorch’s perspective, `device="cuda"` is still OK
# (on ROCm builds, `torch.cuda.is_available()` is expected to be True).
# DirectML: depending on your environment, you may need a torch-directml device object.
# Follow the official instructions.
# Example (environment-dependent):
# import torch_directml
# device = torch_directml.device()

xb = torch.randn(200000, d, device=device)
xq = torch.randn(4096, d, device=device)

index = IndexIVFFlat(d=d, nlist=512, nprobe=32, metric="l2").to(device)
index.search_mode = "auto"
index.train(xb[:20480])
index.add(xb)
#
# `nprobe` is “how many lists to probe”. Larger `nprobe` improves recall but tends to slow down search.
#
with torch.inference_mode():
    D, I = index.search(xq, k=20)
```

## 4. `search_mode` (matrix / csr / auto)

`IndexIVFFlat.search_mode` selects the search path:

- `matrix`: build a fixed-shape candidate matrix, then run one large `topk`
- `csr`: assume list packing + slicing; process lists with online topk (repeated small `topk`)
- `auto`: on GPU, choose `csr` if `nq*nprobe/nlist` is large enough (otherwise prefer `matrix` for tiny batches)

If you’re unsure, using `auto` is the safest choice (it adapts to both tiny-batch and throughput regimes).

## 5. `max_codes` (Faiss-compatible approximation knob)

`IndexIVFFlat.max_codes` is the **cap of scanned candidates per query** (`0` means unlimited).
When lists are imbalanced and candidate counts inflate, `max_codes` can stabilize runtime, at the cost of stronger approximation and lower `recall@k`.

Example:

```python
index.max_codes = 16384
D, I = index.search(xq, k=20)
```

## 6. Common pitfalls (performance)

- **Transfers included in timing**: if you call `.to()` from CPU→GPU every time for `xb`/`xq`, it becomes slow. Create tensors on the target device, or move once and reuse.
- **Tiny batches can be CPU/other-path favored**: for small `nq`, GPU kernel-launch overhead can dominate. Use `search_mode="auto"` or batch more queries.
- **Too-small `train_n`**: a small `train_n` can produce imbalanced lists and inflate candidate counts. As a rule of thumb, use `train_n >= 40*nlist`.
- **Use scripts for performance comparisons**: ad-hoc timing is noisy (sync/warmup/statistics). Prefer `scripts/benchmark*.py`.
