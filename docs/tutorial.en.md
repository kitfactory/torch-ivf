# torch-ivf Tutorial (For Users)

This tutorial is for users who want to run embedding search with PyTorch. It walks through the basics of `torch-ivf` (Flat / IVFFlat) with minimal steps.

## 0. Prerequisite: Install PyTorch first

`torch-ivf` does not force-install PyTorch. You must choose the right PyTorch build for your environment (CUDA/ROCm/DirectML/CPU) and install it first, then install `torch-ivf`.

- If you already have PyTorch installed:
  ```bash
  pip install torch-ivf
  ```
- If you want a quick CPU-only setup (install PyTorch via pip):
  ```bash
  pip install "torch-ivf[pytorch]"
  ```

## 1. Quick smoke test (synthetic vectors)

You can run the repository example as-is:

```bash
python examples/ivf_demo.py --device cpu --verify
python examples/ivf_demo.py --device cuda --verify
```

`--verify` compares against `IndexFlat*` on CPU as a basic correctness check (not a performance check).

## 2. Flat (exhaustive search)

Flat is useful for small datasets, ground-truth generation, and validating IVF settings.

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

IVF requires `train` (k-means) to learn coarse centroids, then `add` to insert vectors.

```python
import torch
from torch_ivf.index import IndexIVFFlat

d = 128
xb = torch.randn(200000, d, device="cuda")
xq = torch.randn(4096, d, device="cuda")

index = IndexIVFFlat(d, nlist=512, nprobe=32, device="cuda")
index.train(xb[:20480])
index.add(xb)
D, I = index.search(xq, k=20)
```

## 4. `search_mode` (matrix / csr / auto)

`IndexIVFFlat.search_mode` selects the search path:

- `matrix`: build a fixed-shape candidate matrix and run one large `topk`
- `csr`: assume list-reordered storage and process lists via slice + online topk (many small `topk`)
- `auto`: on GPU, choose `csr` when `nq*nprobe/nlist` is large enough (otherwise prefer `matrix` for tiny batches)

## 5. `max_codes` (Faiss-compatible approximation knob)

`IndexIVFFlat.max_codes` caps the number of scanned candidates per query (`0` means unlimited).
It can stabilize runtime when inverted lists are imbalanced, at the cost of recall.

```python
index.max_codes = 16384
D, I = index.search(xq, k=20)
```

## 6. Performance pitfalls

- **Transfers included in timing**: avoid moving tensors CPU→GPU every call. Create tensors on the target device or move once and reuse.
- **Tiny batches may not benefit from GPU**: GPU kernel-launch overhead can dominate. Use `search_mode="auto"` or batch more queries.
- **Too-small `train_n`**: small training sets can produce imbalanced lists and inflate candidate counts. As a rule of thumb, use `train_n >= 40*nlist`.

