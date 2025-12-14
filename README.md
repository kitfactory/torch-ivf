# torch-ivf

torch-ivf is a PyTorch-native IVF (Inverted File Index) library that mimics the Faiss `IndexFlat` / `IndexIVFFlat` APIs while running on CPU, CUDA, ROCm, or DirectML from the same codebase. The project is developed primarily on Windows + ROCm PyTorch, so that GPU search workloads can run without switching libraries per vendor.

## Quick Start

## Installation (PyTorch prerequisite)

torch-ivf does not force-install PyTorch because you may need a specific wheel variant (CUDA/ROCm/DirectML/CPU).

- If you already have PyTorch installed (recommended for CUDA/ROCm/DirectML), install torch-ivf:
  ```bash
  pip install torch-ivf
  ```
- If you want a quick CPU-only setup, you can let pip install a compatible PyTorch:
  ```bash
  pip install "torch-ivf[pytorch]"
  ```

1. Install dependencies (project uses `uv`).
   ```bash
   uv sync
   ```
2. Run the demo with synthetic vectors.
   ```bash
   uv run python examples/ivf_demo.py --device cuda --verify
   ```
   Use `--device cpu` / `--device rocm` / `--device dml` as needed. `--verify` compares against `IndexFlat`.
3. Execute tests.
   ```bash
   uv run pytest
   ```

## Best Practices (reduce transfer overhead)

- Generate tensors directly on the target device (`torch.randn(..., device=device)`).
- Feed large mini-batches to `add` / `search` (thousands of vectors per call) to reduce PCIe transfers.
- Move the index once (`index = IndexIVFFlat(...).to(device)`) and keep all internal buffers on that device.
- When loading data via DataLoader, use `pin_memory=True` and `tensor.to(device, non_blocking=True)`.
- Keep metadata (`list_ids`, etc.) on the device; only bring final results back to the host.

```python
loader = DataLoader(ds, batch_size=4096, pin_memory=True, num_workers=4)
index = IndexIVFFlat(d, nlist=512).to(device)
for xb, _ in loader:
    xb = xb.to(device, non_blocking=True)
    index.add(xb)
```

## Device Support Matrix

| Device / API        | Status  | Notes                                            |
|---------------------|---------|--------------------------------------------------|
| CPU (x86/ARM)       | ✅       | All tests run on CPU.                            |
| CUDA (NVIDIA)       | ✅       | `IndexFlat*` / `IndexIVFFlat` operate on CUDA.   |
| ROCm (Linux)        | ✅       | Same code path, tested on ROCm builds.           |
| ROCm (Windows)      | ✅       | Primary dev environment (Windows + ROCm PyTorch).|
| DirectML (Windows)  | ⚠️ Experimental | Basic smoke tests only.                  |

## Documentation

- `docs/concept.md` – background and goals
- `docs/spec.md` – API specification and behavior
- `docs/plan.md` – checklist-style progress tracking

## Benchmarks

- `scripts/benchmark.py`: torch-ivf benchmark (CPU/ROCm). Outputs JSON with hardware metadata.
- `scripts/benchmark_faiss_cpu.py`: faiss-cpu reference benchmark.
- `scripts/benchmark_sweep_nq.py`: sweep `nq` to reveal tiny-batch vs throughput regimes.
- `scripts/dump_env.py`: dump a reproducible environment snapshot to `benchmarks/env.json`.
- `scripts/profile_ivf_search.py`: print a short `torch.profiler` table for `IndexIVFFlat.search`.

Example:
```bash
uv run python scripts/benchmark.py --device cpu --nb 32768 --nq 128 --json
uv run python scripts/benchmark.py --device cuda --nb 32768 --nq 128 --search-mode auto --json
uv run python scripts/benchmark_faiss_cpu.py --nb 32768 --nq 128
uv run python scripts/dump_env.py
```

`--max-codes` (Faiss-compatible) caps the number of scanned candidates per query to stabilize runtime when inverted lists are imbalanced (default: unlimited). In code, this corresponds to `index.max_codes`.
`--train-n` sets the k-means training sample count; for realistic IVF list balance, use `--train-n (40*nlist)` or more.

`--search-mode` chooses the search path:
- `matrix`: fixed-shape candidate matrix + one large `topk`.
- `csr`: CSR/slice + online topk.
- `auto` (GPU only): selects `csr` when `avg_group = nq*nprobe/nlist` is large enough; otherwise `matrix`.

Sample JSON (CPU):
```json
{
  "library": "torch_ivf",
  "device": "cpu",
  "backend": "CPU",
  "metric": "l2",
  "nb": 32768,
  "nq": 128,
  "train_ms": 842.317,
  "search_ms": 154.883,
  "qps": 826.452
}
```

### Representative Results (`nb=262144`, `train_n=20480`, `nq=512`, `nlist=512`, `nprobe=32`, `k=20`, `dtype=float32`, `--warmup 1 --repeat 5`)

| Library   | Device   | Backend        | search_mode | train_ms | add_ms  | search_ms | QPS      | Notes                             |
|-----------|----------|----------------|-------------|----------|---------|-----------|----------|-----------------------------------|
| torch-ivf | ROCm GPU | ROCm 7.1.52802 | matrix      | 871.547  | 913.179 | 157.057   | 3259.967 | `--device cuda` (ROCm GPU)        |
| torch-ivf | ROCm GPU | ROCm 7.1.52802 | csr         | 901.215  | 938.612 | 28.811    | 17770.929| CSR/slice + online topk (vNext)   |
| faiss-cpu | CPU      | faiss-cpu      | faiss       | 92.301   | 138.889 | 72.081    | 7103.071 | Reference (C++ implementation)    |

Raw records are stored in `benchmarks/benchmarks.jsonl`.
Environment snapshots are stored in `benchmarks/env.json`.

### Representative Results (`nb=262144`, `train_n=20480`, `nq=2048`, `nlist=512`, `nprobe=32`, `k=20`, `dtype=float32`, `--warmup 1 --repeat 5`)

| Library   | Device   | Backend        | search_mode | train_ms | add_ms  | search_ms | QPS       | Notes                             |
|-----------|----------|----------------|-------------|----------|---------|-----------|-----------|-----------------------------------|
| torch-ivf | ROCm GPU | ROCm 7.1.52802 | matrix      | 870.266  | 926.951 | 631.782   | 3241.625  | `--device cuda` (ROCm GPU)        |
| torch-ivf | ROCm GPU | ROCm 7.1.52802 | csr         | 872.731  | 939.334 | 57.546    | 35588.673 | CSR/slice + online topk (vNext)   |
| faiss-cpu | CPU      | faiss-cpu      | faiss       | 101.726  | 140.194 | 218.445   | 9375.375  | Reference (C++ implementation)    |

### Representative Results (`nb=262144`, `train_n=20480`, `nq=19600`, `nlist=512`, `nprobe=32`, `k=20`, `dtype=float32`, `--warmup 1 --repeat 5`)

| Library   | Device   | Backend        | search_mode | train_ms | add_ms  | search_ms | QPS       | Notes                             |
|-----------|----------|----------------|-------------|----------|---------|-----------|-----------|-----------------------------------|
| torch-ivf | ROCm GPU | ROCm 7.1.52802 | matrix      | 873.139  | 929.645 | 6259.560  | 3131.211  | `--device cuda` (ROCm GPU)        |
| torch-ivf | ROCm GPU | ROCm 7.1.52802 | csr         | 908.318  | 932.894 | 362.408   | 54082.736 | CSR/slice + online topk (vNext)   |
| faiss-cpu | CPU      | faiss-cpu      | faiss       | 95.662   | 118.323 | 2109.043  | 9293.315  | Reference (C++ implementation)    |

### nq Sweep (`nb=262144`, `train_n=20480`, `nlist=512`, `nprobe=32`, `k=20`, `dtype=float32`, `--warmup 1 --repeat 5`)

`nq` によって “kernel launch 支配” の影響が変わるため、`scripts/benchmark_sweep_nq.py` で曲線を確認できます（`search_ms` は median）。

| nq    | torch-ivf ROCm QPS (matrix) | torch-ivf ROCm QPS (csr) | faiss-cpu QPS |
|------:|-----------------------------:|--------------------------:|--------------:|
| 1     | 721.501                      | 232.148                   | 2517.623      |
| 8     | 2054.812                     | 547.420                   | 3556.504      |
| 32    | 2882.208                     | 1253.467                  | 4041.374      |
| 128   | 3122.172                     | 4632.495                  | 5634.373      |
| 512   | 3268.823                     | 17562.334                 | 5832.750      |
| 2048  | 3153.656                     | 31242.468                 | 7809.676      |
| 19600 | 3114.037                     | 48977.764                 | 9110.788      |

```mermaid
xychart-beta
  title "QPS vs nq (ROCm GPU / nb=262144, nlist=512, nprobe=32, k=20)"
  x-axis [1, 8, 32, 128, 512, 2048, 19600]
  y-axis "QPS" 0 --> 70000
  line "torch-ivf matrix" [722, 2055, 2882, 3122, 3269, 3154, 3114]
  line "torch-ivf csr" [232, 547, 1253, 4632, 17562, 31242, 48978]
  line "faiss-cpu" [2518, 3557, 4041, 5634, 5833, 7810, 9111]
```

### max_codes Sweep (`nq=19600`, `k=20`, `--warmup 1 --repeat 5`)

`recall@k` は **同一ライブラリの `max_codes=0`（無制限）** を基準にした自己比較です（torch-ivf と faiss-cpu は学習手順が異なるため、相互比較の recall ではありません）。

| max_codes | torch-ivf ROCm search_ms | torch-ivf ROCm QPS | torch-ivf recall@k | faiss-cpu search_ms | faiss-cpu QPS | faiss-cpu recall@k |
|----------:|--------------------------:|-------------------:|-------------------:|--------------------:|--------------:|-------------------:|
| 0         | 6278.106                  | 3121.961           | 1.000000           | 1993.889            | 9830.035      | 1.000000           |
| 16384     | 3085.621                  | 6352.045           | 0.645903           | 1082.505            | 18106.149     | 0.643005           |
| 32768     | 6208.652                  | 3156.885           | 0.997574           | 1954.347            | 10028.923     | 0.996574           |
| 65536     | 6302.022                  | 3110.113           | 1.000000           | 1937.197            | 10117.710     | 1.000000           |
| 131072    | 6295.926                  | 3113.124           | 1.000000           | 1938.122            | 10112.884     | 1.000000           |

### max_codes Sweep (`search_mode=csr`, `nq=19600`, `k=20`, `--warmup 1 --repeat 5`)

`recall@k` は **同一ライブラリの `max_codes=0`（無制限）** を基準にした自己比較です（torch-ivf と faiss-cpu は学習手順が異なるため、相互比較の recall ではありません）。

| max_codes | torch-ivf ROCm search_ms | torch-ivf ROCm QPS | torch-ivf recall@k | faiss-cpu search_ms | faiss-cpu QPS | faiss-cpu recall@k |
|----------:|--------------------------:|-------------------:|-------------------:|--------------------:|--------------:|-------------------:|
| 0         | 373.488                   | 52478.231          | 1.000000           | 2058.244            | 9522.682      | 1.000000           |
| 16384     | 300.448                   | 65236.023          | 0.631319           | 1097.045            | 17866.178     | 0.643005           |
| 32768     | 400.247                   | 48969.798          | 0.995556           | 2019.375            | 9705.972      | 0.996574           |
| 65536     | 401.749                   | 48786.656          | 1.000000           | 2041.963            | 9598.607      | 1.000000           |
| 131072    | 409.354                   | 47880.295          | 1.000000           | 2029.955            | 9655.386      | 1.000000           |

## License

MIT License (planned).
