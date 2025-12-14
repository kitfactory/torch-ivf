"""torch_ivf package initialization."""

try:
    import torch as _torch  # noqa: F401
except Exception as e:  # pragma: no cover
    raise ImportError(
        "torch-ivf requires a working PyTorch installation, but importing 'torch' failed. "
        "Install PyTorch first (choose CPU/CUDA/ROCm/DirectML as appropriate), then install torch-ivf. "
        "If you want a simple CPU setup, you can also install torch-ivf with the 'pytorch' extra: "
        "pip install \"torch-ivf[pytorch]\". "
        f"Original error: {e!r}"
    ) from e

__all__ = [
    "index",
    "nn",
    "utils",
]
