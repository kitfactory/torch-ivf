def test_package_importable():
    import torch_ivf  # noqa: F401

    assert hasattr(torch_ivf, "__all__")
