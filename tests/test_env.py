import importlib


def test_faiss_cpu_available():
    module = importlib.import_module("faiss")
    assert hasattr(module, "IndexFlatL2")


def test_pytest_runs():
    # Dummy assertion to ensure pytest discovers at least one test case.
    assert True
