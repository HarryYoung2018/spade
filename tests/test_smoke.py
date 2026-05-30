import numpy as np


def test_public_api_imports():
    from spade import Dataset, OptimizationResult, SpadeConfig, optimize_spade, train_spade

    assert Dataset is not None
    assert OptimizationResult is not None
    assert SpadeConfig is not None
    assert callable(optimize_spade)
    assert callable(train_spade)


def test_config_construction():
    from spade import SpadeConfig

    cfg = SpadeConfig(diff_epochs=1, support_k=2)
    assert cfg.diff_epochs == 1
    assert cfg.support_k == 2


def test_dataset_from_npz(tmp_path):
    from spade import Dataset

    path = tmp_path / "tiny.npz"
    x = np.array([[0.0, 1.0], [1.0, 3.0], [2.0, 5.0]], dtype=np.float32)
    y = np.array([0.1, 0.5, 0.9], dtype=np.float32)
    np.savez(path, x=x, y=y)

    data = Dataset.from_npz(str(path))

    assert data.x.shape == (3, 2)
    assert data.y.shape == (3, 1)
    assert data.x_norm.shape == (3, 2)
    assert data.y_norm.shape == (3, 1)
    assert data.bounds.shape == (2, 2)
    np.testing.assert_allclose(data.bounds[:, 0], np.array([0.0, 1.0], dtype=np.float32))
    np.testing.assert_allclose(data.bounds[:, 1], np.array([2.0, 5.0], dtype=np.float32))
