import logging
from types import SimpleNamespace

import numpy as np

from graspkit.ml_module import ml_initializer


def test_ci_idx_data_processor_uses_shared_ci_squared_helper(
    monkeypatch,
    tmp_path,
) -> None:
    calls: list[np.ndarray] = []

    def fake_ci_squared(coefficients: np.ndarray) -> np.ndarray:
        calls.append(coefficients.copy())
        return np.full(coefficients.shape, 0.25, dtype=np.float64)

    saved: dict[str, np.ndarray] = {}

    def fake_storage(path, idxs, ci_squared) -> None:
        saved["idxs"] = idxs.copy()
        saved["ci_squared"] = ci_squared.copy()

    monkeypatch.setattr(ml_initializer, "ci_squared", fake_ci_squared)
    monkeypatch.setattr(ml_initializer, "csfs_idxs_ci_storage", fake_storage)

    config = SimpleNamespace(
        cal_settings=SimpleNamespace(cal_loop_num=1),
        cal_path=SimpleNamespace(accumulated_idxs_ci_path=tmp_path / "ci.npz"),
    )
    coefficients = np.array([[0.1, -0.2], [0.3, -0.4]], dtype=np.float64)
    indices = np.array([4, 5], dtype=np.int64)

    result = ml_initializer.ci_idx_data_processor(
        coefficients,
        indices,
        config,
        logging.getLogger("test-ci-processor"),
    )

    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], coefficients)
    np.testing.assert_array_equal(result, np.full((2, 2), 0.25))
    np.testing.assert_array_equal(saved["idxs"], indices)
    np.testing.assert_array_equal(saved["ci_squared"], result)
