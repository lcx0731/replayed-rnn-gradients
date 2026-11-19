import json
import subprocess
import sys

import numpy as np
import pytest

from replayed_rnn import Parameters, checkpoint_gradient, full_gradient, mask_at
from replayed_rnn.experiment import synthetic_problem


def test_masks_use_absolute_time_and_do_not_advance_global_rng():
    mask = mask_at(5, 18, 64, 0.4)
    np.testing.assert_array_equal(mask_at(5, 18, 64, 0.4), mask)
    assert np.any(mask != mask_at(5, 19, 64, 0.4))
    assert set(mask) <= {0, 1 / 0.6}
    np.random.seed(123)
    p, x, y, h0 = synthetic_problem(16, 4)
    checkpoint_gradient(p, x, y, h0, block_size=3, dropout=0.4, seed=12)
    assert np.random.random() == np.random.RandomState(123).random_sample()


def test_parameter_copies_and_caller_inputs_unchanged():
    wx = np.ones((2, 1))
    p = Parameters(wx, np.eye(2), np.zeros(2), np.ones((1, 2)), [0])
    wx[:] = 0
    assert np.all(p.wx == 1) and not p.wx.flags.writeable
    x, y, h0 = np.ones((3, 1)), np.zeros((3, 1)), np.ones(2)
    full_gradient(p, x, y, h0)
    assert np.all(x == 1) and np.all(y == 0) and np.all(h0 == 1)


@pytest.mark.parametrize("options", [{"block_size": 0}, {"block_size": 9}, {"block_size": True},
                                     {"dropout": 1.0}, {"dropout": float("nan")}, {"dropout": True},
                                     {"seed": -1}, {"seed": True}, {"l2": -1}, {"l2": float("inf")}])
def test_invalid_options(options):
    p, x, y, h0 = synthetic_problem(8, 3)
    settings = {"block_size": 4}
    settings.update(options)
    with pytest.raises(ValueError):
        checkpoint_gradient(p, x, y, h0, **settings)


@pytest.mark.parametrize("field,value", [("wx", [[1j]]), ("b", [float("nan")]), ("wy", [[True]]),
                                        ("wh", [[1e7]]), ("c", [0, 0])])
def test_invalid_parameters(field, value):
    values = dict(wx=[[0]], wh=[[0]], b=[0], wy=[[0]], c=[0])
    values[field] = value
    with pytest.raises(ValueError):
        Parameters(**values)


def test_invalid_sequence_shapes_and_values():
    p, x, y, h0 = synthetic_problem(8, 3)
    for bad in ([], np.zeros((4097, 3)), np.zeros((8, 4)), np.full((8, 3), np.inf)):
        with pytest.raises(ValueError):
            full_gradient(p, bad, y, h0)
    with pytest.raises(ValueError):
        full_gradient(p, x, y[:, 0], h0)
    with pytest.raises(ValueError):
        full_gradient(p, x, y, [0])


def test_installed_cli_from_separate_directory(tmp_path):
    result = subprocess.run([sys.executable, "-m", "replayed_rnn.cli", "--length", "23", "--block-size", "5"],
                            cwd=tmp_path, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    assert data["length"] == 23 and data["checkpoint_transition_evaluations"] == 46
    assert data["max_absolute_gradient_difference"] < 1e-12
    invalid = subprocess.run([sys.executable, "-m", "replayed_rnn.cli", "--dropout", "nan"],
                             cwd=tmp_path, capture_output=True, text=True)
    assert invalid.returncode == 2 and "error:" in invalid.stderr


def test_exploding_adjoint_raises_instead_of_returning_nonfinite_gradient():
    p = Parameters([[0]], [[1e6]], [0], [[1]], [0])
    with pytest.raises(FloatingPointError):
        full_gradient(p, np.zeros((100, 1)), np.ones((100, 1)))
