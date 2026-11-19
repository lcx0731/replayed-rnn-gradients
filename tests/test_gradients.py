import numpy as np
import pytest

from replayed_rnn import Parameters, checkpoint_gradient, full_gradient
from replayed_rnn.core import NAMES
from replayed_rnn.diagnostics import negative_control
from replayed_rnn.experiment import gradient_vector, synthetic_problem


def independent_loss(values, x, y, h0, masks, l2):
    """Complex-compatible scalar loss, independent of production step/reverse."""
    state = h0.copy()
    loss = 0.0
    for t, row in enumerate(x):
        state = masks[t] * np.tanh(values["wx"].dot(row) + values["wh"].dot(state) + values["b"])
        output = values["wy"].dot(state) + values["c"]
        loss += np.sum((output - y[t]) ** 2) / (2 * len(x))
    return loss + l2 / 2 * sum(np.sum(v ** 2) for v in values.values())


@pytest.mark.parametrize("dropout", [0.0, 0.35])
def test_every_parameter_and_initial_state_against_complex_step(dropout):
    params, x, y, h0 = synthetic_problem(5, 3, 12)
    h0[:] = [0.1, -0.2, 0.05]
    seed, l2, epsilon = 91, 0.03, 1e-25
    masks = []
    for t in range(len(x)):
        rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, t])))
        masks.append((rng.random(3) >= dropout) / (1 - dropout))
    result = full_gradient(params, x, y, h0, dropout=dropout, seed=seed, l2=l2)
    values = {key: value.astype(complex) for key, value in params.as_dict().items()}
    assert result.loss == pytest.approx(independent_loss(values, x, y, h0, masks, l2).real, abs=1e-14)
    for name, value in values.items():
        expected = np.empty(value.shape)
        for index in np.ndindex(value.shape):
            value[index] += 1j * epsilon
            expected[index] = independent_loss(values, x, y, h0.astype(complex), masks, l2).imag / epsilon
            value[index] -= 1j * epsilon
        np.testing.assert_allclose(result.gradients[name], expected, atol=2e-12, rtol=2e-11)
    expected = np.empty(3)
    for j in range(3):
        initial = h0.astype(complex)
        initial[j] += 1j * epsilon
        expected[j] = independent_loss(values, x, y, initial, masks, l2).imag / epsilon
    np.testing.assert_allclose(result.initial_state_gradient, expected, atol=2e-12)


def test_single_step_chain_rule_by_hand():
    p = Parameters([[0.2, -0.1]], [[0.4]], [0.05], [[0.7]], [-0.03])
    result = full_gradient(p, [[0.3, -0.2]], [[0.4]], [0.1])
    activation = np.tanh(0.17)
    error = 0.7 * activation - 0.03 - 0.4
    delta = 0.7 * error * (1 - activation ** 2)
    assert result.loss == pytest.approx(0.5 * error**2)
    np.testing.assert_allclose(result.gradients["wx"], [[delta * 0.3, delta * -0.2]])
    np.testing.assert_allclose(result.gradients["wh"], [[delta * 0.1]])
    np.testing.assert_allclose(result.gradients["wy"], [[error * activation]])
    np.testing.assert_allclose(result.gradients["b"], [delta])
    np.testing.assert_allclose(result.gradients["c"], [error])
    np.testing.assert_allclose(result.initial_state_gradient, [0.4 * delta])


@pytest.mark.parametrize("length,block", [(1, 1), (2, 1), (7, 3), (17, 1), (17, 4), (17, 17), (31, 8)])
@pytest.mark.parametrize("dropout", [0, 0.3])
def test_block_boundaries_replay_and_counted_buffers(length, block, dropout):
    params, x, y, h0 = synthetic_problem(length, 4, 9)
    h0[:] = 0.1
    full = full_gradient(params, x, y, h0, dropout=dropout, seed=100, l2=0.02)
    replay = checkpoint_gradient(params, x, y, h0, block_size=block, dropout=dropout, seed=100, l2=0.02)
    np.testing.assert_allclose(gradient_vector(full), gradient_vector(replay), rtol=1e-12, atol=1e-13)
    assert full.loss == replay.loss
    assert full.transition_evaluations == length
    assert replay.transition_evaluations == 2 * length
    assert full.reverse_steps == replay.reverse_steps == length
    assert full.activation_buffer_bytes == (2 * length + 1) * 4 * 8
    assert replay.activation_buffer_bytes == ((length + block - 1) // block + 2 * block + 1) * 4 * 8


def test_independent_central_difference_direction_and_negative_controls():
    p, x, y, h0 = synthetic_problem(19, 4, 13)
    settings = dict(dropout=0.3, seed=30)
    result = checkpoint_gradient(p, x, y, h0, block_size=6, **settings)
    rng = np.random.default_rng(123)
    directions = {name: rng.normal(size=value.shape) for name, value in p.as_dict().items()}
    epsilon = 1e-6
    plus = Parameters(**{name: value + epsilon * directions[name] for name, value in p.as_dict().items()})
    minus = Parameters(**{name: value - epsilon * directions[name] for name, value in p.as_dict().items()})
    finite = (full_gradient(plus, x, y, h0, **settings).loss - full_gradient(minus, x, y, h0, **settings).loss) / (2 * epsilon)
    exact = sum(np.sum(result.gradients[name] * directions[name]) for name in NAMES)
    assert finite == pytest.approx(exact, abs=1e-9, rel=1e-7)
    for kind in ("fresh_masks", "truncated"):
        bad = negative_control(p, x, y, h0, kind=kind, block_size=6, **settings)
        assert bad.loss == result.loss
        assert np.linalg.norm(gradient_vector(bad) - gradient_vector(result)) > 1e-4
        assert "diagnostic only" in bad.semantics


def test_negative_controls_vanish_when_their_perturbation_is_irrelevant():
    p, x, y, h0 = synthetic_problem(11, 4)
    ref = full_gradient(p, x, y, h0)
    fresh = negative_control(p, x, y, h0, kind="fresh_masks", block_size=3, dropout=0, seed=0)
    np.testing.assert_allclose(gradient_vector(ref), gradient_vector(fresh), atol=1e-13)
    whole = negative_control(p, x, y, h0, kind="truncated", block_size=11, dropout=0, seed=0)
    np.testing.assert_allclose(gradient_vector(ref), gradient_vector(whole), atol=1e-13)


def test_output_channels_are_summed_not_averaged():
    p, x, y, h0 = synthetic_problem(9, 3)
    one = full_gradient(p, x, y, h0)
    two_params = Parameters(p.wx, p.wh, p.b, np.vstack([p.wy, p.wy]), np.repeat(p.c, 2))
    two = checkpoint_gradient(two_params, x, np.repeat(y, 2, axis=1), h0, block_size=4)
    assert two.loss == pytest.approx(2 * one.loss)
    for name in ("wx", "wh", "b"):
        np.testing.assert_allclose(two.gradients[name], 2 * one.gradients[name], atol=1e-13)
    np.testing.assert_allclose(two.initial_state_gradient, 2 * one.initial_state_gradient, atol=1e-13)
