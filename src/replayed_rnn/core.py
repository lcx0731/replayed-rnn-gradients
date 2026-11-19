"""Manual float64 reverse-mode differentiation of a small stochastic RNN.

The dropout mask is a pure function of (seed, absolute step, hidden width).
Exactness refers to the same finite computation and fixed mask realization,
subject to floating-point arithmetic, not to an expectation over dropout.
"""

from dataclasses import dataclass
import math

import numpy as np

NAMES = ("wx", "wh", "b", "wy", "c")


def _array(value):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise ValueError("arrays must contain real numeric values")
    array = np.array(raw, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(array)) or np.any(np.abs(array) > 1e6):
        raise ValueError("array entries must be finite and at most 1e6 in magnitude")
    return array


def _integer(value, low, high, name):
    if type(value) is not int or not low <= value <= high:
        raise ValueError(f"{name} must be an integer in {low}..{high}")


def _number(value, low, high, name):
    if type(value) not in (int, float) or not low <= value <= high or not math.isfinite(value):
        raise ValueError(f"{name} must be finite and in {low}..{high}")


@dataclass(frozen=True, init=False)
class Parameters:
    wx: np.ndarray
    wh: np.ndarray
    b: np.ndarray
    wy: np.ndarray
    c: np.ndarray

    def __init__(self, wx, wh, b, wy, c):
        values = dict(zip(NAMES, map(_array, (wx, wh, b, wy, c))))
        wx, wh, b, wy, c = (values[name] for name in NAMES)
        if wx.ndim != 2 or wy.ndim != 2:
            raise ValueError("wx and wy must be matrices")
        h, d = wx.shape
        o = wy.shape[0]
        if not 1 <= h <= 64 or not 1 <= d <= 32 or not 1 <= o <= 16:
            raise ValueError("dimension limits: hidden 1..64, input 1..32, output 1..16")
        if wh.shape != (h, h) or b.shape != (h,) or wy.shape != (o, h) or c.shape != (o,):
            raise ValueError("incompatible parameter shapes")
        for name, value in values.items():
            value.flags.writeable = False
            object.__setattr__(self, name, value)

    def as_dict(self):
        return {name: getattr(self, name) for name in NAMES}


@dataclass(frozen=True)
class GradientResult:
    loss: float
    gradients: dict
    initial_state_gradient: np.ndarray
    activation_buffer_bytes: int
    transition_evaluations: int
    reverse_steps: int
    semantics: str


def mask_at(seed, step, hidden, dropout):
    """Reproducible inverted dropout using an explicit PCG64 per time index."""
    _integer(seed, 0, 2**32 - 1, "seed")
    _integer(step, 0, 4095, "step")
    _integer(hidden, 1, 64, "hidden")
    _number(dropout, 0, 0.8, "dropout")
    if dropout == 0:
        return np.ones(hidden)
    rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, step])))
    return (rng.random(hidden) >= dropout).astype(np.float64) / (1 - dropout)


def _inputs(params, x, y, h0, dropout, seed, l2):
    if not isinstance(params, Parameters):
        raise ValueError("params must be Parameters")
    x, y = _array(x), _array(y)
    h = params.b.size
    h0 = np.zeros(h) if h0 is None else _array(h0)
    if x.ndim != 2 or not 1 <= len(x) <= 4096 or x.shape[1] != params.wx.shape[1]:
        raise ValueError("x must have shape (T,input), with T in 1..4096")
    if y.shape != (len(x), params.c.size) or h0.shape != (h,):
        raise ValueError("incompatible target or initial-state shape")
    _number(dropout, 0, 0.8, "dropout")
    _number(l2, 0, 100, "l2")
    _integer(seed, 0, 2**32 - 1, "seed")
    return x, y, h0


def _evaluate(params, x, y, h0, *, block_size, dropout, seed, l2, mode, replay_seed, truncate=False):
    x, y, h0 = _inputs(params, x, y, h0, dropout, seed, l2)
    t_count, hidden = len(x), params.b.size
    _integer(block_size, 1, t_count, "block_size")
    _integer(replay_seed, 0, 2**32 - 1, "replay_seed")
    gradients = {name: np.zeros_like(value) for name, value in params.as_dict().items()}
    transitions, reverse_steps = 0, 0

    def step(previous, index, rng_seed):
        nonlocal transitions
        transitions += 1
        activation = np.tanh(params.wh @ previous + params.wx @ x[index] + params.b)
        state = mask_at(rng_seed, index, hidden, dropout) * activation
        return state, activation

    loss = 0.0
    carry = np.zeros(hidden)

    def reverse(states, activations, start, length, rng_seed):
        nonlocal carry, reverse_steps
        for j in range(length - 1, -1, -1):
            index = start + j
            reverse_steps += 1
            dy = (params.wy @ states[j + 1] + params.c - y[index]) / t_count
            gradients["wy"] += np.outer(dy, states[j + 1])
            gradients["c"] += dy
            delta = (params.wy.T @ dy + carry) * mask_at(rng_seed, index, hidden, dropout) * (1 - activations[j] ** 2)
            gradients["wx"] += np.outer(delta, x[index])
            gradients["wh"] += np.outer(delta, states[j])
            gradients["b"] += delta
            carry = params.wh.T @ delta

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        if mode == "full":
            states = np.empty((t_count + 1, hidden))
            activations = np.empty((t_count, hidden))
            states[0] = h0
            for t in range(t_count):
                states[t + 1], activations[t] = step(states[t], t, seed)
                residual = params.wy @ states[t + 1] + params.c - y[t]
                loss += 0.5 * float(residual @ residual) / t_count
            buffer_bytes = states.nbytes + activations.nbytes
            reverse(states, activations, 0, t_count, seed)
        elif mode == "checkpoint":
            starts = tuple(range(0, t_count, block_size))
            checkpoints = np.empty((len(starts), hidden))
            current = h0.copy()
            for t in range(t_count):
                if t % block_size == 0:
                    checkpoints[t // block_size] = current
                current, _ = step(current, t, seed)
                residual = params.wy @ current + params.c - y[t]
                loss += 0.5 * float(residual @ residual) / t_count
            # Allocate once, reuse across blocks; these are the counted buffers.
            states = np.empty((block_size + 1, hidden))
            activations = np.empty((block_size, hidden))
            buffer_bytes = checkpoints.nbytes + states.nbytes + activations.nbytes
            for block in range(len(starts) - 1, -1, -1):
                start = starts[block]
                length = min(block_size, t_count - start)
                states[0] = checkpoints[block]
                for j in range(length):
                    states[j + 1], activations[j] = step(states[j], start + j, replay_seed)
                reverse(states, activations, start, length, replay_seed)
                if truncate and block > 0:
                    carry = np.zeros(hidden)
        else:
            raise ValueError("unknown gradient mode")
        for name, value in params.as_dict().items():
            loss += 0.5 * l2 * float(np.sum(value * value))
            gradients[name] += l2 * value
    if not math.isfinite(loss) or not all(np.all(np.isfinite(g)) for g in [*gradients.values(), carry]):
        raise FloatingPointError("loss or gradient overflowed; rescale the problem")
    semantics = "fixed-mask full-objective gradient"
    if replay_seed != seed and dropout:
        semantics = "diagnostic only: inconsistent dropout replay"
    if truncate:
        semantics = "diagnostic only: cross-block state derivatives discarded"
    return GradientResult(loss, gradients, carry, buffer_bytes, transitions, reverse_steps, semantics)


def full_gradient(params, x, y, h0=None, *, dropout=0.0, seed=0, l2=0.0):
    return _evaluate(params, x, y, h0, block_size=1, dropout=dropout, seed=seed, l2=l2,
                     mode="full", replay_seed=seed)


def checkpoint_gradient(params, x, y, h0=None, *, block_size=16, dropout=0.0, seed=0, l2=0.0):
    return _evaluate(params, x, y, h0, block_size=block_size, dropout=dropout, seed=seed,
                     l2=l2, mode="checkpoint", replay_seed=seed)
