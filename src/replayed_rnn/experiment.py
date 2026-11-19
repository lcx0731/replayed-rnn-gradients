"""Deterministic synthetic problem and a compact, executable gradient audit."""

import numpy as np

from .core import NAMES, Parameters, _integer, checkpoint_gradient, full_gradient
from .diagnostics import negative_control


def synthetic_problem(length=128, hidden=8, seed=17):
    _integer(length, 1, 4096, "length")
    _integer(hidden, 1, 64, "hidden")
    _integer(seed, 0, 2**32 - 1, "seed")
    t = np.arange(length, dtype=float)
    x = np.column_stack((np.sin(t / 11), np.cos(t / 7), 0.5 * np.sin(t / 3)))
    y = 0.4 * x[:, 0]
    y[1:] += 0.3 * x[:-1, 1]
    y[2:] -= 0.2 * x[:-2, 2]
    rng = np.random.Generator(np.random.PCG64(seed))
    params = Parameters(rng.normal(0, 0.2, (hidden, 3)),
                        rng.normal(0, 0.6 / np.sqrt(hidden), (hidden, hidden)),
                        np.zeros(hidden), rng.normal(0, 0.2, (1, hidden)), np.zeros(1))
    return params, x, y[:, None], np.zeros(hidden)


def gradient_vector(result):
    return np.concatenate([result.gradients[name].ravel() for name in NAMES] + [result.initial_state_gradient])


def audit(length=128, hidden=8, block_size=16, dropout=0.25, seed=17):
    params, x, y, h0 = synthetic_problem(length, hidden, seed)
    full = full_gradient(params, x, y, h0, dropout=dropout, seed=seed)
    checkpoint = checkpoint_gradient(params, x, y, h0, block_size=block_size, dropout=dropout, seed=seed)
    ref = gradient_vector(full)
    difference = gradient_vector(checkpoint) - ref
    if not np.allclose(gradient_vector(checkpoint), ref, rtol=1e-11, atol=1e-12):
        raise ArithmeticError("checkpoint gradient failed its full-tape reference check")
    controls = {}
    for kind in ("fresh_masks", "truncated"):
        result = negative_control(params, x, y, h0, kind=kind, block_size=block_size, dropout=dropout, seed=seed)
        controls[kind] = {"reported_forward_loss": result.loss,
                          "relative_gradient_error": float(np.linalg.norm(gradient_vector(result) - ref) / max(np.linalg.norm(ref), 1e-15)),
                          "semantics": result.semantics}
    return {"data": "deterministic synthetic sequence and seeded parameter initialization",
            "length": length, "hidden": hidden, "block_size": block_size, "dropout": dropout, "seed": seed,
            "full_loss": full.loss, "checkpoint_loss": checkpoint.loss,
            "max_absolute_gradient_difference": float(np.max(np.abs(difference))),
            "full_activation_buffer_bytes": full.activation_buffer_bytes,
            "checkpoint_activation_buffer_bytes": checkpoint.activation_buffer_bytes,
            "buffer_ratio_full_over_checkpoint": full.activation_buffer_bytes / checkpoint.activation_buffer_bytes,
            "full_transition_evaluations": full.transition_evaluations,
            "checkpoint_transition_evaluations": checkpoint.transition_evaluations,
            "reverse_steps_each": full.reverse_steps, "negative_controls": controls,
            "memory_scope": "Only retained state/tanh/checkpoint ndarray buffers; excludes inputs, targets, parameters, gradients, working temporaries and allocator overhead. Not process peak memory."}
