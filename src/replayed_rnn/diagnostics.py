"""Deliberately inconsistent derivatives for negative controls, never training."""

from .core import _evaluate


def negative_control(params, x, y, h0=None, *, kind, block_size, dropout, seed, l2=0.0):
    if kind not in ("fresh_masks", "truncated"):
        raise ValueError("kind must be fresh_masks or truncated")
    return _evaluate(params, x, y, h0, block_size=block_size, dropout=dropout, seed=seed,
                     l2=l2, mode="checkpoint", replay_seed=seed ^ 0xA5A5 if kind == "fresh_masks" else seed,
                     truncate=kind == "truncated")
