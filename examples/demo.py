"""Reproducible buffer/work audit and a fixed-mask synthetic training example."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from replayed_rnn import Parameters, checkpoint_gradient, full_gradient
from replayed_rnn.experiment import audit, gradient_vector, synthetic_problem


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    comparison = audit()
    p, x, y, h0 = synthetic_problem(1024, 12)
    full = full_gradient(p, x, y, h0, dropout=0.25, seed=17)
    blocks = [1, 2, 4, 8, 16, 24, 32, 64, 128, 256, 512, 1024]
    memory = []
    for block in blocks:
        result = checkpoint_gradient(p, x, y, h0, block_size=block, dropout=0.25, seed=17)
        delta = float(np.max(np.abs(gradient_vector(result) - gradient_vector(full))))
        assert np.allclose(gradient_vector(result), gradient_vector(full), atol=1e-12, rtol=1e-11)
        memory.append({"block_size": block, "activation_buffer_bytes": result.activation_buffer_bytes,
                       "transition_evaluations": result.transition_evaluations,
                       "max_absolute_gradient_difference": delta})

    initial, tx, ty, th0 = synthetic_problem(96, 8)
    training = {}
    fitted = {}
    for mode in ("full", "checkpoint"):
        current, losses = initial, []
        for step in range(61):
            settings = dict(dropout=0.25, seed=17, l2=0.001)
            result = (full_gradient(current, tx, ty, th0, **settings) if mode == "full" else
                      checkpoint_gradient(current, tx, ty, th0, block_size=11, **settings))
            losses.append(result.loss)
            if step < 60:
                current = Parameters(**{name: value - 0.15 * result.gradients[name] for name, value in current.as_dict().items()})
        assert losses[-1] < losses[0]
        training[mode] = losses
        fitted[mode] = current
    trajectory_difference = float(np.max(np.abs(np.asarray(training["full"]) - training["checkpoint"])))
    parameter_difference = max(float(np.max(np.abs(fitted["full"].as_dict()[name] - value)))
                               for name, value in fitted["checkpoint"].as_dict().items())
    assert trajectory_difference < 1e-12 and parameter_difference < 1e-12
    metrics = {"data": "deterministic synthetic sequence; seeded initialization and per-time dropout",
               "default_audit": comparison, "memory_length": 1024, "memory_hidden": 12,
               "full_activation_buffer_bytes": full.activation_buffer_bytes,
               "full_transition_evaluations": full.transition_evaluations,
               "memory_sweep": memory,
               "training": {"steps": 60, "learning_rate": 0.15, "l2": 0.001,
                            "length": 96, "hidden": 8, "block_size": 11, "dropout": 0.25,
                            "mask_seed_fixed_for_all_updates": 17, "losses": training,
                            "max_loss_trajectory_difference": trajectory_difference,
                            "max_final_parameter_difference": parameter_difference,
                            "scope": "Fixed-mask training objective only; no validation/test set or generalization claim"}}
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.5), layout="constrained")
    ax = axes[0, 0]
    ax.plot(blocks, [row["activation_buffer_bytes"] / 1024 for row in memory], "o-", color="#197b87", label="Checkpoint buffers")
    ax.axhline(full.activation_buffer_bytes / 1024, color="#bd7520", linestyle="--", label="Full tape buffers")
    ax.set_xscale("log", base=2)
    ax.set_ylabel("Counted activation buffers (KiB)")
    ax.set_xlabel("Steps per checkpoint block")
    ax.set_title("T = 1024, H = 12 | Excludes inputs and working memory")
    ax.legend()
    ax.grid(alpha=0.2)
    ax = axes[0, 1]
    ax.bar(["Full tape", "Checkpoint"], [1024, 2048], color="#197b87", label="Transition evaluations")
    ax.bar(["Full tape", "Checkpoint"], [1024, 1024], bottom=[1024, 2048], color="#9ca3af", label="Reverse steps")
    ax.set_ylabel("Operation counts (not FLOPs or seconds)")
    ax.set_title("One extra transition pass; same reverse-step count")
    ax.legend(loc="upper left")
    ax.set_ylim(0, 3800)
    ax = axes[1, 0]
    errors = [comparison["max_absolute_gradient_difference"],
              comparison["negative_controls"]["fresh_masks"]["relative_gradient_error"],
              comparison["negative_controls"]["truncated"]["relative_gradient_error"]]
    # Recompute the relative error for the valid method rather than mixing units.
    ap, ax_input, ay, ah = synthetic_problem(128, 8)
    ref = full_gradient(ap, ax_input, ay, ah, dropout=0.25, seed=17)
    good = checkpoint_gradient(ap, ax_input, ay, ah, block_size=16, dropout=0.25, seed=17)
    errors[0] = float(np.linalg.norm(gradient_vector(good) - gradient_vector(ref)) / np.linalg.norm(gradient_vector(ref)))
    bars = ax.bar(["Correct replay", "Fresh masks\n(negative control)", "Truncated\n(negative control)"], errors,
                  color=["#197b87", "#ba5347", "#bd7520"])
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_ylim(0, max(errors) * 1.25)
    ax.set_ylabel("Relative gradient error (L2)")
    ax.set_title("Same reported forward loss can hide wrong gradients")
    ax = axes[1, 1]
    ax.plot(training["full"], color="#bd7520", linewidth=3, label="Full tape")
    ax.plot(training["checkpoint"], color="#197b87", linestyle="--", label="Correct checkpoint replay")
    ax.set_xlabel("Gradient-descent updates")
    ax.set_ylabel("Fixed-mask training objective")
    ax.set_title("Synthetic training | No held-out performance claim")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.suptitle("Replayed RNN gradients | Memory, work, and derivative checks", fontsize=15)
    fig.savefig(output / "diagnostics.png", dpi=170)
    plt.close(fig)
    print(json.dumps({"default_audit": comparison, "smallest_tested_buffers": min(memory, key=lambda r: r["activation_buffer_bytes"]),
                      "training_initial_loss": training["full"][0], "training_final_loss": training["full"][-1],
                      "max_final_parameter_difference": parameter_difference}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    run(parser.parse_args().output)
