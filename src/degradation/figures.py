"""Figures for the report, built from the CSV files in ``results/``.

Colours follow a colour-blind-checked categorical order; each model keeps the
same colour in every figure, and every figure also labels series directly or
with a legend, so identity never depends on colour alone.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#: One fixed colour per model, used in every figure.
COLOURS = {"linear": "#2a78d6", "tree": "#eb6834", "svr": "#1baf7a",
           "mean_baseline": "#8a8985"}
#: Display names for models.
LABELS = {"linear": "Linear regression", "tree": "Decision tree",
          "svr": "SVR", "mean_baseline": "Mean baseline"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e", "axes.grid": True,
    "grid.color": "#e6e5e1", "grid.linewidth": 0.6, "lines.linewidth": 2,
    "savefig.dpi": 200, "savefig.bbox": "tight",
})


def plot_svr_tuning(svr_table, path=None):
    """Validation MAE against epsilon for each C (gamma = 'scale'), log-log axes."""
    t = svr_table[svr_table["gamma"].astype(str) == "scale"]
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    shades = ["#bcd3f0", "#7fa9e2", "#2a78d6", "#174a8a"]
    for shade, (C, g) in zip(shades, t.sort_values("C").groupby("C")):
        g = g.sort_values("epsilon")
        ax.plot(g["epsilon"], g["mae_mean"], marker="o", markersize=4, color=shade,
                label=f"C = {C:g}")
    ax.axvline(0.001, color="#8a8985", linestyle="--", linewidth=1)
    ax.text(0.00105, ax.get_ylim()[1] * 0.6, "decay step\n(0.001)", fontsize=8, color="#52514e")
    ax.set(xscale="log", yscale="log", xlabel="SVR epsilon", ylabel="Validation MAE",
           title="SVR: accuracy improves once epsilon < decay step")
    ax.legend(frameon=False, title="gamma = scale", fontsize=8, title_fontsize=8,
              loc="lower right")
    return _save(fig, path)


def plot_tree_tuning(tree_table, path=None):
    """Validation MAE against maximum depth (min_samples_leaf = 1)."""
    t = tree_table[tree_table["min_samples_leaf"] == 1].copy()
    t["depth"] = t["max_depth"].fillna(30)
    t = t.sort_values("depth")
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    ax.plot(t["depth"], t["mae_mean"], marker="o", markersize=4, color=COLOURS["tree"])
    ax.set_xticks(list(t["depth"]))
    ax.set_xticklabels([("None" if d == 30 else str(int(d))) for d in t["depth"]])
    ax.set(yscale="log", xlabel="Maximum tree depth", ylabel="Validation MAE",
           title="Decision tree: deeper is better (noise-free data)")
    return _save(fig, path)


def plot_sensor_selection(tree_sel, svr_sel, path=None):
    """Two panels: validation MAE and SVR fit time against number of sensors."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.2))
    for name, t in [("tree", tree_sel), ("svr", svr_sel)]:
        a1.plot(t["n_sensors"], t["mae_mean"], marker="o", markersize=4,
                color=COLOURS[name], label=LABELS[name])
    a1.set(yscale="log", xlabel="Number of sensors (in ranked order)",
           ylabel="Validation MAE", title="Accuracy vs sensors used")
    a1.legend(frameon=False)
    a1.set_xticks(range(1, len(tree_sel) + 1))
    a2.bar(svr_sel["n_sensors"], svr_sel["fit_s"], color=COLOURS["svr"], width=0.7)
    a2.set(xlabel="Number of sensors (in ranked order)", ylabel="Fit time (s)",
           title="SVR training time vs sensors used")
    a2.set_xticks(range(1, len(svr_sel) + 1))
    a2.grid(axis="x", visible=False)
    fig.tight_layout()
    return _save(fig, path)


def plot_predicted_vs_true(predictions, path=None):
    """One panel per model: predicted against true decay on the test set."""
    names = ["linear", "tree", "svr"]
    fig, axes = plt.subplots(1, 3, figsize=(8.4, 3.0), sharex=True, sharey=True)
    lo, hi = 0.945, 1.005
    for ax, name in zip(axes, names):
        ax.plot([lo, hi], [lo, hi], color="#8a8985", linewidth=1, linestyle="--")
        ax.scatter(predictions["true"], predictions[name], s=6, alpha=0.35,
                   color=COLOURS[name], edgecolors="none")
        ax.set(xlim=(lo, hi), ylim=(lo, hi), title=LABELS[name], xlabel="True decay")
        ax.set_aspect("equal")
    axes[0].set_ylabel("Predicted decay")
    fig.tight_layout()
    return _save(fig, path)


def plot_extrapolation(extrap, path=None):
    """Grouped bars: MAE on each held-out decay band, per model (log scale)."""
    names = ["linear", "tree", "svr"]
    bands = sorted(extrap["band"].unique())
    width = 0.26
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for i, name in enumerate(names):
        t = extrap[extrap["model"] == name].sort_values("band")
        ax.bar(np.array(bands) + (i - 1) * (width + 0.02), t["mae"], width=width,
               color=COLOURS[name], label=LABELS[name])
    labels = [f"{r.decay_low:.2f}-{r.decay_high:.2f}" + ("\n(edge)" if r.edge_band else "")
              for r in extrap[extrap["model"] == "svr"].sort_values("band").itertuples()]
    ax.set_xticks(bands)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set(yscale="log", xlabel="Held-out compressor decay band",
           ylabel="MAE on held-out band", title="Predicting degradation states outside the training data")
    ax.grid(axis="x", visible=False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.28))
    return _save(fig, path)


def _save(fig, path):
    if path is not None:
        fig.savefig(path)
    return fig
