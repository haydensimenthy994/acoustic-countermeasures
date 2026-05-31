"""
Shared matplotlib style for all thesis figures.

Import this at the top of every figure script:
    from src.viz.style import apply_style, COLORS, save_fig
    apply_style()
"""
import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path


# Colour palette — colour-blind safe, print-friendly
COLORS = {
    "cnn14":       "#1f77b4",   # blue
    "scratch":     "#ff7f0e",   # orange
    "train":       "#2ca02c",   # green
    "val":         "#d62728",   # red
    "fgsm":        "#ff7f0e",   # orange
    "pgd":         "#d62728",   # red
    "eot":         "#9467bd",   # purple
    "jamming":     "#1f77b4",   # blue
    "spoofing":    "#8c564b",   # brown
    "clean":       "#2ca02c",   # green
    "adversarial": "#d62728",   # red
    "perturb":     "#7f7f7f",   # grey
    "drone":       "#d62728",
    "no_drone":    "#1f77b4",
}


def apply_style():
    """Apply consistent thesis-quality styling to all plots."""
    mpl.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "grid.linewidth": 0.5,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    })


def save_fig(fig, name, outdir="outputs/figures", formats=("png", "pdf")):
    """
    Save a figure in multiple formats.
    PNG for quick viewing, PDF for LaTeX/thesis inclusion.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    saved = []
    for fmt in formats:
        path = outdir / f"{name}.{fmt}"
        fig.savefig(path, format=fmt)
        saved.append(str(path))
    return saved
