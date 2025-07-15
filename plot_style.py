"""
Centralized configuration for plot styling and visualization settings.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple

# Figure sizes
FIGURE_SIZES = {
    "default": (10, 6),
    "wide": (12, 6),
    "square": (8, 8),
    "subplots": (15, 6)
}

# Color palettes
COLORS = {
    "P2": "#2ecc71",  # Green
    "R2": "#e67e22",  # Orange
    "P1_CM": "#3498db",  # Blue
    "TRANSFER": "#9b59b6",  # Purple
    "Optimal": "#e74c3c"  # Red
}

# Agent type order for consistent plotting
AGENT_ORDER = ["P2", "R2", "P1_CM", "TRANSFER"]

# Plot configuration
PLOT_CONFIG = {
    "bar_width": 0.8,
    "grid_alpha": 0.3,
    "value_label_offset": 5,
    "legend_bbox": (1.05, 1),
    "legend_loc": "upper left",
    "legend_title": "Agent Type",
    "optimal_line_style": {
        "color": COLORS["Optimal"],
        "linestyle": "--",
        "label": "Optimal"
    }
}

# Heatmap configuration
HEATMAP_CONFIG = {
    "cmap": "YlOrRd",
    "annot": False,
    "fmt": ".1f"
}

def set_plot_style():
    """Set global plot style settings."""
    # Set seaborn style
    sns.set_style("whitegrid")
    
    # Set matplotlib rcParams
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 16,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.1
    })

def create_figure(figsize: Tuple[float, float] = FIGURE_SIZES["default"]) -> plt.Figure:
    """Create a new figure with consistent settings."""
    fig = plt.figure(figsize=figsize)
    return fig

def add_value_labels(ax, offset: int = PLOT_CONFIG["value_label_offset"]):
    """Add value labels on top of bars."""
    for p in ax.patches:
        ax.annotate(
            f'{p.get_height():.2f}',
            (p.get_x() + p.get_width() / 2., p.get_height()),
            ha='center', va='bottom',
            xytext=(0, offset),
            textcoords='offset points'
        )

def add_optimal_line(ax):
    """Add optimal performance line to plot."""
    ax.axhline(
        y=1.0,
        color=PLOT_CONFIG["optimal_line_style"]["color"],
        linestyle=PLOT_CONFIG["optimal_line_style"]["linestyle"],
        label=PLOT_CONFIG["optimal_line_style"]["label"]
    )

def add_legend(ax):
    """Add consistent legend to plot."""
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        title=PLOT_CONFIG["legend_title"],
        bbox_to_anchor=PLOT_CONFIG["legend_bbox"],
        loc=PLOT_CONFIG["legend_loc"]
    )

def save_plot(fig: plt.Figure, filename: str):
    """Save plot with consistent settings."""
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig) 