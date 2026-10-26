"""Shared figure style of the reports (per-task figures, study benchmark, design figures).

Reference palette (dataviz skill, light surface): the two series slots were validated all-pairs
(validate_palette.js: CVD dE 24.7, normal 33.6, contrast >= 3:1); the AA reference is drawn in ink.
"""
SURFACE, INK, INK2, MUTED, GRID, AXIS = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#c3c2b7'
SERIES = ('#2a78d6', '#eb6834')                  # two-state colours (first, second)
TYPE_COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']


def style(plt):
    plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 8, 'axes.facecolor': SURFACE,
                         'figure.facecolor': SURFACE, 'axes.edgecolor': AXIS, 'axes.labelcolor': INK2,
                         'xtick.color': MUTED, 'ytick.color': INK2, 'axes.titlecolor': INK,
                         'axes.spines.top': False, 'axes.spines.right': False, 'savefig.dpi': 200})
