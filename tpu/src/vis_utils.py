# src/vis_utils.py
from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import numpy as np

# Optional imports for plotting
try:
    import io, base64
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, Normalize
    _MATPLOTLIB_AVAILABLE = True
except Exception:
    plt = ListedColormap = Normalize = None
    _MATPLOTLIB_AVAILABLE = False

# Public alias if you prefer non-underscored name:
MATPLOTLIB_AVAILABLE = _MATPLOTLIB_AVAILABLE

# ARC palette & normalization (matches your original)
_cmap_list = ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
              '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25']
_cmap = ListedColormap(_cmap_list) if _MATPLOTLIB_AVAILABLE else None
_norm = Normalize(vmin=0, vmax=9) if _MATPLOTLIB_AVAILABLE else None

def get_arc_arrays(task_data: Optional[Dict]) -> Tuple[List, List, List, List]:
    """Return (train_inputs, train_outputs, test_inputs, test_outputs) lists of grids."""
    if not task_data:
        return [], [], [], []
    ti = [ex['input'] for ex in task_data.get('train', [])]
    to = [ex['output'] for ex in task_data.get('train', [])]
    si = [ex['input'] for ex in task_data.get('test',  [])]
    so = [ex.get('output') for ex in task_data.get('test', [])]
    so = [g for g in so if g is not None]
    return ti, to, si, so

def _plot_grid(ax, grid, title: str = "", gridlines: bool = True) -> None:
    g = np.asarray(grid, dtype=int)
    ax.imshow(g, cmap=_cmap, norm=_norm, origin='upper')
    ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])
    if gridlines and g.size:
        ax.hlines(np.arange(g.shape[0]) + 0.5, -0.5, g.shape[1]-0.5, linewidth=0.5)
        ax.vlines(np.arange(g.shape[1]) + 0.5, -0.5, g.shape[0]-0.5, linewidth=0.5)
    ax.set_aspect('equal', adjustable='box')

def plot_riddle_two_rows(
    train_input_grids: List, train_output_grids: List,
    test_input_grids: List,  test_output_grids: List,
    gridlines: bool = False, return_base64: bool = False,
    max_train_pairs: Optional[int] = None
):
    """
    Draws a 2-row panel: Train (input/output) pairs, then Test (input/output).
    If return_base64=True, returns a PNG data URI payload (base64 string).
    Otherwise returns the matplotlib Figure and shows nothing.
    """
    if not _MATPLOTLIB_AVAILABLE:
        return None

    ti = [np.asarray(g, int) for g in train_input_grids if np.asarray(g).size]
    to = [np.asarray(g, int) for g in train_output_grids if np.asarray(g).size]
    si = [np.asarray(g, int) for g in test_input_grids  if np.asarray(g).size]
    so = [np.asarray(g, int) for g in test_output_grids if np.asarray(g).size]

    if si and not so:  # keep shape of panel when test outputs are unknown
        so = [np.full_like(si[0], 0)]

    if max_train_pairs is not None:
        ti, to = ti[:max_train_pairs], to[:max_train_pairs]

    n_train, n_test = len(ti), len(si)
    total_cols = n_train + n_test
    if total_cols == 0:
        return None

    fig_w = max(1, total_cols) * 3.5
    fig, axs = plt.subplots(2, total_cols, figsize=(fig_w, 7), squeeze=False)

    # Train columns
    for c in range(n_train):
        _plot_grid(axs[0, c], ti[c], f"Train Input {c+1}", gridlines)
        _plot_grid(axs[1, c], to[c], f"Train Output {c+1}", gridlines)

    # Test columns
    for t in range(n_test):
        col = n_train + t
        _plot_grid(axs[0, col], si[t], f"Test Input {t+1}", gridlines)
        _plot_grid(axs[1, col], so[t], f"Test Output {t+1}", gridlines)

    fig.tight_layout()

    if return_base64:
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return b64
    return fig
