import matplotlib.pyplot as plt
import ipywidgets as widgets
import numpy as np
from IPython.display import display


def show_CT_series(Volume: np.ndarray, vmin=-110, vmax=190):
    fig, ax = plt.subplots(figsize=(8, 8))
    slices = Volume.shape[0]

    # display(fig)

    @widgets.interact(i=(0, slices - 1))
    def update(i=0):
        ax.clear()
        ax.imshow(Volume[i], cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(f'Slice {i}/{slices - 1}')
        fig.canvas.draw_idle()


def show_CT_series_with_roi(Volume: np.ndarray, roi_volume: np.ndarray, vmin=-110, vmax=190):
    fig, ax = plt.subplots(figsize=(8, 8))
    slices = Volume.shape[0]

    # display(fig)

    @widgets.interact(i=(0, slices - 1))
    def update(i=0):
        ax.clear()
        ax.imshow(Volume[i], cmap='gray', vmin=vmin, vmax=vmax)

        mask = roi_volume[i].astype(bool)
        overlay = np.ma.masked_where(~mask, np.ones_like(mask, dtype=float))
        ax.imshow(overlay, cmap="Reds", vmin=0, vmax=1, alpha=0.6, interpolation="nearest")

        ax.set_title(f'Slice {i}/{slices - 1}')
        fig.canvas.draw_idle()
