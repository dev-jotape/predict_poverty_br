"""
Utilities used by example notebooks
"""
from typing import Any, Optional, Tuple
from PIL import Image

import matplotlib.pyplot as plt
import numpy as np


def plot_image(
    image: np.ndarray, factor: float = 1.0, clip_range: Optional[Tuple[float, float]] = None, **kwargs: Any
) -> None:
    """Utility function for plotting RGB images."""
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    if clip_range is not None:
        img = np.clip(image * factor, *clip_range)
        ax.imshow(img, **kwargs)
    else:
        img = image * factor
        ax.imshow(img, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])

def save_image(
    image: np.ndarray, filename: str, factor: float = 1.0, clip_range: Optional[Tuple[float, float]] = None, **kwargs: Any
) -> None:
    if clip_range is not None:
        img = np.clip(image * factor, *clip_range)
        plt.imsave(filename, img)

    else:
        img = image * factor
        plt.imsave(filename, img)