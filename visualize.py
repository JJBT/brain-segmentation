from typing import List

import numpy as np
import torch
import matplotlib.pyplot as plt


def fig2data(fig):
    fig.canvas.draw()
    buf = fig.canvas.tostring_rgb()
    ncols, nrows = fig.canvas.get_width_height()
    return np.fromstring(buf, dtype=np.uint8).reshape(nrows, ncols, 3)


def create_image_visual(
        source: np.ndarray,
        target: np.ndarray,
        output: np.ndarray,
        title: str = 'image'
) -> np.ndarray:
    index = source.squeeze().shape[-1] // 2

    source = source.squeeze()[..., index]
    target = target.squeeze()[..., index]
    output = output.squeeze()[..., index]

    fig = plt.figure(figsize=(8, 4))
    ax1 = fig.add_subplot(131)
    ax1.imshow(source, cmap='gray')
    plt.title('input')
    ax2 = fig.add_subplot(132)
    ax2.imshow(output)
    plt.title('prediction')
    ax3 = fig.add_subplot(133)
    ax3.imshow(target)
    plt.title('target')

    plt.xticks([])
    plt.yticks([])

    image = fig2data(fig)
    # plt.savefig('{}.png'.format(title))
    return image
