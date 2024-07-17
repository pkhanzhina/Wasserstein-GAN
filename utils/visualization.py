import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np


def show_batch(batch, title=None, figsize=(8, 8), path_to_save=None):
    fig = plt.figure(figsize=figsize)
    img = np.transpose(vutils.make_grid(batch, padding=2, normalize=True), (1, 2, 0))
    plt.imshow(img)
    plt.axis("off")
    if title is not None:
        plt.title(title)
    if path_to_save is not None:
        plt.savefig(path_to_save)
    plt.show()

    return fig
