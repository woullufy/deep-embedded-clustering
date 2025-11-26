from .datasets import (load_mnist,
                       load_fashion_mnist,
                       )
from .plotting import (plot_ae_losses,
                       plot_ae_reconstructions,
                       plot_all_reconstructions,
                       plot_dec_centers,
                       )

from .training_ae import train_autoencoder
from .training_dec import train_dec
from .training_idec import train_idec

__all__ = [
    'load_mnist',
    'load_fashion_mnist',

    'plot_ae_losses',
    'plot_ae_reconstructions',
    'plot_all_reconstructions',
    'plot_dec_centers',

    'train_autoencoder',
    'train_dec',
    'train_idec',
]
