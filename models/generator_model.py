import torch.nn as nn
from models.utils import weights_init


class Generator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(cfg.z_size, cfg.fm_g * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(cfg.fm_g * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(cfg.fm_g * 8, cfg.fm_g * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(cfg.fm_g * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(cfg.fm_g * 4, cfg.fm_g * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(cfg.fm_g * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(cfg.fm_g * 2, cfg.fm_g, 4, 2, 1, bias=False),
            nn.BatchNorm2d(cfg.fm_g),
            nn.ReLU(True),
            nn.ConvTranspose2d(cfg.fm_g, cfg.num_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            weights_init(m)

    def forward(self, x):
        return self.layers(x)
