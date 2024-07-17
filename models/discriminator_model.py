import torch.nn as nn
from models.utils import weights_init


class Discriminator(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.Sequential(
            nn.Conv2d(cfg.num_channels, cfg.fm_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(cfg.fm_d, cfg.fm_d * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(cfg.fm_d * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(cfg.fm_d * 2, cfg.fm_d * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(cfg.fm_d * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(cfg.fm_d * 4, cfg.fm_d * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(cfg.fm_d * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(cfg.fm_d * 8, 1, 4, 1, 0, bias=False),
        )

        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            weights_init(m)

    def forward(self, x):
        return self.layers(x).view(-1)
