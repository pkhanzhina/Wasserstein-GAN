import torch
import torch.nn as nn
from torch import autograd


class DiscriminatorLoss(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.penalty_lambda = cfg.penalty_lmb
        self.epsilon = torch.FloatTensor(cfg.batch_size, 1, 1, 1).to(device)

        self.device = device

    def forward(self, D_model, fake, disc_fake, real, disc_real):
        loss = disc_fake - disc_real
        self.epsilon.uniform_(0, 1)
        interpolates = self.epsilon * real + (1 - self.epsilon) * fake
        disc_interpolates = D_model(interpolates)
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).to(self.device), create_graph=True)[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean() * self.penalty_lambda
        loss += gradient_penalty
        return loss


class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, fake):
        return -fake

