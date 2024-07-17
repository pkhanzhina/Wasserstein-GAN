import os
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from models.discriminator_model import Discriminator
from models.generator_model import Generator
from loss.wasserstein_gan.losses import DiscriminatorLoss, GeneratorLoss
from datasets.mnist import MNIST
from utils.neptune_logger import NeptuneLogger
from utils.visualization import show_batch
from utils.utils import set_seed


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg

        set_seed(self.cfg.seed)

        self.start_epoch, self.global_step, self.disc_iters = 0, 0, 0
        self.max_epoch = cfg.max_epoch

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.cfg.device == 'gpu' else "cpu")

        self.__get_data()
        self.__get_model()

        self.logger = NeptuneLogger(self.cfg.logger)

    def __get_model(self):
        self.D_model = Discriminator(self.cfg.model).to(self.device)
        self.G_model = Generator(self.cfg.model).to(self.device)

        self.G_opt = torch.optim.Adam(self.G_model.parameters(), lr=self.cfg.lr, betas=self.cfg.betas)
        self.D_opt = torch.optim.Adam(self.D_model.parameters(), lr=self.cfg.lr, betas=self.cfg.betas)

        self.D_criterion = DiscriminatorLoss(self.cfg, self.device)
        self.G_criterion = GeneratorLoss()

        self.sigmoid = nn.Sigmoid()

        print(self.device)
        print('D: number of trainable params:\t', sum(p.numel() for p in self.D_model.parameters() if p.requires_grad))
        print('G: number of trainable params:\t', sum(p.numel() for p in self.G_model.parameters() if p.requires_grad))

    def __get_data(self):
        train_preprocess = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

        self.train_dataset = MNIST(self.cfg.dataset, 'train', train_preprocess)
        self.train_dataloader = torch.utils.data.dataloader.DataLoader(
            self.train_dataset, batch_size=self.cfg.batch_size, shuffle=True, drop_last=True)

        self.fixed_z = torch.randn(self.cfg.eval_batch_size, self.cfg.model.z_size, 1, 1).to(self.device)
        self.z = torch.FloatTensor(self.cfg.batch_size, self.cfg.model.z_size, 1, 1).to(self.device)

    def _dump_model(self, epoch):
        state_dict = {
            'epoch': epoch,
            'discriminator_state_dict': self.D_model.state_dict(),
            'discriminator_opt_state_dict': self.D_opt.state_dict(),
            'generator_state_dict': self.G_model.state_dict(),
            'generator_opt_state_dict': self.G_opt.state_dict(),
            'const_z': self.fixed_z,
            'global_step': self.global_step,
            'disc_iters': self.disc_iters
        }
        if not os.path.exists(self.cfg.checkpoints_dir):
            os.makedirs(self.cfg.checkpoints_dir)
        path_to_save = os.path.join(self.cfg.checkpoints_dir, f'epoch-{epoch}.pt')
        torch.save(state_dict, path_to_save)

    def _load_model(self, epoch):
        path = os.path.join(self.cfg.pretrained_dir, f"epoch-{epoch}.pt")
        start_epoch = 0
        try:
            state_dict = torch.load(path)
            self.D_model.load_state_dict(state_dict['discriminator_state_dict'])
            self.D_opt.load_state_dict(state_dict['discriminator_opt_state_dict'])
            self.G_model.load_state_dict(state_dict['generator_state_dict'])
            self.G_opt.load_state_dict(state_dict['generator_opt_state_dict'])
            self.fixed_z = state_dict['const_z']
            self.global_step = state_dict['global_step']
            self.disc_iters = state_dict['disc_iters']
            start_epoch = state_dict['epoch']
        except Exception as e:
            print(e)
        return start_epoch

    def training_batch(self, batch):

        ones = torch.ones(self.cfg.batch_size)
        zeros = torch.zeros(self.cfg.batch_size)

        data, labels = batch

        for i in range(self.cfg.n_critic):
            self.D_opt.zero_grad()

            real = data.to(self.device)
            realv = Variable(real)
            disc_real = self.D_model(realv)
            prob_real = self.sigmoid(disc_real).detach().cpu()
            disc_real = disc_real.mean(0)
            self.z.normal_(0, 1)
            zv = Variable(self.z)
            fake = self.G_model(zv)
            disc_fake = self.D_model(fake)
            prob_fake = self.sigmoid(disc_fake).detach().cpu()
            disc_fake = disc_fake.mean(0)
            loss_D = self.D_criterion(self.D_model, fake, disc_fake, real, disc_real)
            loss_D.backward()
            self.D_opt.step()

            acc_real = (torch.round(prob_real) == ones).sum() / self.cfg.batch_size
            acc_fake = (torch.round(prob_fake) == zeros).sum() / self.cfg.batch_size

            self.logger.log_metrics(
                ['disc/loss', 'disc/acc_real', 'disc/acc_fake'],
                [loss_D.item(), acc_real.item(), acc_fake.item()],
                self.disc_iters
            )

            self.disc_iters += 1

        self.G_opt.zero_grad()
        self.z.normal_(0, 1)
        fake = self.G_model(self.z)
        disc_fake = self.D_model(fake)
        disk_fake = disc_fake.mean(0)
        loss_G = self.G_criterion(disk_fake)
        loss_G.backward()
        self.G_opt.step()

        torch.cuda.empty_cache()

        return loss_G.item(), loss_D.item(), acc_real.item(), acc_fake.item()

    def fit(self):
        if self.cfg.epoch_to_load is not None:
            self.start_epoch = self._load_model(self.cfg.epoch_to_load)

        self.generation(step=self.global_step)

        self._dump_model(self.start_epoch)
        pbar = tqdm(range(self.start_epoch, self.max_epoch))
        for epoch in pbar:
            self.G_model.train(), self.D_model.train()
            batch = next(iter(self.train_dataloader))
            loss_g, loss_d, acc_r, acc_f = self.training_batch(batch)
            self.logger.log_metrics(
                ['batch/loss_g', 'batch/loss_d', 'batch/acc_real', 'batch/acc_fake'],
                [loss_g, loss_d, acc_r, acc_f],
                self.global_step
            )

            self.global_step += 1
            pbar.set_description(
                '[{}] loss G: {:.4f}, loss D: {:.4f}, acc real: {:.4f}, acc fake: {:.4f}'.format(epoch, loss_g, loss_d, acc_r, acc_f))

            if self.global_step > 0 and self.global_step % self.cfg.eval_step == 0:
                # show_batch(batch[0], title=f"training step - {epoch}")
                self._dump_model(self.global_step)
                self.generation(step=self.global_step)

    @torch.no_grad()
    def generation(self, z=None, step=None):
        if z is None:
            z = self.fixed_z
        generated_imgs = self.G_model(z).detach().cpu()
        fig = show_batch(generated_imgs, title=f"training step - {step}")
        self.logger.log_images([f'fixed_z/{step}'], [fig])


if __name__ == '__main__':
    from configs.wasserstein_gan.train_cfg import cfg
    trainer = Trainer(cfg)
    trainer.fit()
