import os
from datetime import datetime
from easydict import EasyDict
from configs.wasserstein_gan.dataset_cfg import cfg as dataset_cfg
from configs.wasserstein_gan.model_cfg import cfg as model_cfg

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


cfg = EasyDict()

cfg.seed = 0

cfg.lr = 0.0001
cfg.max_epoch = 5000
cfg.device = 'gpu'
cfg.betas = [0.0, 0.9]
cfg.n_critic = 5
cfg.penalty_lmb = 10
cfg.batch_size = 8

cfg.eval_batch_size = 8

cfg.eval_step = 100

cfg.epoch_to_load = None

cfg.dataset = dataset_cfg
cfg.model = model_cfg

# checkpoint
cfg.checkpoints_dir = os.path.join(ROOT_DIR, 'data/GAN', datetime.now().strftime('%Y-%m-%d_%H-%M'))
cfg.pretrained_dir = cfg.checkpoints_dir


neptune_cfg = EasyDict()
neptune_cfg.project_name = ''
neptune_cfg.api_token = ''
neptune_cfg.run_name = None

cfg.logger = neptune_cfg
