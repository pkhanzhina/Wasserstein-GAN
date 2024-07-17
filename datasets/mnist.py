import torch.utils.data as data
import torchvision


class MNIST(data.Dataset):
    def __init__(self, cfg, data_type, transforms=None):
        self.cfg = cfg
        self.transforms = transforms

        is_train = data_type == 'train'
        data = torchvision.datasets.MNIST(cfg.root, is_train, download=True)
        images = data.data
        labels = data.targets

        if cfg.number is not None:
            self.images = images[labels == cfg.number]
            self.labels = labels[labels == cfg.number]
        else:
            self.images = images
            self.labels = labels

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transforms is not None:
            img = self.transforms(img)
        return img, label

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    from easydict import EasyDict

    cfg = EasyDict()
    cfg.root = r'C:\Users\khanz\PycharmProjects\acv_khanzhinapv\data'
    cfg.number = 8

    dataset = MNIST(cfg, 'train')
    print()
