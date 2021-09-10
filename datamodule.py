"""Contains the datamodule class"""
from multiprocessing import cpu_count
from os import path

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T

from timm.data.auto_augment import augment_and_mix_transform as AugMix, \
                                   auto_augment_transform as AutoAugment

from sampler import BalancedBatchSampler
from utils import CustomParser as ArgumentParser, get_data_dir
from hparams_namespace import HparamsNamespace

def parse_hparams(args: HparamsNamespace):
    config_str = f'augmix-m{args.augmix_magnitude}-w{args.augmix_width}'
    return f'{config_str}-d{args.augmix_depth}-mstd{args.augmix_mstd}-b{args.augmix_blend}'

def get_augmentations(args: HparamsNamespace) -> list:
    if args.auto_augment:
        return AutoAugment(
            args.auto_augment_policy,
            args.auto_augment_mstd,
        )
    if args.augmix:
        config_str = parse_hparams(args)
        return [AugMix(config_str, {})]
    return [
        T.RandomRotation(degrees=5),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
    ]

class DataModule(LightningDataModule):
    """Datamodule to handle and prepare the Flame dataset"""
    data_dir = ''
    train_ds = None
    test_ds = None
    val_ds = None

    def __init__(self, args: HparamsNamespace):
        super().__init__()
        self.batch_size = args.batch_size
        self.use_balanced_sampler = args.balanced_sampler
        resize = T.Resize((224, 224))
        normalize = T.Normalize([0.485, 0.456, 0.406],
                                [0.229, 0.224, 0.225])
        to_tensor = T.ToTensor()
        augmentations = get_augmentations(args)
        self.train_transforms = T.Compose([
            resize,
            *augmentations,
            to_tensor,
            normalize
        ])
        self.transforms = T.Compose([resize, to_tensor, normalize])

    def prepare_data(self):
        self.data_dir = get_data_dir()

    def create_dataset(self, folder_name, transforms):
        return ImageFolder(path.join(self.data_dir, folder_name), transforms)

    def setup(self, stage=None):
        self.train_ds = self.create_dataset('train', self.train_transforms)
        self.val_ds = self.create_dataset('val', self.transforms)
        self.test_ds = self.create_dataset('test', self.transforms)

    def _general_dataloader(self, dataset, **kwargs):
        return DataLoader(
            dataset, batch_size=self.batch_size, num_workers=cpu_count(),
            drop_last=True, pin_memory=True, **kwargs)

    def train_dataloader(self):
        if self.use_balanced_sampler:
            sampler = BalancedBatchSampler(self.train_ds, shuffle=True)
        else:
            sampler = None
        return self._general_dataloader(self.train_ds, sampler=sampler)

    def val_dataloader(self):
        return self._general_dataloader(self.test_ds)

    def test_dataloader(self):
        return self._general_dataloader(self.val_ds), self._general_dataloader(self.test_ds)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        parser.add_bool_argument('--augmix')
        parser.add_bool_argument('--augmix_blend')
        parser.add_argument('--augmix_depth', type=int, default=1)
        parser.add_argument('--augmix_magnitude', type=float, default=3)
        parser.add_argument('--augmix_mstd', type=float, default=0)
        parser.add_argument('--augmix_width', type=int, default=3)

        parser.add_bool_argument('--auto_augment')
        parser.add_argument('--auto_augment_mstd', type=float, default=0.5)
        parser.add_argument(
            '--auto_augment_policy',
            type=str,
            default='v0',
            choices=['original', 'originalr', 'v0', 'v0r']
        )

        parser.add_bool_argument('--balanced_sampler')
        parser.add_argument('--batch_size', type=int, default=32)
        return parser

    @staticmethod
    def from_argparse_args(args: HparamsNamespace):
        return DataModule(args)
