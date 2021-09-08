'''Classification model module'''
from functools import cached_property

from torchmetrics import Accuracy, F1, MetricCollection

from timm import create_model
from timm.data import auto_augment_transform as AutoAugment, Mixup
from timm.loss import SoftTargetCrossEntropy

from torch.nn import CrossEntropyLoss, ModuleDict
from torch.nn.functional import sigmoid, softmax
import torchvision.transforms as T

from model import Model

class ClassificationModel(Model):
    num_classes = 2

    @cached_property
    def model(self):
        return create_model(
            self.hparams.model_name,
            pretrained=True,
            num_classes=2,
            drop_rate=self.hparams.drop_rate
        )

    @cached_property
    def activation(self):
        return lambda y_val: softmax(y_val, dim=1)

    @cached_property
    def train_criterion(self):
        return SoftTargetCrossEntropy

    @cached_property
    def val_criterion(self):
        return CrossEntropyLoss

    @cached_property
    def metrics(self):
        general_metrics = [
            Accuracy(compute_on_step=False),
            F1(num_classes=2, compute_on_step=False),
        ]
        metric = MetricCollection(general_metrics)
        return ModuleDict({
            'test_metrics': metric.clone(),
            'train_metrics': metric.clone(),
            'val_metrics': metric.clone(),
        })

    @cached_property
    def transforms(self):
        hparams = self.hparams
        if hparams.auto_augment:
            return AutoAugment(
                hparams.auto_augment_policy,
                hparams.auto_augment_mstd,
            )

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = Model.add_argparse_args(parent_parser)
        parser.add_bool_argument('--auto-augment')
        parser.add_argument('--auto-augment-mstd', type=float, default=0.5)
        parser.add_argument(
            '--auto-augment-policy',
            type=str,
            default='v0',
            choices=['original', 'originalr', 'v0', 'v0r']
        )

        parser.add_bool_argument('--mixup')
        parser.add_argument('--mixup-alpha', type=float, default=1)

        parser.add_argument('--tta', type=int, default=0)
        return parser
