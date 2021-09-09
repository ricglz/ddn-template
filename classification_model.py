'''Classification model module'''
from functools import cached_property

from torchmetrics import Accuracy, F1, MetricCollection

from timm import create_model
from timm.data import auto_augment_transform as AutoAugment, FastCollateMixup
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
            num_classes=self.num_classes,
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
        return T.Compose([
            T.RandomVerticalFlip(),
            T.RandomHorizontalFlip(),
            T.RandomRotation(5),
        ])

    @cached_property
    def mixup(self):
        hparams = self.hparams
        return FastCollateMixup(
            hparams.mixup_alpha,
            hparams.cutmix_alpha,
            None,
            hparams.mixup_prob,
            hparams.mixup_switch_prob,
            hparams.mixup_mode,
            hparams.mixup_correct_lam,
            hparams.mixup_label_smoothing,
            self.num_classes
        )

    def _process_batch(self, batch, dataset):
        should_perform_mixup = self.hparams.mixup and dataset == 'train'
        return self.mixup(batch) if should_perform_mixup else batch

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

        parser.add_argument('--cutmix-alpha', type=float, default=0)

        parser.add_bool_argument('--mixup')
        parser.add_argument('--mixup-alpha', type=float, default=1)
        parser.add_bool_argument('--mixup-correct-lam')
        parser.add_argument('--mixup-label-smoothing', type=float, default=0.1)
        parser.add_argument(
            '--mixup-mode',
            type=str,
            default='batch',
            choices=['batch', 'pair', 'elem']
        )
        parser.add_argument('--mixup-prob', type=float, default=1)
        parser.add_argument('--mixup-switch-prob', type=float, default=0.5)

        parser.add_argument('--tta', type=int, default=0)
        return parser
