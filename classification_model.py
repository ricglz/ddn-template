'''Classification model module'''
from torchmetrics import Accuracy, F1, MetricCollection

from timm import create_model
from timm.data import auto_augment_transform as AutoAugment, FastCollateMixup
from timm.loss import SoftTargetCrossEntropy

from torch import stack
from torch.nn import CrossEntropyLoss, ModuleDict
from torch.nn.functional import sigmoid, softmax
import torchvision.transforms as T

from model import Model

class ClassificationModel(Model):
    num_classes = 2

    def __init__(self, hparams):
        super().__init__(hparams)
        self.mixup = self.build_mixup()

    def build_base(self):
        return create_model(
            self.hparams.model_name,
            pretrained=True,
            num_classes=self.num_classes,
            drop_rate=self.hparams.drop_rate
        )

    def build_activation(self):
        return lambda y_val: softmax(y_val, dim=1)

    def build_train_criterion(self):
        return SoftTargetCrossEntropy()

    def build_val_criterion(self):
        return CrossEntropyLoss()

    def build_metrics(self):
        general_metrics = [
            Accuracy(compute_on_step=False),
            F1(num_classes=self.num_classes, compute_on_step=False),
        ]
        metric = MetricCollection(general_metrics)
        return ModuleDict({
            'test_metrics': metric.clone(),
            'train_metrics': metric.clone(),
            'val_metrics': metric.clone(),
        })

    def build_transforms(self):
        hparams = self.hparams
        if hparams.train_auto_augment:
            return AutoAugment(
                hparams.train_auto_augment_policy,
                { 'magnitude_std': hparams.train_auto_augment_mstd },
            )
        return T.Compose([
            T.RandomVerticalFlip(),
            T.RandomHorizontalFlip(),
            T.RandomRotation(5),
        ])

    def build_mixup(self):
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

    def forward(self, x, tta = 0):
        if tta == 0:
            return self.base(x)
        y_hat_stack = stack([self(self.transform(x)) for _ in range(tta)])
        return y_hat_stack.mean(dim=0)

    def _process_batch(self, batch, dataset):
        should_perform_mixup = self.hparams.mixup and dataset == 'train'
        if should_perform_mixup:
            x = [elem.cpu().numpy() for elem in batch[0]]
            y = [elem.cpu().numpy() for elem in batch[1]]
            return self.mixup((x, y))
        return batch

    def _process_y_hat(self, x, dataset):
        tta = self.hparams.tta if dataset == 'test' else 0
        return self(x, tta)

    def _get_loss(self, criterion, y_hat, y):
        if isinstance(criterion, SoftTargetCrossEntropy):
            y = y.unsqueeze(1)
        return criterion(y_hat, y)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = Model.add_argparse_args(parent_parser)
        parser.add_bool_argument('--train_auto_augment')
        parser.add_argument('--train_auto_augment_mstd', type=float, default=0.5)
        parser.add_argument(
            '--train_auto_augment_policy',
            type=str,
            default='v0',
            choices=['original', 'originalr', 'v0', 'v0r']
        )

        parser.add_argument('--cutmix_alpha', type=float, default=0)

        parser.add_bool_argument('--mixup')
        parser.add_argument('--mixup_alpha', type=float, default=1)
        parser.add_bool_argument('--mixup_correct_lam')
        parser.add_argument('--mixup_label_smoothing', type=float, default=0.1)
        parser.add_argument(
            '--mixup_mode',
            type=str,
            default='batch',
            choices=['batch', 'pair', 'elem']
        )
        parser.add_argument('--mixup_prob', type=float, default=1)
        parser.add_argument('--mixup_switch_prob', type=float, default=0.5)

        parser.add_argument('--tta', type=int, default=0)
        return parser
