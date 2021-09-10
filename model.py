"""Model module"""
from typing import Callable

from pytorch_lightning import LightningModule
from timm.optim import Lookahead, RAdam

from torch import stack
from torch.nn import Module, ModuleDict, BCEWithLogitsLoss
from torch.optim import Adam, RMSprop, SGD
from torch.optim.lr_scheduler import OneCycleLR

from constant import DATASET_SIZE
from hparams_namespace import HparamsNamespace
from utils import CustomParser as ArgumentParser

OPTIMIZERS = {
    'adam': Adam,
    'radam': RAdam,
    'rmsprop': RMSprop,
    'sgd': SGD,
}

class Model(LightningModule):
    '''Generic model'''
    hparams: HparamsNamespace

    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)

        self.base = self.build_base()
        self.activation = self.build_activation()
        self.train_criterion = self.build_train_criterion()
        self.val_criterion = self.build_val_criterion()
        self.metrics = self.build_metrics()
        self.transforms = self.build_transforms()

    # Building functions
    def build_base(self) -> Module:
        raise NotImplementedError()

    def build_activation(self) -> Callable:
        raise NotImplementedError()

    def build_train_criterion(self) -> Module:
        raise NotImplementedError()

    def build_val_criterion(self) -> Module:
        raise NotImplementedError()

    def build_metrics(self) -> ModuleDict:
        raise NotImplementedError()

    def build_transforms(self) -> Module:
        raise NotADirectoryError()

    def just_train_classifier(self):
        self.freeze()
        # Freezer.make_trainable(self.base.get_classifier())

    # Properties
    @property
    def total_steps(self):
        steps_per_epoch = DATASET_SIZE // self.hparams.batch_size
        return steps_per_epoch * self.hparams.epochs

    def general_div_factor(self, div_factor):
        epochs = self.hparams.epochs
        value = div_factor * epochs / 5
        return value if epochs <= 5 else value * epochs ** 2

    @property
    def div_factor(self):
        return self.general_div_factor(self.hparams.div_factor)

    @property
    def final_div_factor(self):
        return self.general_div_factor(self.hparams.final_div_factor)

    def forward(self, x):
        return self.base(x)

    def configure_optimizers(self):
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        optimizer = OPTIMIZERS[self.hparams.optimizer](
            parameters,
            self.hparams.lr,
            weight_decay=self.hparams.weight_decay
        )

        if self.hparams.lookahead:
            optimizer = Lookahead(optimizer)

        scheduler_dict = self._build_scheduler_dict(optimizer)
        return [optimizer], [scheduler_dict]

    def _build_scheduler_dict(self, optimizer):
        scheduler = OneCycleLR(
            optimizer,
            self.hparams.lr,
            self.total_steps,
            anneal_strategy=self.hparams.anneal_strategy,
            base_momentum=self.hparams.base_momentum,
            div_factor=self.div_factor,
            final_div_factor=self.final_div_factor,
            max_momentum=self.hparams.max_momentum,
            pct_start=self.hparams.pct_start,
            three_phase=self.hparams.three_phase
        )
        return {'scheduler': scheduler, 'interval': 'step'}

    # Steps
    def _get_dataset_metrics(self, dataset):
        return self.metrics[f'{dataset}_metrics']

    def _update_metrics(self, y_hat, y, dataset):
        proba = self.activation(y_hat)
        self._get_dataset_metrics(dataset).update(proba, y)

    def _process_batch(self, batch, _dataset):
        return batch

    def _process_y_hat(self, x, _dataset):
        return self(x)

    def criterion(self, dataset):
        return self.train_criterion if dataset == 'train' \
                                    else self.val_criterion

    def _on_step(self, batch, dataset):
        x, y = self._process_batch(batch, dataset)
        y_hat = self._process_y_hat(x, dataset)
        criterion = self.criterion(dataset)
        if isinstance(criterion, BCEWithLogitsLoss):
            y = y.float()
        loss = criterion(y_hat, y)
        self._update_metrics(y_hat, batch[1], dataset)
        self.log(f'{dataset}_loss', loss, prog_bar=True)
        return loss

    def _on_end_epochs(self, dataset):
        metrics = self._get_dataset_metrics(dataset)
        metrics_dict = metrics.compute()
        for key, value in metrics_dict.items():
            self.log(f'{dataset}_{key}', value)
        if dataset != 'train':
            score = stack(list(metrics_dict.values())).mean()
            self.log(f'{dataset}_score', score, prog_bar=True)
        metrics.reset()

    def training_step(self, batch, _batch_idx):
        return self._on_step(batch, 'train')

    def training_epoch_end(self, outputs):
        self._on_end_epochs('train')

    def validation_step(self, batch, *extra):
        return self._on_step(batch, 'val')

    def validation_epoch_end(self, outputs):
        self._on_end_epochs('val')

    def test_step(self, batch, _, dataloader_idx: int):
        dataset = 'val' if dataloader_idx == 0 else 'test'
        return self._on_step(batch, dataset)

    def test_epoch_end(self, outputs):
        self._on_end_epochs('val')
        self._on_end_epochs('test')

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--base_momentum', type=float, default=0.825)
        parser.add_argument('--div_factor', type=float, default=25)
        parser.add_argument('--drop_rate', type=float, default=0.4)
        parser.add_argument('--final_div_factor', type=float, default=1e4)
        parser.add_argument('--lr', type=float, required=True)
        parser.add_argument('--max_momentum', type=float, default=0.9)
        parser.add_argument('--pct_start', type=float, default=0.5)
        parser.add_argument('--weight_decay', type=float, default=0)

        parser.add_bool_argument('--three_phase')
        parser.add_bool_argument('--lookahead')

        parser.add_argument(
            '--anneal_strategy',
            type=str,
            default='linear',
            choices=['linear', 'cos']
        )
        parser.add_argument(
            '--optimizer',
            type=str,
            default='sgd',
            choices=list(OPTIMIZERS.keys())
        )
        return parser
