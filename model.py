"""Model module"""
from functools import cached_property
from typing import Callable

from pytorch_lightning import LightningModule
from timm.optim import Lookahead, RAdam

from torch import stack
from torch.nn import Module, ModuleDict
from torch.optim import Adam, RMSprop, SGD
from torch.optim.lr_scheduler import OneCycleLR

from hparams_namespace import HparamsNamespace
from constant import DATASET_SIZE

OPTIMIZERS = {
    'adam': Adam,
    'radam': RAdam,
    'rmsprop': RMSprop,
    'sgd': SGD,
}

class Model(LightningModule):
    '''Classification model'''
    hparams: HparamsNamespace

    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)

        self.base = self.build_model()
        self.activation = self.build_activation()
        self.train_criterion = self.build_train_criterion()
        self.val_criterion = self.build_val_criterion()
        self.metrics = self.build_metrics()
        self.transforms = self.build_transforms()

    # Building functions
    def build_model(self) -> Module:
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
    @cached_property
    def total_steps(self):
        steps_per_epoch = DATASET_SIZE // self.hparams.batch_size
        return steps_per_epoch * self.hparams.epochs

    def general_div_factor(self, div_factor):
        epochs = self.hparams.epochs
        value = div_factor * epochs / 5
        return value if epochs <= 5 else value * epochs ** 2

    @cached_property
    def div_factor(self):
        return self.general_div_factor(self.hparams.div_factor)

    @cached_property
    def final_div_factor(self):
        return self.general_div_factor(self.hparams.final_div_factor)

    def forward(self, x, tta = 0):
        if tta == 0:
            return self.base(x)
        y_hat_stack = stack([self(self.transform(x)) for _ in range(tta)])
        return y_hat_stack.mean(dim=0)

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
            pct_start=self.hparams.pct_start,
            anneal_strategy=self.hparams.anneal_strategy,
            base_momentum=self.hparams.base_momentum,
            max_momentum=self.hparams.max_momentum,
            div_factor=self.div_factor,
            final_div_factor=self.final_div_factor,
            three_phase=self.hparams.three_phase
        )
        return {'scheduler': scheduler, 'interval': 'step'}
