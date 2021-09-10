"""Module containing trainer logic"""
from dataclasses import dataclass

from pytorch_lightning.utilities import xla_device
from pytorch_lightning.loggers import WandbLogger
from torch import cuda
import pytorch_lightning as pl

from callbacks import get_checkpoint, Freezer, ProgressBar
from model import Model
from utils import CustomParser as ArgumentParser
from constant import FAST_DEV_RUN

@dataclass
class Trainer():
    model_name: str

    gradient_clip: float = 0.0
    precision: int = 16
    stages: int = 1
    train_bn: bool = False
    unfreeze_per_step: int = 21

    def get_callbacks(self, model_name: str, epochs: int) -> list:
        checkpoint = get_checkpoint(model_name)
        freezer = Freezer(
            epochs,
            self.stages,
            self.unfreeze_per_step,
            self.train_bn
        )
        return [checkpoint, freezer, ProgressBar()]

    @staticmethod
    def get_accelerator() -> object:
        tpu_device_exists = xla_device.XLADeviceUtils().tpu_device_exists()
        has_gpu = cuda.is_available()

        return {'tpu_cores': 1} if tpu_device_exists else \
               {'gpus': cuda.device_count()} if has_gpu else {}

    def create_trainer(self, model_name, max_epochs=1, **kwargs):
        accelerator = self.get_accelerator()
        callbacks = self.get_callbacks(model_name, max_epochs)
        logger = WandbLogger()
        return pl.Trainer(
            max_epochs=max_epochs, deterministic=True, callbacks=callbacks,
            precision=self.precision, stochastic_weight_avg=False, logger=logger,
            gradient_clip_val=self.gradient_clip, **accelerator, **kwargs)

    def _create_trainer(self, max_epochs: int) -> pl.Trainer:
        return self.create_trainer(
                self.model_name, max_epochs, fast_dev_run=FAST_DEV_RUN)

    def _fit_cycle(self, model: Model, epochs: int, datamodule):
        trainer = self._create_trainer(epochs)
        trainer.fit(model, datamodule=datamodule)
        return trainer

    def train_and_test(self, model: Model, epochs: int, datamodule):
        last_trainer = self._fit_cycle(model, epochs, datamodule)
        last_trainer.test(datamodule=datamodule)

    def test(self, model: Model, datamodule):
        trainer = self.create_trainer(model.hparams.model_name)
        trainer.test(model=model, datamodule=datamodule)

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--gradient_clip', type=float, default=0.0)
        parser.add_argument('--precision', type=int, choices=[16, 32], default=16)
        parser.add_argument('--stages', type=int, default=2)
        parser.add_bool_argument('--train_bn')
        parser.add_argument('--unfreeze_per_step', type=int, default=21)
        return parser

    @staticmethod
    def from_argparse_args(args):
        return Trainer(
            args.model_name,
            args.gradient_clip,
            args.precision,
            args.stages,
            args.train_bn,
            args.unfreeze_per_step
        )
