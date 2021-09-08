"""Model module"""
from pytorch_lightning import LightningModule
from timm import create_model

from hparams_namespace import HparamsNamespace

class Model(LightningModule):
    num_classes = 1
    hparams: HparamsNamespace

    base = None

    def __init__(self, hparams):
        super().__init__()

        self.save_hyperparameters(hparams)
        self.build_model()

    def build_model(self):
        self.base = create_model(
                self.hparams.model_name,
                pretrained=True,
                num_classes=self.num_classes,
                drop_rate=self.hparams.drop_rate)
