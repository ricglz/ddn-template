"""Module to train and test a model"""
from argparse import ArgumentParser

from pytorch_lightning import seed_everything
import wandb

from classification_model import ClassificationModel
from constant import PROBLEM_TYPE
from datamodule import DataModule
from trainer import Trainer

def get_args(model_class):
    parser = ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--model_name', type=str, default='rexnet_200')
    parser.add_argument('--seed', type=int, default=42)

    parser = DataModule.add_argparse_args(parser)
    parser = model_class.add_argparse_args(parser)
    parser = Trainer.add_argparse_args(parser)

    return parser.parse_args()

def main():
    if PROBLEM_TYPE == 'classification':
        model_class = ClassificationModel
    else:
        raise ValueError('Incorrect problem type')

    args = get_args(model_class)

    wandb.config.update(args)
    seed_everything(args.seed)

    model = model_class(args)
    wandb.watch(model)

    datamodule = DataModule.from_argparse_args(args)
    trainer = Trainer.from_argparse_args(args)

    trainer.train_and_test(model, args.epochs, datamodule)

if __name__ == "__main__":
    main()
