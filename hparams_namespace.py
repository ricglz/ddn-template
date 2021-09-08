"""Hyper-parameters typed namespace"""
from typed_argparse import TypedArgs

class HparamsNamespace(TypedArgs):
    # Model
    drop_rate: float
    model_name: str

    # Training
    epochs: int
    tta: int

    # Optimizer
    div_factor: float
    final_div_factor: float
    lookahead: bool
    lr: float
    optimizer: str
    weight_decay: float

    # Dataset
    batch_size: int
