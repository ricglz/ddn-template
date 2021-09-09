"""Hyper-parameters typed namespace"""
from typed_argparse import TypedArgs

class HparamsNamespace(TypedArgs):
    # Model
    drop_rate: float
    model_name: str

    # Training
    epochs: int

    # Classification
    auto_augment: bool
    auto_augment_mstd: float
    auto_augment_policy: str
    cutmix_alpha: float
    mixup: bool
    mixup_alpha: float
    mixup_correct_lam: bool
    mixup_label_smoothing: float
    mixup_mode: str
    mixup_prob: float
    mixup_switch_prob: float
    tta: int

    # Optimizer
    lookahead: bool
    lr: float
    optimizer: str
    weight_decay: float

    # Scheduler
    anneal_strategy: str
    base_momentum: float
    div_factor: float
    final_div_factor: float
    max_momentum: float
    pct_start: float
    three_phase: bool

    # Dataset
    batch_size: int
