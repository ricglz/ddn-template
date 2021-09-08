"""Hyper-parameters typed namespace"""
from typed_argparse import TypedArgs

class HparamsNamespace(TypedArgs):
    model_name: str
    drop_rate: float
