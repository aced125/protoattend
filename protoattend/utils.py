import yaml
import os.path as osp
from pathlib import Path
import torch.nn as nn
import torch
import torch_optimizer
from functools import singledispatch
from types import SimpleNamespace


def is_capitalized(string: str) -> bool:
    return string[0].isupper()


def get_available_optimizers():
    native_optims = [optim for optim in dir(torch.optim) if is_capitalized(optim)]
    extra_optims = [optim for optim in torch_optimizer.__all__ if is_capitalized(optim)]
    all_optims = native_optims + extra_optims
    all_optims.sort()
    return all_optims


def compute_input_candidate_ratio(input_batch_size, candidate_batch_size):
    return input_batch_size / (input_batch_size + candidate_batch_size)


def load_default_config():
    with open(Path(__file__).parent / "default_configuration.yaml", "r") as file:
        return yaml.load(file, Loader=yaml.FullLoader)


def infer_model_output_dim(model):
    modules = [child for child in model.children() if hasattr(child, "weight")]
    final_module = modules[-1]
    if not isinstance(final_module, nn.Linear):
        raise ValueError(
            f"Could not infer model output dimensions. "
            f"Please provide these dimensions manually by passing "
            f"in `model_output_dim` arg in `ProtoAttend`"
        )
    return final_module.out_features


@singledispatch
def wrap_namespace(ob):
    return ob


@wrap_namespace.register(dict)
def _wrap_dict(ob):
    return SimpleNamespace(**{k: wrap_namespace(v) for k, v in ob.items()})


@wrap_namespace.register(list)
def _wrap_list(ob):
    return [wrap_namespace(v) for v in ob]
