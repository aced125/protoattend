import yaml
import os.path as osp
import torch.nn as nn
from functools import singledispatch
from types import SimpleNamespace


def compute_input_candidate_ratio(input_batch_size, candidate_batch_size):
    return input_batch_size / (input_batch_size + candidate_batch_size)


def load_default_config():
    dir_path = osp.dirname(osp.realpath(__file__))
    default_config_file = osp.join(dir_path, "default_configuration.yaml")
    with open(default_config_file, "r") as file:
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
