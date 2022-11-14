import contextlib
import copy
import os
import shutil
from functools import wraps
from typing import Dict, List, Tuple, TypeVar

import torch

import ray
from nataili import disable_local_ray_temp
from nataili.util import logger
from nataili.inference.aitemplate.model import Model

T = TypeVar("T")

if not disable_local_ray_temp.active:
    ray_temp_dir = os.path.abspath("./ray")
    shutil.rmtree(ray_temp_dir, ignore_errors=True)
    os.makedirs(ray_temp_dir, exist_ok=True)
    ray.init(_temp_dir=ray_temp_dir)
    logger.init(f"Ray temp dir '{ray_temp_dir}'", status="Prepared")
else:
    logger.init_warn("Ray temp dir'", status="OS Default")


def performance(f: T) -> T:
    @wraps(f)
    def wrapper(*args, **kwargs):
        return torch.cuda.amp.autocast()(torch.no_grad()(f))(*args, **kwargs)

    return wrapper


def extract_tensors(m: torch.nn.Module) -> Tuple[torch.nn.Module, List[Dict]]:
    tensors = []
    for _, module in m.named_modules():
        params = {
            name: torch.clone(param).cpu().detach().numpy() for name, param in module.named_parameters(recurse=False)
        }
        buffers = {name: torch.clone(buf).cpu().detach().numpy() for name, buf in module.named_buffers(recurse=False)}
        tensors.append({"params": params, "buffers": buffers})

    m_copy = copy.deepcopy(m)
    for _, module in m_copy.named_modules():
        for name in [name for name, _ in module.named_parameters(recurse=False)] + [
            name for name, _ in module.named_buffers(recurse=False)
        ]:
            setattr(module, name, None)

    m_copy.train(False)
    return m_copy, tensors


def replace_tensors(m: torch.nn.Module, tensors: List[Dict], device="cuda"):
    modules = [module for _, module in m.named_modules()]
    for module, tensor_dict in zip(modules, tensors):
        # There are separate APIs to set parameters and buffers.
        for name, array in tensor_dict["params"].items():
            module.register_parameter(
                name,
                torch.nn.Parameter(torch.as_tensor(array, device=device), requires_grad=False),
            )
        for name, array in tensor_dict["buffers"].items():
            module.register_buffer(name, torch.as_tensor(array, device=device))


@contextlib.contextmanager
def load_from_plasma(ref, device="cuda"):
    skeleton, weights = ray.get(ref)
    replace_tensors(skeleton, weights, device=device)
    skeleton.eval().to(device, memory_format=torch.channels_last)
    yield skeleton
    torch.cuda.empty_cache()


def push_model_to_plasma(model: torch.nn.Module) -> ray.ObjectRef:
    ref = ray.put(extract_tensors(model))

    return ref

def init_ait_module(
        model_name,
        workdir,
    ):
        mod = Model(os.path.join(workdir, model_name))
        return mod

def push_ait_module(module: Model) -> ray.ObjectRef:
    ref = ray.put(module)

    return ref

def load_ait_module(ref):
    ait_module = ray.get(ref)

    return ait_module
