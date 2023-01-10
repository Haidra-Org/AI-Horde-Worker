from functools import wraps
from typing import TypeVar

import torch

T = TypeVar("T")


def performance(f: T) -> T:
    @wraps(f)
    def wrapper(*args, **kwargs):
        return torch.cuda.amp.autocast()(torch.no_grad()(f))(*args, **kwargs)

    return wrapper
