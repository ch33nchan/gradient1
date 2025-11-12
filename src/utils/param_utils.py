"""
Utilities for parameter manipulation in PyTorch models.
"""

import torch
import torch.nn as nn
from typing import List


def flatten_params(model: nn.Module) -> torch.Tensor:
    """
    Flatten all parameters of a model into a single vector.

    Args:
        model: PyTorch model

    Returns:
        Flattened parameter vector
    """
    params = []
    for p in model.parameters():
        params.append(p.data.flatten())
    return torch.cat(params)


def unflatten_params(model: nn.Module, flat_params: torch.Tensor) -> None:
    """
    Set model parameters from a flattened vector.

    Args:
        model: PyTorch model
        flat_params: Flattened parameter vector
    """
    idx = 0
    for p in model.parameters():
        numel = p.numel()
        p.data.copy_(flat_params[idx:idx+numel].reshape(p.shape))
        idx += numel


def get_param_count(model: nn.Module) -> int:
    """
    Get total number of parameters in a model.

    Args:
        model: PyTorch model

    Returns:
        Total number of parameters
    """
    return sum(p.numel() for p in model.parameters())


def flatten_gradients(model: nn.Module) -> torch.Tensor:
    """
    Flatten all gradients of a model into a single vector.

    Args:
        model: PyTorch model

    Returns:
        Flattened gradient vector
    """
    grads = []
    for p in model.parameters():
        if p.grad is not None:
            grads.append(p.grad.flatten())
        else:
            grads.append(torch.zeros_like(p.data.flatten()))
    return torch.cat(grads)


def clip_grad_norm(parameters: List[torch.Tensor], max_norm: float) -> float:
    """
    Clip gradient norm of parameters.

    Args:
        parameters: List of parameters
        max_norm: Maximum gradient norm

    Returns:
        Total norm before clipping
    """
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm)
