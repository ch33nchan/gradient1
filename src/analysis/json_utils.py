"""JSON serialization utilities for handling numpy and torch types.

This module provides helpers for converting numpy/torch types to JSON-serializable
Python types, which is needed when saving metrics and results.
"""

import numpy as np
import torch
from typing import Any


def convert_to_json_serializable(obj: Any) -> Any:
    """Convert numpy/torch types to JSON-serializable Python types.

    Args:
        obj: Object to convert (can be scalar, dict, list, etc.)

    Returns:
        JSON-serializable version of obj

    Examples:
        >>> convert_to_json_serializable(np.float32(1.5))
        1.5
        >>> convert_to_json_serializable({'a': np.int64(42)})
        {'a': 42}
        >>> convert_to_json_serializable(torch.tensor([1.0, 2.0]))
        [1.0, 2.0]
    """
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        if obj.numel() == 1:
            return obj.item()
        else:
            return obj.tolist()
    else:
        # Plain Python types: return as-is
        return obj
