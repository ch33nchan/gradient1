"""
Utility functions for setting random seeds for reproducibility.
"""

import random
import numpy as np
import torch


def set_seed(seed: int) -> None:
    """
    Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Ensure deterministic behavior on CPU
    torch.use_deterministic_algorithms(False)  # Some ops don't support deterministic

    # Set number of threads for CPU
    torch.set_num_threads(4)
