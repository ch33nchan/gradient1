"""Agent implementations."""

from .self_gradient_agent import SelfGradientBanditAgent
from .reinforce_agent import REINFORCEAgent

__all__ = ['SelfGradientBanditAgent', 'REINFORCEAgent']
