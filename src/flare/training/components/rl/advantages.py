from __future__ import annotations

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    next_value: float = 0.0,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Generalized Advantage Estimation (GAE) and returns using NumPy."""
    n_steps = len(rewards)
    advantages = np.zeros(n_steps, dtype=np.float32)
    last_gae = 0.0

    for step in reversed(range(n_steps)):
        if step == n_steps - 1:
            next_non_terminal = 1.0 - dones[step]
            next_val = next_value
        else:
            next_non_terminal = 1.0 - dones[step]
            next_val = values[step + 1]

        delta = rewards[step] + gamma * next_val * next_non_terminal - values[step]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[step] = last_gae

    returns = advantages + values
    return advantages, returns


def compute_discounted_returns(
    rewards: np.ndarray,
    dones: np.ndarray,
    gamma: float = 0.99,
) -> np.ndarray:
    """Compute cumulative discounted returns using NumPy."""
    n_steps = len(rewards)
    returns = np.zeros(n_steps, dtype=np.float32)
    running_return = 0.0

    for step in reversed(range(n_steps)):
        if dones[step]:
            running_return = 0.0
        running_return = rewards[step] + gamma * running_return
        returns[step] = running_return

    return returns


__all__ = ["compute_discounted_returns", "compute_gae"]
