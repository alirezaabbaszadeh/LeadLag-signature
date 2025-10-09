"""Utilities for evaluating trained policies on the lead-lag environment."""
from __future__ import annotations

from typing import Callable, Dict, List

import numpy as np

from leadlag_rl.envs import LeadLagEnv


def rollout_policy(
    env: LeadLagEnv,
    policy_fn: Callable[[np.ndarray], np.ndarray],
    episodes: int = 1,
) -> List[Dict[str, float]]:
    """Execute a deterministic rollout of ``policy_fn`` inside ``env``.

    Parameters
    ----------
    env:
        The environment to evaluate in.
    policy_fn:
        Callable that maps observations to actions.
    episodes:
        Number of evaluation episodes.
    """

    results: List[Dict[str, float]] = []
    for _ in range(episodes):
        obs, info = env.reset()
        terminated = False
        truncated = False
        reward_sum = 0.0
        steps = 0
        while not (terminated or truncated):
            action = policy_fn(obs)
            obs, reward, terminated, truncated, step_info = env.step(action)
            reward_sum += reward
            steps += 1
        results.append({"episode_reward": reward_sum, "steps": steps})
    return results


__all__ = ["rollout_policy"]
