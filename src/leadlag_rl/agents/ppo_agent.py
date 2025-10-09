"""Utilities for constructing PPO-based agents for the lead-lag environment."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

try:  # pragma: no cover - optional runtime dependency
    from stable_baselines3 import PPO
    from stable_baselines3.common.type_aliases import GymEnv

    SB3_AVAILABLE = True
except ImportError:  # pragma: no cover
    PPO = None  # type: ignore
    GymEnv = Any  # type: ignore
    SB3_AVAILABLE = False

try:  # pragma: no cover - recurrent PPO is optional
    from sb3_contrib import RecurrentPPO

    SB3_RECURRENT_AVAILABLE = True
except ImportError:  # pragma: no cover
    RecurrentPPO = None  # type: ignore
    SB3_RECURRENT_AVAILABLE = False


@dataclass
class PPOAgentConfig:
    """Configuration for constructing PPO models."""

    policy: str = "MlpPolicy"
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_sde: bool = False
    tensorboard_log: Optional[str] = None
    policy_kwargs: Optional[Dict[str, Any]] = None
    use_lstm: bool = False


def create_ppo_model(env: GymEnv, config: PPOAgentConfig):
    """Instantiate a PPO or RecurrentPPO model based on ``config``.

    Parameters
    ----------
    env:
        Gym-compatible environment instance.
    config:
        Configuration describing the desired PPO variant and hyper-parameters.
    """

    if not SB3_AVAILABLE:
        raise ImportError(
            "stable-baselines3 is required to construct PPO agents. Install it via "
            "`pip install stable-baselines3`."
        )

    policy_kwargs = config.policy_kwargs or {}
    algo_kwargs: Dict[str, Any] = dict(
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        use_sde=config.use_sde,
        tensorboard_log=config.tensorboard_log,
        policy_kwargs=policy_kwargs,
    )

    if config.use_lstm:
        if not SB3_RECURRENT_AVAILABLE:
            raise ImportError(
                "sb3-contrib is required for recurrent PPO. Install it via "
                "`pip install sb3-contrib`."
            )
        model = RecurrentPPO(config.policy, env, **algo_kwargs)
    else:
        model = PPO(config.policy, env, **algo_kwargs)

    return model


__all__ = ["PPOAgentConfig", "create_ppo_model"]
