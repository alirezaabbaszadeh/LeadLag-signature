"""Reinforcement learning environment for adaptive lead-lag analysis.

This module defines :class:`LeadLagEnv`, a Gym-compatible environment that
allows an RL agent to adjust the lookback window used by
:class:`models.LeadLag_main.LeadLagAnalyzer`.  The environment exposes a compact
state vector based on the current lead-lag matrix, while the action controls the
length of the historical window employed in the analysis.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency in the execution environment
    import gymnasium as gym
except ImportError:  # pragma: no cover - fallback to classic gym if available
    import gym

from gym import spaces

from models.LeadLag_main import LeadLagAnalyzer, LeadLagConfig


@dataclass
class RewardWeights:
    """Container for reward coefficients.

    Attributes
    ----------
    signal_strength:
        Coefficient applied to the instantaneous strength of the lead-lag
        relationships observed in the current window.
    stability:
        Coefficient applied to the similarity between the current lead-lag
        matrix and the previous one.  This encourages consistent causal
        structure over time.
    regularization:
        Coefficient applied to the penalty associated with extreme lookback
        selections.  The penalty is subtracted from the reward, therefore the
        coefficient should be positive.
    """

    signal_strength: float = 1.0
    stability: float = 0.5
    regularization: float = 0.1


class LeadLagEnv(gym.Env):
    """Gym environment that exposes adaptive lookback control as an RL task."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        price_data: pd.DataFrame,
        analyzer_config: Union[LeadLagConfig, Dict],
        universe: Optional[pd.Series] = None,
        *,
        min_lookback: int = 5,
        max_lookback: int = 252,
        default_lookback: Optional[int] = None,
        discrete_actions: bool = True,
        reward_weights: RewardWeights | Dict[str, float] | None = None,
        episode_length: Optional[int] = None,
        random_start: bool = True,
        include_strength_feature: bool = True,
    ) -> None:
        """Initialise the environment.

        Parameters
        ----------
        price_data:
            Historical price data indexed by ``DatetimeIndex`` with assets in
            the columns.
        analyzer_config:
            Configuration for :class:`LeadLagAnalyzer`.  Either a
            :class:`LeadLagConfig` instance or a dictionary compatible with
            :meth:`LeadLagConfig.from_dict`.
        min_lookback, max_lookback:
            Bounds on the lookback window (inclusive).
        default_lookback:
            Lookback employed when :meth:`reset` is called.  If ``None`` the
            midpoint of ``[min_lookback, max_lookback]`` is used.
        discrete_actions:
            Whether the action space is discrete (each integer corresponds to a
            concrete window length) or continuous.
        reward_weights:
            Coefficients used to aggregate the reward components.  If a
            dictionary is provided it should contain the keys
            ``{"signal_strength", "stability", "regularization"}``.
        episode_length:
            Maximum number of decisions in an episode.  Defaults to the number
            of available windows implied by ``max_lookback``.
        random_start:
            If ``True`` the starting index is sampled uniformly from all valid
            positions.  Otherwise the episode always starts at the earliest
            possible index.
        include_strength_feature:
            Whether to append the instantaneous signal strength to the
            observation vector.
        """

        if price_data.empty:
            raise ValueError("price_data must contain at least one observation")

        if not isinstance(price_data.index, pd.DatetimeIndex):
            raise TypeError("price_data must be indexed by a DatetimeIndex")

        self.price_data = price_data.sort_index()
        self.assets: Tuple[str, ...] = tuple(self.price_data.columns)
        if len(self.assets) < 2:
            raise ValueError("price_data must contain at least two assets")

        self.min_lookback = int(min_lookback)
        self.max_lookback = int(max_lookback)
        if self.min_lookback < 2:
            raise ValueError("min_lookback must be at least 2")
        if self.max_lookback <= self.min_lookback:
            raise ValueError("max_lookback must be greater than min_lookback")

        self.default_lookback = (
            int(default_lookback)
            if default_lookback is not None
            else (self.min_lookback + self.max_lookback) // 2
        )
        self.default_lookback = int(
            np.clip(self.default_lookback, self.min_lookback, self.max_lookback)
        )

        self.universe = universe
        if isinstance(analyzer_config, LeadLagConfig):
            self.analyzer = LeadLagAnalyzer(analyzer_config, df_universe=universe)
        else:
            leadlag_cfg = LeadLagConfig.from_dict(analyzer_config)
            self.analyzer = LeadLagAnalyzer(leadlag_cfg, df_universe=universe)

        if reward_weights is None:
            reward_weights = RewardWeights()
        elif isinstance(reward_weights, dict):
            reward_weights = RewardWeights(**reward_weights)
        self.reward_weights = reward_weights

        self.discrete_actions = discrete_actions
        self.random_start = random_start
        self.include_strength_feature = include_strength_feature

        max_available_steps = len(self.price_data) - self.max_lookback
        if max_available_steps <= 0:
            raise ValueError(
                "price_data must have more rows than the maximum lookback window"
            )
        self.episode_length = (
            int(episode_length)
            if episode_length is not None
            else max_available_steps
        )
        self.episode_length = max(1, min(self.episode_length, max_available_steps))

        # Pre-compute observation space dimensionality.
        extra_features = 4  # max, min, normalised lookback, penalty term
        if include_strength_feature:
            extra_features += 1
        self.state_dim = len(self.assets) + extra_features

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.state_dim,),
            dtype=np.float32,
        )

        if self.discrete_actions:
            self.action_space = spaces.Discrete(self.max_lookback - self.min_lookback + 1)
        else:
            self.action_space = spaces.Box(
                low=np.array([self.min_lookback], dtype=np.float32),
                high=np.array([self.max_lookback], dtype=np.float32),
                dtype=np.float32,
            )

        self._rng = np.random.default_rng()
        self._current_index: int = 0
        self._steps_elapsed: int = 0
        self._prev_matrix: Optional[pd.DataFrame] = None
        self._prev_lookback: int = self.default_lookback

    # ------------------------------------------------------------------
    # Gym API
    # ------------------------------------------------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        options = options or {}
        start_min = self.max_lookback - 1
        start_max = len(self.price_data) - self.episode_length - 1
        if start_max < start_min:
            start_max = start_min
        if self.random_start:
            self._current_index = int(self._rng.integers(start_min, start_max + 1))
        else:
            self._current_index = start_min
        self._steps_elapsed = 0

        initial_lookback = options.get("initial_lookback", self.default_lookback)
        initial_lookback = int(
            np.clip(initial_lookback, self.min_lookback, self.max_lookback)
        )
        self._prev_lookback = initial_lookback

        matrix = self._compute_leadlag_matrix(self._current_index, initial_lookback)
        self._prev_matrix = matrix
        observation = self._build_state(matrix, initial_lookback)
        info = {
            "lookback": initial_lookback,
            "signal_strength": self._signal_strength(matrix),
        }
        return observation, info

    def step(self, action):
        lookback = self._convert_action_to_lookback(action)
        matrix = self._compute_leadlag_matrix(self._current_index, lookback)
        signal = self._signal_strength(matrix)
        stability = self._stability(matrix)
        penalty = self._lookback_penalty(lookback)

        reward = (
            self.reward_weights.signal_strength * signal
            + self.reward_weights.stability * stability
            - self.reward_weights.regularization * penalty
        )

        observation = self._build_state(matrix, lookback, signal_strength=signal, penalty=penalty)
        self._prev_matrix = matrix
        self._prev_lookback = lookback

        self._current_index += 1
        self._steps_elapsed += 1
        terminated = self._current_index >= len(self.price_data)
        truncated = self._steps_elapsed >= self.episode_length

        info = {
            "lookback": lookback,
            "signal_strength": signal,
            "stability": stability,
            "penalty": penalty,
        }
        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # Helper utilities
    # ------------------------------------------------------------------
    def _convert_action_to_lookback(self, action: Union[int, float, np.ndarray]) -> int:
        if self.discrete_actions:
            if isinstance(action, np.ndarray):
                action = int(action.item())
            lookback = self.min_lookback + int(action)
        else:
            value = float(action[0] if isinstance(action, np.ndarray) else action)
            lookback = int(round(value))
        lookback = int(np.clip(lookback, self.min_lookback, self.max_lookback))
        return lookback

    def _compute_leadlag_matrix(self, index: int, lookback: int) -> pd.DataFrame:
        start_idx = index - lookback + 1
        if start_idx < 0:
            start_idx = 0
            lookback = index + 1
        start_ts = self.price_data.index[start_idx]
        end_ts = self.price_data.index[index]

        log_returns = self.analyzer._compute_log_returns_for_window(
            self.price_data, start_ts, end_ts
        )
        if log_returns.empty or log_returns.shape[1] < 2:
            empty_matrix = pd.DataFrame(
                0.0,
                index=self.assets,
                columns=self.assets,
            )
            return empty_matrix

        matrix = self.analyzer._compute_single_lead_lag_matrix(log_returns)
        return matrix.reindex(index=self.assets, columns=self.assets, fill_value=0.0)

    def _build_state(
        self,
        matrix: pd.DataFrame,
        lookback: int,
        *,
        signal_strength: Optional[float] = None,
        penalty: Optional[float] = None,
    ) -> np.ndarray:
        row_strength = matrix.sum(axis=1).reindex(self.assets, fill_value=0.0).values
        if matrix.size:
            finite_values = matrix.values[np.isfinite(matrix.values)]
            if finite_values.size:
                max_val = float(finite_values.max())
                min_val = float(finite_values.min())
            else:
                max_val = min_val = 0.0
        else:
            max_val = min_val = 0.0
        norm_lookback = (lookback - self.min_lookback) / (self.max_lookback - self.min_lookback)
        if signal_strength is None:
            signal_strength = self._signal_strength(matrix)
        if penalty is None:
            penalty = self._lookback_penalty(lookback)

        features = [max_val, min_val, norm_lookback, penalty]
        if self.include_strength_feature:
            features.append(signal_strength)

        state = np.concatenate([row_strength.astype(np.float32), np.array(features, dtype=np.float32)])
        return state.astype(np.float32)

    def _signal_strength(self, matrix: pd.DataFrame) -> float:
        if matrix.empty:
            return 0.0
        row_strength = matrix.sum(axis=1)
        return float((row_strength.max() - row_strength.min()))

    def _stability(self, matrix: pd.DataFrame) -> float:
        if self._prev_matrix is None:
            return 0.0
        prev = self._prev_matrix.values.flatten()
        curr = matrix.values.flatten()
        if np.allclose(prev, prev[0]) or np.allclose(curr, curr[0]):
            return 0.0
        prev_mean = prev.mean()
        curr_mean = curr.mean()
        prev_std = prev.std()
        curr_std = curr.std()
        if math.isclose(prev_std, 0.0) or math.isclose(curr_std, 0.0):
            return 0.0
        corr = float(np.corrcoef(prev - prev_mean, curr - curr_mean)[0, 1])
        return corr

    def _lookback_penalty(self, lookback: int) -> float:
        if self.max_lookback == self.min_lookback:
            return 0.0
        norm = (lookback - self.min_lookback) / (self.max_lookback - self.min_lookback)
        return float((norm - 0.5) ** 2)

    def close(self):  # pragma: no cover - follows Gym API signature
        return None


__all__ = ["LeadLagEnv", "RewardWeights"]
