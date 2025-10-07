"""Reinforcement-learning environment for adaptive lookback selection.

This module leverages :class:`LeadLagAnalyzer` to simulate the effect of
changing the lookback window over time.  The agent receives compact observations
derived from the current lead-lag matrix and chooses the next lookback length.
The reward encourages the discovery of windows that produce strong and stable
lead-lag relationships while discouraging extreme lookback values.

The environment is intentionally lightweight so it can be plugged directly into
RL libraries such as Stable-Baselines3.  The implementation mirrors the
analytical workflow already present in :mod:`LeadLag_main` and therefore reuses
the same helper methods for computing log-returns and pairwise statistics.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    import gym
    from gym import spaces
except ImportError as exc:  # pragma: no cover - handled at runtime
    raise ImportError(
        "The `gym` package is required to use the reinforcement-learning "
        "environment. Please install it with `pip install gym`."
    ) from exc

from .LeadLag_main import LeadLagAnalyzer, LeadLagConfig


def _is_new_gym_api() -> bool:
    """Return ``True`` if the installed Gym exposes the post-0.26 API."""

    version_str = getattr(gym, "__version__", "0.0.0")
    try:
        major, minor, *_ = (int(part) for part in version_str.split("."))
    except ValueError:  # pragma: no cover - extremely defensive
        return False

    return (major > 0) or (major == 0 and minor >= 26)


GYM_NEW_API = _is_new_gym_api()


@dataclass
class RewardWeights:
    """Hyper-parameters for the composite reward."""

    signal: float = 1.0
    stability: float = 1.0
    extremity: float = 0.1


class LeadLagEnv(gym.Env):
    """Gym environment that lets an agent tune the lookback dynamically.

    Parameters
    ----------
    price_data:
        Price DataFrame with a ``DatetimeIndex`` and one column per asset.
    config:
        Configuration for the underlying :class:`LeadLagAnalyzer`.  The
        ``lookback`` value inside the config is treated as the default lookback
        and can be left ``None``.
    lookback_range:
        Inclusive bounds ``(min_lookback, max_lookback)`` for the agent's
        actions.
    reward_weights:
        Weighting applied to the reward components (signal strength,
        stability and extremity penalty).
    discrete_actions:
        If ``True`` (default) the action space is discrete with one action per
        integer lookback.  When ``False`` the environment expects continuous
        actions that will be rounded to the nearest integer within the allowed
        range.
    window_stride:
        Number of rows to advance after each action.  Setting it to values
        greater than one effectively down-samples the episode.
    max_episode_steps:
        Optional hard limit for the number of steps per episode.  ``None`` means
        the environment runs until it reaches the end of ``price_data``.
    random_start:
        When ``True`` each ``reset`` will sample a random valid starting index
        instead of always beginning at ``lookback_range[1]``.
    df_universe:
        Optional universe selection Series forwarded to :class:`LeadLagAnalyzer`
        to ensure the same coin filtering logic as the analytical pipeline.
    """

    metadata = {"render.modes": []}

    def __init__(
        self,
        price_data: pd.DataFrame,
        config: Optional[LeadLagConfig | Dict] = None,
        lookback_range: Tuple[int, int] = (5, 252),
        reward_weights: RewardWeights | Tuple[float, float, float] = RewardWeights(),
        *,
        discrete_actions: bool = True,
        window_stride: int = 1,
        max_episode_steps: Optional[int] = None,
        random_start: bool = False,
        df_universe: Optional[pd.Series] = None,
    ) -> None:
        super().__init__()

        if not isinstance(price_data, pd.DataFrame):
            raise TypeError("price_data must be a pandas DataFrame")
        if not isinstance(price_data.index, pd.DatetimeIndex):
            raise TypeError("price_data index must be a DatetimeIndex")
        if price_data.shape[1] < 2:
            raise ValueError("price_data must contain at least two assets")

        self.price_data = price_data.sort_index()
        self.asset_names = list(self.price_data.columns)
        self.n_assets = len(self.asset_names)

        min_lb, max_lb = lookback_range
        if min_lb < 2:
            raise ValueError("lookback_range[0] must be at least 2")
        if max_lb <= min_lb:
            raise ValueError("lookback_range must satisfy max > min")

        max_possible = len(self.price_data) - 1
        if max_lb > max_possible:
            max_lb = max_possible
        if min_lb > max_lb:
            raise ValueError(
                "Not enough data points to honour the requested minimum lookback"
            )

        self.lookback_min = int(min_lb)
        self.lookback_max = int(max_lb)

        if isinstance(reward_weights, tuple):
            reward_weights = RewardWeights(*reward_weights)
        self.reward_weights = reward_weights

        self.window_stride = int(window_stride)
        if self.window_stride < 1:
            raise ValueError("window_stride must be >= 1")

        self.max_episode_steps = max_episode_steps
        self.random_start = random_start

        # Instantiate analyzer while reusing the existing configuration helpers.
        if config is None:
            config = LeadLagConfig(lag=1)
        elif isinstance(config, dict):
            config = LeadLagConfig.from_dict(config)
        elif isinstance(config, LeadLagConfig):
            config = replace(config)
        else:
            raise TypeError("config must be None, dict, or LeadLagConfig")

        if config.lookback is None or config.lookback < self.lookback_min:
            config.lookback = self.lookback_max
        self.default_lookback = int(np.clip(config.lookback, self.lookback_min, self.lookback_max))

        # The analyzer will handle the heavy lifting for lead-lag computations.
        self.analyzer = LeadLagAnalyzer(config=config, df_universe=df_universe)

        # Action space definition.
        self.discrete_actions = discrete_actions
        if discrete_actions:
            self.action_space = spaces.Discrete(self.lookback_max - self.lookback_min + 1)
        else:
            self.action_space = spaces.Box(
                low=np.array([self.lookback_min], dtype=np.float32),
                high=np.array([self.lookback_max], dtype=np.float32),
                dtype=np.float32,
            )

        # Observation: row sums for each asset + [max, min, signal_strength, lookback_norm]
        self.observation_dim = self.n_assets + 4
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32,
        )

        # Internal state trackers.
        self.current_index: Optional[int] = None
        self.current_lookback: Optional[int] = None
        self.prev_matrix: Optional[pd.DataFrame] = None
        self.steps_taken: int = 0

    # ------------------------------------------------------------------
    # Gym API
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        elif not hasattr(self, "np_random"):
            self.np_random, _ = gym.utils.seeding.np_random()

        if options is None:
            options = {}

        start_index = options.get("start_index")
        if start_index is None:
            if self.random_start:
                high = len(self.price_data) - 1
                low = self.lookback_max
                if high <= low:
                    start_index = low
                else:
                    start_index = int(self.np_random.integers(low, high + 1))
            else:
                start_index = self.lookback_max

        start_index = int(start_index)
        if start_index < self.lookback_max:
            start_index = self.lookback_max
        if start_index >= len(self.price_data):
            raise ValueError("start_index is outside the available data range")

        self.current_index = start_index
        self.current_lookback = int(options.get("initial_lookback", self.default_lookback))
        self.current_lookback = self._clip_lookback(self.current_lookback, self.current_index)
        self.steps_taken = 0

        matrix = self._compute_matrix(self.current_index, self.current_lookback)
        self.prev_matrix = matrix
        observation = self._build_observation(matrix, self.current_lookback)

        if GYM_NEW_API:
            return observation, {}
        return observation

    def step(self, action):
        if self.current_index is None:
            raise RuntimeError("Environment must be reset before stepping")

        lookback = self._action_to_lookback(action)
        next_index = self.current_index + self.window_stride

        terminated = False
        truncated = False

        if next_index >= len(self.price_data):
            # No more data available; terminate without advancing state.
            terminated = True
            observation = self._build_observation(self.prev_matrix, self.current_lookback)
            self.current_index = None
            if GYM_NEW_API:
                return observation, 0.0, terminated, truncated, {}
            return observation, 0.0, terminated, {}

        lookback = self._clip_lookback(lookback, next_index)
        matrix = self._compute_matrix(next_index, lookback)

        reward = self._compute_reward(matrix, self.prev_matrix, lookback)

        self.prev_matrix = matrix
        self.current_index = next_index
        self.current_lookback = lookback
        self.steps_taken += 1

        if self.max_episode_steps is not None and self.steps_taken >= self.max_episode_steps:
            truncated = True

        observation = self._build_observation(matrix, lookback)

        if terminated or truncated:
            self.current_index = None

        if GYM_NEW_API:
            return observation, float(reward), terminated, truncated, {}
        done = terminated or truncated
        return observation, float(reward), done, {}

    # ------------------------------------------------------------------
    # Helpers
    def _action_to_lookback(self, action) -> int:
        if self.discrete_actions:
            if not self.action_space.contains(action):
                raise ValueError(f"Action {action} is outside the action space")
            lookback = self.lookback_min + int(action)
        else:
            value = np.asarray(action, dtype=np.float32).reshape(-1)[0]
            lookback = int(np.round(value))
        return int(np.clip(lookback, self.lookback_min, self.lookback_max))

    def _clip_lookback(self, lookback: int, index: int) -> int:
        max_allowed = min(self.lookback_max, index + 1)
        return int(np.clip(lookback, self.lookback_min, max_allowed))

    def _compute_matrix(self, end_index: int, lookback: int) -> pd.DataFrame:
        start_index = end_index - lookback + 1
        if start_index < 0:
            raise ValueError("Lookback exceeds available data")

        window_start = self.price_data.index[start_index]
        window_end = self.price_data.index[end_index]
        window_log_returns = self.analyzer._compute_log_returns_for_window(
            self.price_data, window_start, window_end
        )

        if window_log_returns.empty or window_log_returns.shape[1] < 2:
            return pd.DataFrame(0.0, index=self.asset_names, columns=self.asset_names)

        valid_columns = window_log_returns.columns.tolist()
        window_data = window_log_returns.values
        n_assets = len(valid_columns)

        lead_lag_matrix = np.zeros((n_assets, n_assets), dtype=float)
        if n_assets >= 2:
            for i in range(n_assets - 1):
                for j in range(i + 1, n_assets):
                    pair_data = window_data[:, [i, j]]
                    if np.all(np.isnan(pair_data[:, 0])) or np.all(np.isnan(pair_data[:, 1])):
                        continue
                    value = self.analyzer._compute_lead_lag_measure_optimized(pair_data)
                    if np.isnan(value):
                        continue
                    lead_lag_matrix[i, j] = value
                    lead_lag_matrix[j, i] = -value

        matrix_df = pd.DataFrame(lead_lag_matrix, index=valid_columns, columns=valid_columns)
        matrix_df = matrix_df.reindex(index=self.asset_names, columns=self.asset_names, fill_value=0.0)

        return matrix_df

    def _build_observation(self, matrix: pd.DataFrame, lookback: int) -> np.ndarray:
        row_sums = matrix.sum(axis=1).reindex(self.asset_names).fillna(0.0).to_numpy(dtype=float)
        max_value = float(matrix.values.max()) if not matrix.empty else 0.0
        min_value = float(matrix.values.min()) if not matrix.empty else 0.0

        if matrix.shape[0] > 1:
            upper = matrix.values[np.triu_indices(matrix.shape[0], k=1)]
            if upper.size:
                signal_strength = float(np.mean(np.abs(upper)))
            else:
                signal_strength = 0.0
        else:
            signal_strength = 0.0

        lookback_norm = (lookback - self.lookback_min) / (self.lookback_max - self.lookback_min + 1e-8)

        observation = np.concatenate(
            [row_sums, np.array([max_value, min_value, signal_strength, lookback_norm], dtype=float)]
        )
        return observation.astype(np.float32)

    def _compute_reward(
        self,
        current_matrix: pd.DataFrame,
        previous_matrix: Optional[pd.DataFrame],
        lookback: int,
    ) -> float:
        if current_matrix.empty:
            return 0.0

        # Signal strength component (average absolute pairwise relation).
        n = current_matrix.shape[0]
        upper = current_matrix.values[np.triu_indices(n, k=1)]
        signal_strength = float(np.mean(np.abs(upper))) if upper.size else 0.0

        # Stability component based on correlation of row sums.
        stability = 0.0
        if previous_matrix is not None and not previous_matrix.empty:
            curr_row = current_matrix.sum(axis=1).reindex(self.asset_names).fillna(0.0).to_numpy(dtype=float)
            prev_row = previous_matrix.sum(axis=1).reindex(self.asset_names).fillna(0.0).to_numpy(dtype=float)
            curr_std = np.std(curr_row)
            prev_std = np.std(prev_row)
            if curr_std > 1e-8 and prev_std > 1e-8:
                corr = np.corrcoef(curr_row, prev_row)[0, 1]
                if np.isfinite(corr):
                    stability = float(np.clip(corr, -1.0, 1.0))

        # Extremity penalty keeps lookback away from the bounds unless justified.
        norm = (lookback - self.lookback_min) / (self.lookback_max - self.lookback_min + 1e-8)
        extremity_penalty = float((norm - 0.5) ** 2)

        reward = (
            self.reward_weights.signal * signal_strength
            + self.reward_weights.stability * stability
            - self.reward_weights.extremity * extremity_penalty
        )
        return float(reward)


__all__ = ["LeadLagEnv", "RewardWeights"]

