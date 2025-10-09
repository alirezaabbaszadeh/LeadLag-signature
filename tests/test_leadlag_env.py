import numpy as np
import pandas as pd
import pytest

from leadlag_rl.envs import LeadLagEnv
from leadlag_rl.envs import leadlag_env as leadlag_env_module
from models.LeadLag_main import LeadLagConfig


if leadlag_env_module.gym is None:  # pragma: no cover - environment dependency guard
    pytest.skip("gymnasium/gym is required for LeadLagEnv tests", allow_module_level=True)


@pytest.fixture
def price_frame() -> pd.DataFrame:
    dates = pd.date_range("2021-01-01", periods=90, freq="D")
    base = np.linspace(0.0, 1.5, len(dates))
    data = {
        "asset_a": 10.0 + base,
        "asset_b": 10.5 + base * 1.1 + 0.05 * np.sin(np.linspace(0, 3, len(dates))),
        "asset_c": 11.0 + base * 0.8 + 0.03 * np.cos(np.linspace(0, 2, len(dates))),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def analyzer_config() -> LeadLagConfig:
    return LeadLagConfig(
        method="ccf_at_lag",
        lag=1,
        lookback=30,
        update_freq=1,
        use_parallel=False,
        num_cpus=1,
        show_progress=False,
        Scaling_Method="mean-centering",
        quantiles=4,
    )


def test_leadlag_env_reset_and_step(price_frame: pd.DataFrame, analyzer_config: LeadLagConfig) -> None:
    env = LeadLagEnv(
        price_data=price_frame,
        analyzer_config=analyzer_config,
        min_lookback=5,
        max_lookback=15,
        default_lookback=10,
        episode_length=12,
        random_start=False,
        cache_returns=True,
    )
    obs, info = env.reset(seed=123)
    assert obs.shape == env.observation_space.shape
    assert info["lookback"] == 10
    assert env._cached_returns is not None

    next_obs, reward, terminated, truncated, step_info = env.step(0)
    assert next_obs.shape == env.observation_space.shape
    assert step_info["lookback"] == env.min_lookback
    assert not terminated
    assert not truncated

    expected_reward = (
        env.reward_weights.signal_strength * step_info["signal_strength"]
        + env.reward_weights.stability * step_info["stability"]
        - env.reward_weights.regularization * step_info["penalty"]
    )
    assert np.isfinite(reward)
    assert reward == pytest.approx(expected_reward)
    env.close()


def test_leadlag_env_disables_cache_for_universe(price_frame: pd.DataFrame, analyzer_config: LeadLagConfig) -> None:
    universe_dates = price_frame.index[::20]
    universe = pd.Series([["asset_a", "asset_b"]] * len(universe_dates), index=universe_dates)

    env = LeadLagEnv(
        price_data=price_frame,
        analyzer_config=analyzer_config,
        universe=universe,
        min_lookback=5,
        max_lookback=15,
        default_lookback=10,
        episode_length=12,
        random_start=False,
        cache_returns=True,
    )

    assert env._cached_returns is None
    env.close()
