"""Command-line utility to train RL agents on the lead-lag environment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Optional
import pandas as pd

from models.LeadLag_main import LeadLagConfig
from models.leadlag_rl_env import LeadLagEnv, RewardWeights
from preprocessing_data.preprocessing import (
    preprocess_ffill,
    resample_crypto_data,
    selected_uni,
)

try:  # pragma: no cover - optional heavy dependencies
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import CheckpointCallback
except ImportError as exc:  # pragma: no cover - user feedback
    raise ImportError(
        "The training script requires `stable-baselines3`. Install it via\n"
        "  pip install stable-baselines3"
    ) from exc


def _load_prices(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Price file not found: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    if df.empty:
        raise ValueError("Price file is empty")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError("Price file must have a DatetimeIndex")
    return df.sort_index()


def _load_universe(path: Optional[Path]) -> Optional[pd.DataFrame]:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Universe file not found: {path}")
    df = pd.read_csv(path)
    if "date" not in df or "symbols" not in df:
        raise ValueError("Universe file must contain 'date' and 'symbols' columns")
    return df


def _make_env_builder(
    price_data: pd.DataFrame,
    config: LeadLagConfig,
    lookback_min: int,
    lookback_max: int,
    reward_weights: RewardWeights,
    window_stride: int,
    random_start: bool,
    df_universe: Optional[pd.Series],
    max_episode_steps: Optional[int],
) -> Callable[[], LeadLagEnv]:
    def _builder() -> LeadLagEnv:
        return LeadLagEnv(
            price_data=price_data,
            config=config,
            lookback_range=(lookback_min, lookback_max),
            reward_weights=reward_weights,
            window_stride=window_stride,
            random_start=random_start,
            max_episode_steps=max_episode_steps,
            df_universe=df_universe,
        )

    return _builder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--price-csv",
        type=Path,
        default=Path("raw_data/1H_prices_20250811.csv"),
        help="CSV file containing price history (default: sample dataset)",
    )
    parser.add_argument(
        "--universe-csv",
        type=Path,
        default=Path("raw_data/universe_data.csv"),
        help="CSV file with monthly universe definitions (optional)",
    )
    parser.add_argument(
        "--timeframe",
        type=str,
        default="1D",
        help="Optional Pandas offset alias to resample prices (default: 1D)",
    )
    parser.add_argument(
        "--max-coins",
        type=int,
        default=20,
        help="Maximum number of assets to keep in the rolling universe",
    )
    parser.add_argument(
        "--lookback-min",
        type=int,
        default=5,
        help="Minimum lookback allowed for the agent",
    )
    parser.add_argument(
        "--lookback-max",
        type=int,
        default=120,
        help="Maximum lookback allowed for the agent",
    )
    parser.add_argument(
        "--window-stride",
        type=int,
        default=1,
        help="Number of rows to skip after each environment step",
    )
    parser.add_argument(
        "--random-start",
        action="store_true",
        help="Randomize the reset start index for additional variability",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100_000,
        help="Total PPO timesteps",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Policy optimizer learning rate",
    )
    parser.add_argument(
        "--reward-weights",
        type=float,
        nargs=3,
        metavar=("SIGNAL", "STABILITY", "EXTREMITY"),
        default=(1.0, 0.5, 0.05),
        help="Weights for signal, stability, and extremity components",
    )
    parser.add_argument(
        "--policy",
        type=str,
        default="MlpPolicy",
        help="Policy architecture for PPO (e.g., MlpPolicy, CnnPolicy)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("rl_checkpoints"),
        help="Directory to store periodic checkpoints",
    )
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=25_000,
        help="Frequency (timesteps) to save checkpoints",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Optional JSON file to override LeadLagConfig parameters",
    )
    parser.add_argument(
        "--max-episode-steps",
        type=int,
        default=None,
        help="Optional limit for steps per episode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    prices = _load_prices(args.price_csv)

    universe_df = _load_universe(args.universe_csv)
    if universe_df is not None:
        universe_series = selected_uni(prices, universe_df, maximum_coin=args.max_coins)
    else:
        universe_series = None

    if args.timeframe:
        prices = resample_crypto_data(prices, args.timeframe)

    if universe_series is not None:
        prices, universe_series = preprocess_ffill(prices, universe_series)

    if args.config is not None:
        config_data = json.loads(Path(args.config).read_text())
        leadlag_config = LeadLagConfig.from_dict(config_data)
    else:
        leadlag_config = LeadLagConfig(
            method="ccf_at_lag",
            lookback=args.lookback_max,
            lag=1,
            update_freq=1,
            use_parallel=False,
        )

    reward_weights = RewardWeights(*args.reward_weights)

    env_builder = _make_env_builder(
        price_data=prices,
        config=leadlag_config,
        lookback_min=args.lookback_min,
        lookback_max=args.lookback_max,
        reward_weights=reward_weights,
        window_stride=args.window_stride,
        random_start=args.random_start,
        df_universe=universe_series,
        max_episode_steps=args.max_episode_steps,
    )

    vec_env = DummyVecEnv([env_builder])
    model = PPO(
        policy=args.policy,
        env=vec_env,
        learning_rate=args.learning_rate,
        verbose=1,
        seed=args.seed,
        tensorboard_log="tensorboard_logs",
    )

    checkpoint_dir = args.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = []
    if args.checkpoint_freq > 0:
        callbacks.append(
            CheckpointCallback(
                save_freq=max(1, args.checkpoint_freq // vec_env.num_envs),
                save_path=str(checkpoint_dir),
                name_prefix="leadlag_ppo",
            )
        )

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks or None)

    output_path = checkpoint_dir / "leadlag_ppo_final"
    model.save(str(output_path))
    print(f"Training complete. Final model saved to {output_path}")


if __name__ == "__main__":
    main()
