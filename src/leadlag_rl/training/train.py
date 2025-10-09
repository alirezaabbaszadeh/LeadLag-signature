"""Hydra-powered training entry-point for adaptive lead-lag RL."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import hydra
import pandas as pd
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from leadlag_rl.agents import PPOAgentConfig, create_ppo_model
from leadlag_rl.envs import LeadLagEnv, RewardWeights
from models.LeadLag_main import LeadLagConfig


def _load_price_data(path: str) -> pd.DataFrame:
    absolute_path = Path(to_absolute_path(path))
    if not absolute_path.exists():
        raise FileNotFoundError(f"Price data not found at {absolute_path}")
    if absolute_path.suffix in {".csv", ".txt"}:
        df = pd.read_csv(absolute_path, index_col=0, parse_dates=True)
    else:
        df = pd.read_parquet(absolute_path)
    return df.sort_index()


def _load_universe(path: Optional[str]) -> Optional[pd.Series]:
    if path is None:
        return None
    absolute_path = Path(to_absolute_path(path))
    if not absolute_path.exists():
        raise FileNotFoundError(f"Universe file not found at {absolute_path}")
    df = pd.read_csv(absolute_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    if df.shape[1] == 1:
        return df.iloc[:, 0]
    return df.apply(lambda row: row.dropna().tolist(), axis=1)


def _build_analyzer_config(cfg: DictConfig) -> LeadLagConfig:
    leadlag_dict = OmegaConf.to_container(cfg, resolve=True)
    return LeadLagConfig.from_dict(dict(leadlag_dict))


@hydra.main(config_path="../../../configs", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    price_df = _load_price_data(cfg.data.price_path)
    df_universe = _load_universe(cfg.data.get("universe_path"))

    analyzer_cfg = _build_analyzer_config(cfg.env.leadlag)
    env = LeadLagEnv(
        price_data=price_df,
        analyzer_config=analyzer_cfg,
        universe=df_universe,
        min_lookback=cfg.env.min_lookback,
        max_lookback=cfg.env.max_lookback,
        default_lookback=cfg.env.default_lookback,
        discrete_actions=cfg.env.discrete_actions,
        reward_weights=RewardWeights(**cfg.env.reward_weights),
        episode_length=cfg.env.episode_length,
        random_start=cfg.env.random_start,
        include_strength_feature=cfg.env.include_strength_feature,
    )

    if cfg.training.dry_run:
        return

    agent_config = PPOAgentConfig(**OmegaConf.to_container(cfg.agent, resolve=True))
    model = create_ppo_model(env, agent_config)
    output_dir = Path(to_absolute_path(cfg.training.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    model.learn(total_timesteps=cfg.training.total_timesteps)
    model.save(output_dir / cfg.training.checkpoint_name)


if __name__ == "__main__":
    main()
