# LeadLag Signature RL Extensions

This repository augments the original lead–lag analysis toolkit with a
reinforcement learning pipeline that can adapt the lookback window dynamically.
The new `leadlag_rl` package introduces a Gym-compatible environment, training
entry-points based on Hydra configurations, and helper utilities for evaluation
and agent construction.

## Repository layout

```
LeadLag-signature/
├── configs/              # Hydra configuration hierarchy
├── data/                 # Expected location for raw and processed datasets
├── notebooks/            # Exploratory notebooks from the original project
├── preprocessing_data/   # Data preparation utilities from the original project
├── models/               # Core lead–lag analysis implementation
├── results/              # Output directory for trained models and logs
├── scripts/              # Command-line helpers (e.g. training launcher)
└── src/leadlag_rl/       # Reinforcement learning package
```

## Quick start

1. **Prepare data** – Place a price history file at
   `data/processed/prices.csv` (or update `configs/data/local_csv.yaml` to point
   to your dataset). The file must be indexed by dates and contain one column per
   asset.
2. **Install dependencies** – Besides the existing project requirements you need
   `hydra-core`, `stable-baselines3`, and optionally `sb3-contrib` for recurrent
   PPO policies.
3. **Run training** – Execute `scripts/run_training.sh` to start the default
   experiment:

   ```bash
   ./scripts/run_training.sh
   ```

   Hydra will create a dated output directory containing configuration snapshots
   and logs. Trained policies are saved under `results/checkpoints/` by default.

4. **Customise experiments** – Override configuration values directly from the
   command line, for example to enable a recurrent policy and adjust the lookback
   bounds:

   ```bash
   ./scripts/run_training.sh \
     agent.use_lstm=true \
     env.min_lookback=10 env.max_lookback=120
   ```

## Key components

- `leadlag_rl.envs.LeadLagEnv` – Gym environment that exposes the lookback
  length as the agent's action and summarises the current lead–lag matrix as a
  compact observation vector.
- `leadlag_rl.agents.create_ppo_model` – Factory for constructing PPO or
  recurrent PPO agents using Stable-Baselines3.
- `leadlag_rl.training.train` – Hydra entry-point that wires together the data
  loaders, environment, and agent training loop.
- `leadlag_rl.evaluation.rollout_policy` – Helper for evaluating arbitrary
  policies within the environment.

Refer to the configuration files in `configs/` to explore the available
hyper-parameters and dataset options.
