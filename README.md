# LeadLag Signature Analysis

This repository contains utilities to study lead–lag relationships between crypto assets using a
variety of statistical and path-signature based techniques. The core logic lives in the
`LeadLagAnalyzer` class, which can evaluate different correlation metrics (cross-correlation,
dynamic time warping, signatures, mutual information, etc.) and aggregate them into rolling
leader–follower scores.

## Repository layout

```
├── models/
│   ├── LeadLag_main.py        # LeadLagAnalyzer implementation and configuration helpers
│   └── leadlag_rl_env.py      # Gym environment for adaptive lookback selection
├── notebooks/
│   └── LeadLag_signature.ipynb  # Example workflow that orchestrates preprocessing and analysis
├── preprocessing_data/
│   └── preprocessing.py       # Utility functions for resampling and preparing market data
├── raw_data/                  # Sample market data used in the notebook
└── README.md
```

## Getting started

1. **Create an environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use `.venv\\Scripts\\activate`
   ```

2. **Install the Python requirements**

   The notebook expects a handful of scientific packages. Install them with:
   ```bash
   pip install pandas numpy scipy scikit-learn tqdm p_tqdm dcor iisignature quantstats
   ```
   Some features (e.g., signature transforms or distance correlation) are optional; the
   implementation automatically falls back when the corresponding packages are unavailable.

3. **Launch the demo notebook**
   ```bash
   jupyter notebook notebooks/LeadLag_signature.ipynb
   ```

## Example workflow

The notebook demonstrates a typical workflow:

1. Load price, turnover, and universe data from `raw_data/`.
2. Resample the intraday bars to a desired interval with `resample_crypto_data`.
3. Select a coin universe with `selected_uni` based on liquidity and availability.
4. Forward-fill missing values and align the universe using `preprocess_ffill`.
5. Instantiate `LeadLagAnalyzer` with a configuration dictionary specifying the analysis
   method (cross-correlation, DTW, signature features, etc.).
6. Call `analyze` to obtain a rolling lead–lag score matrix and
   `leader_follower_detector` to summarize the current leaders and followers.

A minimal Python example (outside of the notebook) looks like:
```python
from models.LeadLag_main import LeadLagAnalyzer
from preprocessing_data.preprocessing import resample_crypto_data, selected_uni, preprocess_ffill
import pandas as pd

close_price = pd.read_csv('raw_data/1H_prices_20250811.csv', index_col='date', parse_dates=True)
close_price = resample_crypto_data(close_price, '1D')

universe_meta = pd.read_csv('raw_data/universe_data.csv')
universe = selected_uni(close_price, universe_meta, maximum_coin=20)
close_price, universe = preprocess_ffill(close_price, universe, end_date='2025-01-01')

config = {
    'method': 'dtw',
    'lookback': 30,
    'update_freq': 1,
    'use_parallel': False,
    'Scaling_Method': 'mean-centering',
    'dtw': {'quantiles': 4}
}

analyzer = LeadLagAnalyzer(config, universe)
lead_lag_matrix = analyzer.analyze(close_price, return_rolling=True)
leaders = analyzer.leader_follower_detector(lead_lag_matrix, {
    'method': 'percentile',
    'agg_func': 'mean',
    'top_percentile': 50,
    'bottom_percentile': 50
})
```

## Data

The `raw_data/` folder contains sample CSV files (prices, volumes, and universe definitions)
that make the notebook reproducible. Replace these with your proprietary data if needed. Make
sure that price files use a `date` column that can be parsed into a `DatetimeIndex`.

## Reinforcement-learning environment

The module `models/leadlag_rl_env.py` exposes a Gym-compatible environment that
lets an RL agent learn how to adapt the lookback window dynamically while
reusing the existing analysis code.

```python
from models.leadlag_rl_env import LeadLagEnv, RewardWeights
from models.LeadLag_main import LeadLagConfig
import pandas as pd

prices = pd.read_csv("raw_data/1H_prices_20250811.csv", index_col="date", parse_dates=True)

config = LeadLagConfig(lag=1, lookback=30, update_freq=1, use_parallel=False)
env = LeadLagEnv(
    price_data=prices,
    config=config,
    lookback_range=(5, 120),
    reward_weights=RewardWeights(signal=1.0, stability=0.5, extremity=0.05),
)

observation, _ = env.reset()
observation, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

The environment provides a compact observation vector (row sums, extremes and
signal intensity) together with a composite reward that balances signal strength
and temporal stability while discouraging extreme lookback values.  This makes
it straightforward to plug the simulator into frameworks such as
Stable-Baselines3 or RLlib.

## License

The project inherits the license of this repository. If none is specified, treat the content as
all rights reserved.
