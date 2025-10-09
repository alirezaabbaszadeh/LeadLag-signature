# RL Lead-Lag Initiative – Standardized 100-Step Action Blueprint

The following blueprint refines and standardizes the previously drafted 100 action plans. Each action plan now follows a consistent template (Objective, Rationale, Prerequisites, Implementation Steps, Deliverables) to ensure clarity, reproducibility, and alignment with top-tier research and engineering standards. Execute the plans sequentially (01 → 100) to deliver a production-grade, research-ready reinforcement learning platform for dynamic lookback selection in lead–lag analysis.

---

## Action Plan 01 – Project Skeleton & Module Boundaries
**Objective:** Establish a clean, extensible repository structure that isolates RL components from existing analytics code.
**Rationale:** A disciplined layout prevents code sprawl, accelerates onboarding, and simplifies CI/CD integration.
**Prerequisites:** Existing repository cloned; agreement on directory naming conventions.
**Implementation Steps:**
1. Create top-level packages `rl/`, `training/`, `evaluation/`, `configs/`, `scripts/`, `docs/`, `results/` with `__init__.py` where applicable.
2. Update `.gitignore` to exclude experiment artifacts (e.g., `results/`, `mlruns/`, `outputs/`, `.hydra/`).
3. Document the new structure in `README.md` with a tree overview and component responsibilities.
**Deliverables:** Repository tree reflecting the new structure; updated README and `.gitignore`.

## Action Plan 02 – Deterministic Environment Setup Guide
**Objective:** Document reproducible environment setup covering Python versioning, dependency isolation, and verification steps.
**Rationale:** Deterministic setups mitigate “works on my machine” issues and satisfy publication reproducibility requirements.
**Prerequisites:** Finalized dependency lists; consensus on package manager (pip/conda/poetry).
**Implementation Steps:**
1. Create `docs/setup.md` with instructions for virtualenv/conda creation, GPU prerequisites, and optional extras (signature, MLflow, W&B).
2. Produce `requirements-base.txt` and `requirements-rl.txt`; explain install order and compatibility notes.
3. Add a smoke-test command (e.g., `python -m tests.smoke_env`) verifying critical imports.
**Deliverables:** Setup guide, requirement files, and passing smoke test output logged in documentation.

## Action Plan 03 – Hydra Configuration Backbone
**Objective:** Centralize configuration management using Hydra for composable experiment definitions.
**Rationale:** Hydra enables hierarchical configs, repeatability, and convenient overrides, reducing human error.
**Prerequisites:** Action Plans 01–02 completed; baseline configuration parameters collected.
**Implementation Steps:**
1. Create `configs/config.yaml` referencing defaults for `env`, `agent`, `training`, `data`, and `logging` groups.
2. Implement sub-configs (e.g., `configs/env/default.yaml`, `configs/agent/ppo.yaml`) mapping to dataclasses.
3. Decorate `training/train_rl.py` with `@hydra.main`, load config dataclasses, and print resolved config for traceability.
**Deliverables:** Working Hydra-powered entry point with sample command documented (e.g., `python training/train_rl.py`).

## Action Plan 04 – RL Configuration Dataclasses
**Objective:** Introduce strongly typed dataclasses encapsulating RL-specific parameters (lookback bounds, reward weights, curriculum, logging).
**Rationale:** Dataclasses ensure validation, IDE support, and prevent misconfigured runs.
**Prerequisites:** Hydra scaffolding operational; parameter catalog defined.
**Implementation Steps:**
1. Create `rl/config.py` with dataclasses (`LeadLagRLConfig`, `RewardConfig`, `CurriculumConfig`, etc.).
2. Add validation logic (e.g., `__post_init__`) for bounds (`min_lookback >= 2`, `max_lookback >= min_lookback`).
3. Map Hydra configs to dataclasses in `train_rl.py`, logging resolved parameter values via MLflow/W&B.
**Deliverables:** Config dataclasses with unit tests validating edge cases and type hints.

## Action Plan 05 – Gymnasium-Compatible LeadLag Environment
**Objective:** Implement `LeadLagEnv` conforming to Gymnasium API, wrapping analyzer computations while exposing lookback as the action.
**Rationale:** A robust environment is the linchpin for RL experimentation and reproducibility.
**Prerequisites:** Analyzer helper functions accessible; config dataclasses available.
**Implementation Steps:**
1. In `rl/envs/leadlag_env.py`, define `LeadLagEnv` with Gymnasium methods (`__init__`, `reset`, `step`, `close`).
2. Register the env via `gymnasium.envs.registration.register` for ease of use (`id="LeadLag-v0"`).
3. Incorporate logging hooks and info dictionary entries (lookback, clamping, reward components).
**Deliverables:** Importable environment passing Gymnasium’s `check_env` validation.

## Action Plan 06 – Single-Window Analyzer Helper
**Objective:** Provide a public analyzer method for computing lead–lag matrices over arbitrary windows without full rolling loops.
**Rationale:** RL steps require fast per-window computation; reusing analyzer internals avoids duplication.
**Prerequisites:** Access to `LeadLagAnalyzer` internals; understanding of existing `_compute_rolling_lead_lag_matrix`.
**Implementation Steps:**
1. Add `compute_window_matrix(price_df, start_idx, end_idx)` in `models/leadlag_analyzer.py` leveraging existing private helpers.
2. Ensure thread/process safety (no shared mutable state) and optional progress-bar suppression.
3. Return matrix plus metadata (asset order, returns) for downstream feature builders.
**Deliverables:** Helper function with unit tests comparing output to rolling implementation for identical windows.

## Action Plan 07 – Observation Encoding Framework
**Objective:** Build modular encoders that transform lead–lag matrices into fixed-size observation vectors.
**Rationale:** Consistent, informative observations are critical for stable learning and ablation studies.
**Prerequisites:** Matrix helper operational; config schema specifying encoder options.
**Implementation Steps:**
1. Create `rl/features/encoder.py` with strategies: full flatten, row/column sums, top-k magnitudes, lookback metadata.
2. Implement selection logic driven by config (e.g., `encoder.name`, `encoder.params`).
3. Write tests verifying shapes, masks, and deterministic behavior given identical inputs.
**Deliverables:** Encoder module with documentation and coverage across encoding strategies.

## Action Plan 08 – Signature Feature Integration
**Objective:** Integrate optional path-signature features leveraging `iisignature` when available.
**Rationale:** Signatures capture path-order information, enhancing lead–lag representation.
**Prerequisites:** Encoder framework in place; optional dependency flagged.
**Implementation Steps:**
1. Create `rl/features/signature.py` detecting `iisignature`; gracefully degrade if absent.
2. Offer configurable signature depth/order and feature selection (e.g., Levy areas).
3. Cache intermediate computations for repeated asset pairs within an episode.
**Deliverables:** Signature encoder with unit/integration tests and configuration examples.

## Action Plan 09 – Observation Normalization Module
**Objective:** Apply running mean/variance normalization to observations for training stability.
**Rationale:** Normalized inputs improve convergence, especially across varying asset universes.
**Prerequisites:** Encoder outputs deterministic; config indicates normalization preferences.
**Implementation Steps:**
1. Implement `rl/features/normalizer.py` using Welford’s algorithm with methods `update`, `normalize`, `state_dict`.
2. Integrate normalizer into `LeadLagEnv`, persisting state for evaluation runs.
3. Save normalizer parameters alongside policy checkpoints (MLflow artifacts).
**Deliverables:** Normalizer module with serialization tests and toggles validated via ablation.

## Action Plan 10 – Dynamic Universe Handling
**Objective:** Support assets entering/exiting the universe without breaking observation dimensionality.
**Rationale:** Real datasets feature sparse coverage; RL agents must handle variable universes.
**Prerequisites:** Observation encoder supports padding/masking; analyzer reports active assets.
**Implementation Steps:**
1. Maintain a mapping of active asset indices per step; pad inactive slots with zeros or sentinel values.
2. Emit an accompanying mask tensor (same length as features) for use by policies/attention modules.
3. Log universe changes in `info` (e.g., `info["dropped_assets"]`).
**Deliverables:** Environment resilient to universe shifts with unit tests covering asset drop/add scenarios.

## Action Plan 11 – Action Wrappers (Absolute vs. Delta Lookback)
**Objective:** Provide interchangeable wrappers for absolute lookback selection and delta-based adjustments.
**Rationale:** Different action semantics support varied exploration strategies and policy architectures.
**Prerequisites:** Base environment accepts direct lookback integers; config specifies action mode.
**Implementation Steps:**
1. Implement `AbsoluteLookbackWrapper` and `DeltaLookbackWrapper` in `rl/envs/action_wrappers.py`.
2. Handle conversion between discrete/continuous action spaces and valid lookback values.
3. Propagate original and transformed actions through `info` for logging.
**Deliverables:** Action wrappers registered and covered by tests ensuring bounds respect and reversibility.

## Action Plan 12 – Lookback Feasibility Enforcement
**Objective:** Guarantee selected lookbacks never exceed available historical data within an episode.
**Rationale:** Prevents index errors and ensures chronological integrity.
**Prerequisites:** Environment has access to current index and data length.
**Implementation Steps:**
1. Compute `allowable_max = min(config.max_lookback, current_index + 1)` each step.
2. Clip or resample actions exceeding bounds; flag via `info["lookback_clamped"]`.
3. Add tests covering early-episode behavior with limited history.
**Deliverables:** Safe action handling validated by negative tests.

## Action Plan 13 – Window Matrix Caching
**Objective:** Reduce redundant analyzer calls by caching `(end_idx, lookback)` computations.
**Rationale:** RL often revisits similar windows; caching cuts latency and CPU usage.
**Prerequisites:** Deterministic analyzer outputs; memory budget defined.
**Implementation Steps:**
1. Implement LRU cache (e.g., `functools.lru_cache` or custom dict with eviction) storing matrices and metadata.
2. Expose cache stats via logging/MLflow for performance monitoring.
3. Provide cache invalidation when dataset pointer advances beyond cached windows.
**Deliverables:** Cache module with benchmarking evidence showing reduced computation time.

## Action Plan 14 – Modular Reward Component Library
**Objective:** Compose total reward from modular components (signal strength, stability, extremity penalties, optional PnL).
**Rationale:** Modular design facilitates experimentation and ablation without rewriting environment code.
**Prerequisites:** Reward design finalized; config schema supports component lists.
**Implementation Steps:**
1. Create `rl/rewards/components.py` implementing reusable functions returning scalar rewards and metadata.
2. Define a registry in `rl/rewards/registry.py` mapping names to component callables.
3. Aggregate rewards in environment, logging per-component contributions in `info`.
**Deliverables:** Reward library with unit tests per component and integration tests verifying combined outputs.

## Action Plan 15 – Reward Normalization & Scaling
**Objective:** Normalize reward components to comparable scales, preventing dominance by any single metric.
**Rationale:** Balanced rewards stabilize training and cross-dataset comparisons.
**Prerequisites:** Reward components implemented; logging infrastructure in place.
**Implementation Steps:**
1. Track running stats per component; apply z-score or min-max scaling as configured.
2. Allow switching between on-policy normalization (environment-level) or callback-level normalization.
3. Expose toggles via config; analyze impact in ablation runs.
**Deliverables:** Normalized reward pipeline with documented defaults and ablation results.

## Action Plan 16 – Trading PnL Reward Augmentation
**Objective:** Integrate a simple trading simulator that converts lead–lag matrices into actionable PnL contributions.
**Rationale:** Aligns reward with real-world profitability while complementing analytical metrics.
**Prerequisites:** Price change data accessible per window; signal thresholds defined.
**Implementation Steps:**
1. Implement strategy (e.g., long top lead assets, short top lag assets) using latest window data.
2. Compute per-step PnL and risk-adjusted metrics (e.g., Sharpe) for reward blending.
3. Parameterize via config to enable/disable or weight PnL contributions.
**Deliverables:** PnL reward component with validation tests using synthetic data.

## Action Plan 17 – Curriculum Scheduling for Lookback Bounds
**Objective:** Gradually expand allowable lookback range based on episode count or performance milestones.
**Rationale:** Curriculum reduces exploration complexity early in training and improves convergence.
**Prerequisites:** Reward baseline established; config supports curriculum parameters.
**Implementation Steps:**
1. Extend config with curriculum schedule (stage thresholds, min/max bounds per stage).
2. Implement `CurriculumManager` updating environment bounds at reset based on stage.
3. Log curriculum transitions via callbacks and MLflow metrics.
**Deliverables:** Curriculum engine with tests simulating progression across stages.

## Action Plan 18 – Randomized Episode Start Indices
**Objective:** Sample diverse starting points within training data to prevent overfitting to fixed regimes.
**Rationale:** Enhances generalization and mirrors online trading scenarios.
**Prerequisites:** Dataset segmentation available; curriculum constraints satisfied.
**Implementation Steps:**
1. Implement `EpisodeSampler` selecting valid start indices respecting lookback requirements and temporal splits.
2. Support stratified sampling by regime labels when provided.
3. Ensure evaluation uses deterministic seeds to maintain comparability.
**Deliverables:** Configurable sampler with tests covering edge cases (near dataset boundaries).

## Action Plan 19 – Multi-Resolution Episode Support
**Objective:** Allow episodes to run at different data granularities (daily, hourly) within training schedules.
**Rationale:** Agents exposed to multi-resolution data learn flexible strategies.
**Prerequisites:** Resampled datasets prepared; curriculum design inclusive of resolution changes.
**Implementation Steps:**
1. Extend config with resolution schedule (e.g., `[{episodes: 0-500, freq: '1D'}, {episodes: 501+, freq: '1H'}]`).
2. Update data loader to select appropriate resampled frame per episode.
3. Adjust observation encoder metadata to include resolution indicator if beneficial.
**Deliverables:** Multi-resolution pipeline validated through smoke tests across frequencies.

## Action Plan 20 – Synthetic Data Generator for Controlled Testing
**Objective:** Produce synthetic time series with known lead–lag structures to validate environment and reward correctness.
**Rationale:** Controlled scenarios expose bugs and confirm theoretical expectations before real-market training.
**Prerequisites:** Statistical specifications defined (delays, noise, amplitude, regime shifts).
**Implementation Steps:**
1. Implement `rl/data/synthetic.py` generating multi-asset series with configurable parameters and seeds.
2. Provide CLI/API to mix synthetic and real data for hybrid training.
3. Unit test by verifying generated lead–lag matrices match expected ground truth.
**Deliverables:** Synthetic generator module with documented usage and validation plots.

## Action Plan 21 – Domain Randomization for Synthetic Regimes
**Objective:** Randomize synthetic parameters per episode to enhance policy robustness.
**Rationale:** Varying synthetic regimes prevent overfitting to a single synthetic pattern.
**Prerequisites:** Synthetic generator operational; config extends to randomization options.
**Implementation Steps:**
1. Define parameter distributions (delays, volatility, correlation) in config.
2. Sample and apply parameters during each `reset`; log in `info` and MLflow.
3. Optionally blend with real data episodes via configurable ratios.
**Deliverables:** Randomized synthetic mode with reproducible logging of sampled parameters.

## Action Plan 22 – Monte Carlo Market Simulator Evaluation
**Objective:** Evaluate policies on Monte Carlo-generated markets modeling various lead–lag behaviors.
**Rationale:** Stress-testing against simulated regimes reveals overfitting and resilience.
**Prerequisites:** Synthetic generator + domain randomization ready; evaluation pipeline flexible.
**Implementation Steps:**
1. Implement `rl/simulation/monte_carlo.py` producing long-run synthetic markets with regime transitions.
2. Add evaluation flag `--simulated` to run policies on Monte Carlo paths.
3. Compare metrics (reward, PnL, drawdown) between real and simulated evaluations.
**Deliverables:** Monte Carlo evaluation reports stored under `results/simulated/` with narrative analysis.

## Action Plan 23 – Adversarial Episode Selection
**Objective:** Identify and evaluate on windows where lead–lag relationships are weak or unstable.
**Rationale:** Ensures robustness by focusing on worst-case scenarios.
**Prerequisites:** Historical metrics (instability scores) computed; evaluation framework modular.
**Implementation Steps:**
1. Compute instability indicators (e.g., high variance in row sums) across rolling windows.
2. Rank windows and store top-K as adversarial test cases.
3. Run trained policies on adversarial sets; compare vs. baselines and log results.
**Deliverables:** Adversarial dataset and evaluation report with actionable insights.

## Action Plan 24 – Per-Asset Fairness & Contribution Analytics
**Objective:** Monitor how often each asset influences actions/rewards to avoid bias.
**Rationale:** Balanced treatment supports interpretability and fairness metrics.
**Prerequisites:** Reward components output per-asset contributions; logging pipeline robust.
**Implementation Steps:**
1. Aggregate reward and action stats per asset or asset pair during evaluation.
2. Compute fairness metrics (e.g., Herfindahl index, Gini coefficient).
3. Visualize contributions in notebooks and store metrics in MLflow.
**Deliverables:** Fairness analytics module with summary charts included in evaluation artifacts.

## Action Plan 25 – Action Masking for Invalid Choices
**Objective:** Provide masks to RL algorithms indicating which actions are currently valid.
**Rationale:** Prevents agents from sampling illegal lookbacks, improving training efficiency.
**Prerequisites:** Action wrappers in place; algorithm supports masks (or custom distribution required).
**Implementation Steps:**
1. Compute boolean mask representing valid discrete actions per step.
2. Return mask via `info["action_mask"]`; adjust training code to consume masks (custom SB3 distribution if needed).
3. Add tests ensuring masked actions never appear in trajectories.
**Deliverables:** Mask-enabled environment with supporting policy logic and validation tests.

## Action Plan 26 – Transformer-Based Observation Encoder
**Objective:** Introduce attention-driven encoders capturing complex inter-asset dependencies.
**Rationale:** Transformers offer flexible context modeling beyond linear flattening.
**Prerequisites:** Base encoder stable; PyTorch dependency available.
**Implementation Steps:**
1. Implement transformer encoder module with configurable depth, heads, and hidden sizes.
2. Integrate with observation pipeline (optionally preceded by linear projection/padding masks).
3. Benchmark on sandbox dataset; document performance vs. baseline encoders.
**Deliverables:** Transformer encoder with config toggles and empirical evaluation results.

## Action Plan 27 – Recurrent PPO (LSTM/GRU) Support
**Objective:** Enable memory-based policies using Stable-Baselines3 Recurrent PPO.
**Rationale:** Temporal dependencies may require memory beyond stacked observations.
**Prerequisites:** Environment supports recurrent hidden state resets; SB3-contrib installed.
**Implementation Steps:**
1. Add policy selection flag (e.g., `policy.type`) allowing `MlpPolicy`, `MlpLstmPolicy`, `MlpGruPolicy`.
2. Handle hidden state storage/reset within training and evaluation loops.
3. Compare learning curves vs. non-recurrent baselines; log results.
**Deliverables:** Recurrent PPO pipeline with documented configuration and performance metrics.

## Action Plan 28 – Attention-Driven Policy Architectures
**Objective:** Implement custom policies applying attention over observation features before action/value heads.
**Rationale:** Attention can dynamically focus on salient assets.
**Prerequisites:** Observation encoder returns mask metadata; PyTorch custom policy integration understood.
**Implementation Steps:**
1. Extend SB3 `ActorCriticPolicy` to include multi-head attention layers consuming observation + mask.
2. Register policy class and expose via config (e.g., `policy.type=attention`).
3. Evaluate attention policies on sandbox dataset; analyze interpretability outputs.
**Deliverables:** Attention policy implementation with unit tests and evaluation results.

## Action Plan 29 – Hierarchical RL Prototype
**Objective:** Explore a two-level architecture (meta-controller + low-level lookback tuner) for complex regime decisions.
**Rationale:** Hierarchical control may capture multi-timescale dynamics more effectively.
**Prerequisites:** Base environment stable; ability to wrap env or design meta-env.
**Implementation Steps:**
1. Design meta-environment where high-level agent selects regime parameters (e.g., reward weights or resolution) every N steps.
2. Train low-level agent conditioned on regime context (e.g., appended to observations).
3. Evaluate hierarchy vs. single-agent baseline on curated scenarios.
**Deliverables:** Prototype hierarchical training script and evaluation report detailing benefits/trade-offs.

## Action Plan 30 – Multi-Objective Reward Handling & Pareto Analysis
**Objective:** Treat reward components as vectors, enabling explicit multi-objective optimization.
**Rationale:** Provides transparency into trade-offs and supports Pareto frontier analysis.
**Prerequisites:** Reward components accessible individually; logging supports vector metrics.
**Implementation Steps:**
1. Modify environment to optionally return reward vectors alongside scalarized rewards.
2. Implement scalarization strategies (weighted sum, Chebyshev) selectable via config.
3. Generate Pareto plots comparing policies; store under `results/pareto/`.
**Deliverables:** Multi-objective evaluation toolkit with documented methodology.

## Action Plan 31 – Reward Weight Annealing Schedules
**Objective:** Dynamically adjust reward component weights over training (e.g., increase stability importance later).
**Rationale:** Guides agent focus as learning progresses, improving convergence.
**Prerequisites:** Reward registry in place; curriculum scheduling mechanism available.
**Implementation Steps:**
1. Add weight schedule definitions (piecewise, linear, exponential) to config.
2. Update reward aggregator to reference current schedule stage each episode.
3. Log active weights per episode and analyze effect in ablation studies.
**Deliverables:** Annealing framework with experiment results demonstrating benefits.

## Action Plan 32 – Reward Smoothing & EMA Filtering
**Objective:** Apply smoothing to noisy reward signals before policy updates.
**Rationale:** Smooth rewards stabilize policy gradients, particularly in volatile regimes.
**Prerequisites:** Reward components aggregated; config includes smoothing options.
**Implementation Steps:**
1. Implement EMA filter configurable by decay factor; apply either in env or callback.
2. Optionally maintain both raw and smoothed rewards for logging/comparison.
3. Evaluate effect on convergence for high-volatility datasets.
**Deliverables:** Reward smoothing feature with documented performance impact.

## Action Plan 33 – Training Callback Suite
**Objective:** Centralize callbacks for logging, checkpointing, curriculum updates, reward scheduling, and early stopping.
**Rationale:** Modular callbacks improve maintainability and reuse across experiments.
**Prerequisites:** Logging systems (MLflow/W&B) integrated; curriculum & annealing schedules implemented.
**Implementation Steps:**
1. Implement callbacks in `training/callbacks.py` (Logging, Checkpoint, Curriculum, RewardScheduler, EarlyStopping).
2. Register callbacks via config list; ensure compatibility with SB3 training loop.
3. Unit test callbacks using minimal training sessions or mocked agents.
**Deliverables:** Callback library with coverage and sample configuration.

## Action Plan 34 – Validation-Based Early Stopping
**Objective:** Terminate training when validation reward plateaus, preserving best checkpoints.
**Rationale:** Prevents overfitting and conserves compute resources.
**Prerequisites:** Validation data split defined; evaluation routine callable during training.
**Implementation Steps:**
1. Reserve validation episodes/time range separate from training and testing.
2. Callback runs validation rollouts every N updates; tracks moving average improvements.
3. Save best-performing model and halt when no improvement after `patience` intervals.
**Deliverables:** Early stopping mechanism with logs showing stop criteria triggered appropriately.

## Action Plan 35 – Hyperparameter Optimization with Optuna
**Objective:** Automate search over RL and reward hyperparameters for optimal performance.
**Rationale:** Systematic HPO replaces manual tuning, increasing performance and reproducibility.
**Prerequisites:** Training script parameterizable via Hydra; sandbox dataset for quick trials.
**Implementation Steps:**
1. Implement `scripts/hpo.py` orchestrating Optuna trials; define search spaces (learning rate, entropy coeff, reward weights, encoder params).
2. Log trial results to MLflow/W&B; store best config JSON.
3. Provide template command for larger-scale HPO on compute clusters.
**Deliverables:** HPO pipeline with documented best-performing configs and search summary plots.

## Action Plan 36 – Seed Sweep Automation
**Objective:** Quantify performance variance by running multiple seeds automatically.
**Rationale:** Publication-level rigor demands reporting mean ± std across seeds.
**Prerequisites:** Training script deterministic given seed; logging aggregated metrics.
**Implementation Steps:**
1. Build `scripts/run_seed_sweep.py` iterating over seed list and invoking `train_rl.py` with overrides.
2. Aggregate results into a combined CSV; compute statistical summaries.
3. Visualize seed variance in notebooks (boxplots, confidence intervals).
**Deliverables:** Seed sweep tooling with sample output integrated into evaluation workflow.

## Action Plan 37 – Rolling Time-Based Cross-Validation
**Objective:** Evaluate model robustness across multiple chronological folds.
**Rationale:** Ensures strategy effectiveness is not confined to a specific era.
**Prerequisites:** Data splits defined; evaluation scripts reusable.
**Implementation Steps:**
1. Partition data into K sequential folds (train on fold i, validate on i, test on i+1).
2. Automate fold iteration using Hydra multirun; aggregate metrics per fold.
3. Present fold-wise results in evaluation report with statistical context.
**Deliverables:** Cross-validation results stored under `results/cv/` with narrative analysis.

## Action Plan 38 – Cross-Universe Generalization Testing
**Objective:** Measure performance when training on one asset set and evaluating on another.
**Rationale:** Tests generalization and avoids overfitting to specific assets.
**Prerequisites:** Multiple universes prepared; environment handles dynamic assets.
**Implementation Steps:**
1. Define training/test universes in config (e.g., `universe.train`, `universe.test`).
2. Train policy on training universe; evaluate on test universe without fine-tuning.
3. Compare metrics vs. in-universe evaluation; document performance gaps.
**Deliverables:** Cross-universe results table and analysis in evaluation report.

## Action Plan 39 – Domain Adaptation via Gradient Reversal
**Objective:** Encourage domain-invariant features across different markets/regimes.
**Rationale:** Enhances transferability to unseen markets.
**Prerequisites:** Attention/encoder modules adaptable; PyTorch training hooks accessible.
**Implementation Steps:**
1. Add discriminator network predicting domain label from encoded observations.
2. Apply gradient reversal layer to minimize domain discrimination while maximizing policy reward.
3. Evaluate adaptation on target domains, comparing to baseline without adaptation.
**Deliverables:** Domain adaptation module with experiments demonstrating improved cross-domain performance.

## Action Plan 40 – Transfer Learning Workflow
**Objective:** Fine-tune policies trained on source markets for new target data efficiently.
**Rationale:** Speeds up deployment to new assets/regimes with limited data.
**Prerequisites:** Model serialization/export stable; data splits defined.
**Implementation Steps:**
1. Modify `train_rl.py` to accept `--load-policy` and `--freeze-layers` options.
2. Track pre- and post-fine-tuning metrics; log to MLflow for comparison.
3. Document recommended fine-tuning hyperparameters and convergence diagnostics.
**Deliverables:** Transfer learning playbook and example results.

## Action Plan 41 – Online Learning Capability
**Objective:** Enable incremental updates as new market data streams in.
**Rationale:** Maintains policy relevance in non-stationary environments.
**Prerequisites:** Data ingestion pipeline; policy checkpointing functioning.
**Implementation Steps:**
1. Implement streaming data loader appending new bars to buffers while respecting temporal integrity.
2. Periodically resume training with recent mini-batches; manage replay buffer via reservoir or prioritized sampling.
3. Monitor catastrophic forgetting by evaluating on historical validation sets.
**Deliverables:** Online update script with monitoring dashboards.

## Action Plan 42 – Asynchronous Algorithm Support (IMPALA/A3C)
**Objective:** Benchmark asynchronous RL algorithms for throughput gains.
**Rationale:** Some workloads benefit from asynchronous rollouts and centralized learners.
**Prerequisites:** Environment picklable; RLlib or CleanRL dependencies installed.
**Implementation Steps:**
1. Implement wrappers bridging `LeadLagEnv` to RLlib (observation/action spaces, masks).
2. Provide configuration and scripts for IMPALA/A3C training.
3. Compare throughput and performance vs. PPO baseline; document findings.
**Deliverables:** Async training scripts with benchmark metrics.

## Action Plan 43 – Distributed Training via Ray or MPI
**Objective:** Scale training across multiple nodes/GPUs.
**Rationale:** Enables large experiments and hyperparameter sweeps.
**Prerequisites:** Async environment wrappers ready; cluster infrastructure accessible.
**Implementation Steps:**
1. Ensure environment serialization and deterministic seeding per worker.
2. Provide sample Ray cluster config and instructions for launching distributed jobs.
3. Collect performance metrics demonstrating scaling efficiency.
**Deliverables:** Distributed training documentation and example logs.

## Action Plan 44 – Mixed Precision Training
**Objective:** Reduce GPU training time and memory usage via AMP.
**Rationale:** Enables larger batch sizes and faster iterations on compatible hardware.
**Prerequisites:** PyTorch >=1.6; GPU environment available.
**Implementation Steps:**
1. Wrap forward/backward passes in `torch.cuda.amp.autocast` and `GradScaler`.
2. Validate numerical stability by comparing to full precision results.
3. Document hardware requirements and fallback to FP32 when unavailable.
**Deliverables:** Mixed precision-enabled training with benchmarking data.

## Action Plan 45 – Analyzer Performance Optimization
**Objective:** Vectorize lead–lag computations and add optional GPU acceleration.
**Rationale:** Reduces per-step latency, enabling larger environments and faster training.
**Prerequisites:** Profiling results from current implementation; GPU libraries (CuPy/Torch) available.
**Implementation Steps:**
1. Profile existing loops; refactor to use vectorized NumPy or Torch operations.
2. Abstract computation backend (CPU/GPU) via strategy pattern; expose via config.
3. Benchmark improvements on representative workloads; ensure numerical parity via tests.
**Deliverables:** Optimized analyzer with performance report and regression tests.

## Action Plan 46 – Signature Computation Caching
**Objective:** Memoize signature results to avoid recomputation for repeated windows.
**Rationale:** Signatures are computationally expensive; caching saves time during training/evaluation.
**Prerequisites:** Signature module implemented; cache policy defined.
**Implementation Steps:**
1. Implement caching (e.g., `functools.lru_cache` or joblib Memory) keyed by asset pair and window indices.
2. Expose cache size/TTL in config to control memory usage.
3. Record cache statistics in logs and MLflow for monitoring.
**Deliverables:** Signature cache with measurable speed improvements and tests ensuring correctness.

## Action Plan 47 – Irregular Timestamp Alignment
**Objective:** Normalize irregular time series before lead–lag computation via resampling/interpolation.
**Rationale:** Ensures consistent window sizes and prevents index errors.
**Prerequisites:** Understanding of data gaps; decision on interpolation strategy.
**Implementation Steps:**
1. Implement alignment utilities handling forward-fill, interpolation, or drop strategies.
2. Integrate alignment in data preprocessing pipeline; mark windows with high missingness.
3. Validate using synthetic irregular datasets; log alignment decisions.
**Deliverables:** Alignment toolkit with tests and documentation.

## Action Plan 48 – Data Integrity Monitoring
**Objective:** Enforce chronological order and detect potential data leaks.
**Rationale:** Prevents inadvertent future-information leakage undermining research validity.
**Prerequisites:** Access to window indices and dataset metadata.
**Implementation Steps:**
1. Before each step, assert `window_end <= current_index` and raise detailed errors on violations.
2. Implement unit tests intentionally violating integrity to confirm detection.
3. Log integrity checks per episode, optionally halting training on repeated failures.
**Deliverables:** Integrity guard with coverage and logging.

## Action Plan 49 – Dataset Versioning with DVC
**Objective:** Track raw and processed data versions to guarantee reproducible experiments.
**Rationale:** DVC ensures data provenance and supports collaborative workflows.
**Prerequisites:** Data stored locally; remote storage configured (optional).
**Implementation Steps:**
1. Initialize DVC in repository; add data pipelines (raw → processed → splits).
2. Document commands for pulling/pushing datasets and locking versions.
3. Integrate DVC stages into Makefile/pipeline scripts.
**Deliverables:** DVC-managed data assets with documented workflow.

## Action Plan 50 – Preprocessing Test Suite
**Objective:** Cover preprocessing utilities (universe selection, resampling, alignment) with unit tests.
**Rationale:** Changes in data pipelines should trigger clear regressions.
**Prerequisites:** Data utilities implemented; synthetic test data prepared.
**Implementation Steps:**
1. Add tests in `tests/test_preprocessing.py` for `selected_uni`, resampling, alignment, missing data handling.
2. Use fixtures representing edge cases (irregular timestamps, sparse assets).
3. Ensure tests run in CI and document coverage goals.
**Deliverables:** Green test suite covering preprocessing edge cases.

## Action Plan 51 – Sandbox Dataset for CI Smoke Tests
**Objective:** Provide lightweight dataset enabling fast end-to-end checks in CI.
**Rationale:** Ensures full pipeline executes without heavy compute.
**Prerequisites:** Data pipeline stable; ability to slice small timeframe.
**Implementation Steps:**
1. Curate sandbox dataset (e.g., 3 assets, 30 days) stored in `data/sandbox/` (tracked via DVC if large).
2. Create preset config referencing sandbox data.
3. Configure CI to run quick training/evaluation using sandbox preset.
**Deliverables:** Sandbox dataset and CI job demonstrating pass/fail.

## Action Plan 52 – Comprehensive CI Pipeline
**Objective:** Automate linting, formatting, tests, notebook execution, and smoke training in CI.
**Rationale:** Maintains quality and reproducibility across contributions.
**Prerequisites:** Lint/test commands defined; sandbox dataset available.
**Implementation Steps:**
1. Add GitHub Actions workflow (or equivalent) running `black --check`, `flake8`, `pytest`, sandbox training, and notebook execution via `papermill`.
2. Cache dependencies for faster runs; document expected runtime.
3. Configure status badges in README.
**Deliverables:** CI pipeline passing with logs stored as artifacts.

## Action Plan 53 – Reproducibility Validator Script
**Objective:** Generate a reproducibility report capturing environment versions, seeds, configs, and data hashes.
**Rationale:** Facilitates auditability and paper appendix material.
**Prerequisites:** Access to config objects, git metadata, DVC info.
**Implementation Steps:**
1. Implement `scripts/validate_reproducibility.py` printing JSON with Python version, package versions, git commit, config hash, data checksums.
2. Execute script at experiment start/end; store output with results.
3. Add CLI flag to fail if mismatches detected vs. expected manifest.
**Deliverables:** Reproducibility artifact automatically generated for each run.

## Action Plan 54 – Structured Logging Architecture
**Objective:** Standardize logging using Python logging with configurable verbosity and handlers.
**Rationale:** Structured logs ease debugging and analysis at scale.
**Prerequisites:** Logging requirements defined; MLflow/W&B integration running.
**Implementation Steps:**
1. Configure loggers (`leadlag`, `leadlag.env`, `leadlag.rewards`, etc.) with formatters including timestamps, experiment IDs.
2. Allow log level and output (console/file) selection via config.
3. Ensure logs integrate with MLflow (artifact upload) and optionally W&B text logs.
**Deliverables:** Logging configuration documented with examples and tests verifying log outputs.

## Action Plan 55 – MLflow Experiment Tracking
**Objective:** Log parameters, metrics, and artifacts for every experiment using MLflow.
**Rationale:** Centralized tracking supports comparison, audit, and reproduction.
**Prerequisites:** MLflow installed; directory permissions available.
**Implementation Steps:**
1. Initialize MLflow run at training start; log configs, seeds, hardware info, metrics (reward components, lookbacks, runtime).
2. Save artifacts (policy checkpoints, normalizer state, plots) to MLflow storage.
3. Document how to launch MLflow UI to browse experiments.
**Deliverables:** MLflow-integrated training script with sample run recorded.

## Action Plan 56 – Optional Weights & Biases Logging
**Objective:** Provide real-time dashboards via W&B for collaborative monitoring.
**Rationale:** Enhances visibility during long experiments; complements MLflow artifacts.
**Prerequisites:** W&B account or offline mode configured; logging hooks planned.
**Implementation Steps:**
1. Add config toggle `logging.wandb.enabled`; initialize with project/run names derived from config hash.
2. Sync key metrics, configuration files, and media (plots) to W&B.
3. Document authentication (API key) and offline mode procedures in `docs/setup.md`.
**Deliverables:** W&B logging integration with sample dashboard screenshot.

## Action Plan 57 – Resource Utilization Monitoring
**Objective:** Record CPU, GPU, and memory usage to diagnose bottlenecks.
**Rationale:** Data-driven optimization ensures efficient use of compute resources.
**Prerequisites:** `psutil`/`GPUtil` dependencies installed; logging infrastructure ready.
**Implementation Steps:**
1. Implement monitoring utility sampling resources every N seconds during training.
2. Log stats to MLflow/W&B; store raw CSV under `results/resource_usage/`.
3. Visualize resource trends in notebooks.
**Deliverables:** Resource monitoring logs and visualizations integrated into evaluation.

## Action Plan 58 – Latency Profiling & Benchmarking
**Objective:** Measure per-step and per-component latency to target optimizations.
**Rationale:** Profiling guides performance engineering and hardware planning.
**Prerequisites:** Resource monitoring running; ability to isolate components.
**Implementation Steps:**
1. Instrument environment steps and analyzer calls using `time.perf_counter`.
2. Summarize latency stats per episode; log to MLflow.
3. Provide `scripts/profile_training.py` to run `pyinstrument`/`cProfile` and generate flamegraphs.
**Deliverables:** Latency benchmarks with documentation referencing optimization priorities.

## Action Plan 59 – Robust Error Handling in Analyzer/Environment
**Objective:** Gracefully handle numerical issues (singular matrices, missing data) without crashing training.
**Rationale:** Improves resilience during large-scale experiments.
**Prerequisites:** Analyzer helper accessible; logging configured.
**Implementation Steps:**
1. Wrap critical computations in try/except; catch specific numerical exceptions.
2. Apply fallback strategies (e.g., skip reward component, assign penalty, continue) while logging detailed diagnostics.
3. Unit test failure scenarios to ensure predictable behavior.
**Deliverables:** Resilient environment/analyzer with documented error policies.

## Action Plan 60 – Deterministic Evaluation Mode
**Objective:** Ensure evaluation runs yield identical trajectories when seeded.
**Rationale:** Facilitates qualitative analysis and regression testing.
**Prerequisites:** Environment seeding supported; deterministic policy wrapper available.
**Implementation Steps:**
1. Add CLI flag `--seed` to evaluation script; propagate to environment, numpy, torch, random.
2. Implement deterministic policy wrapper selecting mean/argmax actions during evaluation if requested.
3. Save action sequences and rewards for reproducibility.
**Deliverables:** Deterministic evaluation logs enabling exact reproduction of results.

## Action Plan 61 – Baseline Strategy Suite
**Objective:** Establish non-RL baselines (fixed lookbacks, heuristics) for fair comparisons.
**Rationale:** Baselines contextualize RL performance and support publications.
**Prerequisites:** Analyzer operations accessible; evaluation metrics defined.
**Implementation Steps:**
1. Implement baseline runners using static lookbacks (e.g., 10, 30, 90) and heuristic adjustments (volatility-based, correlation-based).
2. Compute identical metrics as RL evaluation; log to MLflow.
3. Include baselines in comparative plots/tables.
**Deliverables:** Baseline results repository and integration into evaluation workflow.

## Action Plan 62 – Differential Regression Testing vs. Baselines
**Objective:** Automatically detect when RL performance drops below baseline thresholds.
**Rationale:** Protects against regressions during development.
**Prerequisites:** Baseline metrics stored; deterministic evaluation available.
**Implementation Steps:**
1. Implement test executing RL and baseline on sandbox dataset with fixed seed.
2. Assert RL reward ≥ baseline reward − tolerance; fail if violated.
3. Integrate test into CI pipeline.
**Deliverables:** Regression test ensuring RL does not underperform baseline unexpectedly.

## Action Plan 63 – Financial Metric Computation Module
**Objective:** Calculate Sharpe, Sortino, max drawdown, Calmar, win rate for RL and baselines.
**Rationale:** Financial metrics are mandatory for publication-quality evaluation.
**Prerequisites:** PnL series available; evaluation pipeline modular.
**Implementation Steps:**
1. Implement metric calculators in `evaluation/metrics.py` with vectorized NumPy/pandas operations.
2. Validate formulas via unit tests and cross-check with known results.
3. Integrate metrics into evaluation reports and MLflow logging.
**Deliverables:** Financial metric outputs stored in CSV/LaTeX and visualized in notebooks.

## Action Plan 64 – Automated Evaluation Report Generation
**Objective:** Assemble plots, tables, and textual summaries into a publication-ready report.
**Rationale:** Streamlines dissemination and archival of experiment results.
**Prerequisites:** Evaluation metrics/plots generated; templating tool selected.
**Implementation Steps:**
1. Use Jinja2 + LaTeX (or nbconvert) to compile evaluation artifacts into PDF/HTML.
2. Include sections (Experiment setup, Metrics, Plots, Ablations, Baselines, Conclusions).
3. Command `python evaluation/generate_report.py` outputs report under `reports/`. 
**Deliverables:** Automated report generation pipeline with sample output committed.

## Action Plan 65 – RL Analysis Notebook
**Objective:** Provide interactive analysis of policy behavior, lookback trajectories, and reward components.
**Rationale:** Supports qualitative insights and publication figures.
**Prerequisites:** Evaluation data stored; notebooks executed in CI.
**Implementation Steps:**
1. Create `notebooks/rl_agent_analysis.ipynb` with data loading utilities.
2. Plot lookback distributions, reward decomposition, fairness metrics, anomaly overlays.
3. Document interpretation notes for each visualization.
**Deliverables:** Executable notebook saved with outputs and referenced in README.

## Action Plan 66 – Notebook Execution in CI
**Objective:** Guarantee notebooks remain current and reproducible.
**Rationale:** Prevents outdated instructions/results from persisting unnoticed.
**Prerequisites:** Sandbox dataset; CI pipeline running.
**Implementation Steps:**
1. Execute notebooks via `papermill` or `nbconvert --execute` within CI using sandbox preset.
2. Fail pipeline on execution error; optionally compare generated outputs vs. stored reference.
3. Cache heavy computations or parameterize notebooks for fast execution.
**Deliverables:** CI job ensuring notebooks execute successfully.

## Action Plan 67 – Ablation Study Automation
**Objective:** Systematically disable/enable components (reward terms, encoders) to evaluate contributions.
**Rationale:** Builds evidence for design choices in publications.
**Prerequisites:** Config supports toggles; evaluation pipeline flexible.
**Implementation Steps:**
1. Implement `evaluation/run_ablation.py` iterating over configurations (e.g., stability weight=0, no normalization).
2. Aggregate results into summary tables/plots with statistical comparisons.
3. Include ablation findings in reports/notebooks.
**Deliverables:** Ablation toolkit with documented command examples and results.

## Action Plan 68 – CEM-Based Reward Weight Optimization
**Objective:** Tune reward weights using Cross-Entropy Method (CEM) for performance maximization.
**Rationale:** Provides alternative to gradient-free search when HPO is heavy.
**Prerequisites:** Reward components normalized; evaluation routine quick (sandbox).
**Implementation Steps:**
1. Implement CEM loop sampling reward weight vectors, evaluating via short rollouts.
2. Update distribution parameters toward best-performing samples.
3. Output optimized weights and incorporate into main configs.
**Deliverables:** CEM optimizer script with convergence plots and selected weights.

## Action Plan 69 – Anomaly Detection on Lead–Lag Matrices
**Objective:** Flag unusual market states based on reconstruction errors or statistical thresholds.
**Rationale:** Supports risk monitoring and interpretability.
**Prerequisites:** Historical matrix dataset; anomaly detection model selection (autoencoder, z-score).
**Implementation Steps:**
1. Train anomaly detector on historical matrices; define alert thresholds.
2. Run detector during evaluation; log anomaly scores alongside rewards.
3. Visualize anomalies with overlays in notebooks/reports.
**Deliverables:** Anomaly detection module with evaluation results and visualizations.

## Action Plan 70 – Policy Interpretability Toolkit
**Objective:** Quantify feature importance influencing policy decisions using SHAP/Integrated Gradients.
**Rationale:** Enhances transparency for stakeholders and publications.
**Prerequisites:** Policy implemented in PyTorch; observation encoder exposes feature mapping.
**Implementation Steps:**
1. Integrate SHAP or Captum to compute attributions for representative episodes.
2. Aggregate results by asset pair/feature; plot importance heatmaps.
3. Document methodology and key findings in notebooks/reports.
**Deliverables:** Interpretability analysis artifacts incorporated into evaluation.

## Action Plan 71 – Gradient Checking for Custom Modules
**Objective:** Verify gradients through custom encoders/policies to ensure differentiability.
**Rationale:** Prevents subtle bugs leading to unstable training.
**Prerequisites:** PyTorch modules implemented; ability to run double precision tests.
**Implementation Steps:**
1. Use `torch.autograd.gradcheck` on custom layers (attention, transformers, normalization).
2. Create unit tests covering gradient checks with synthetic inputs.
3. Document tolerances and dtype considerations.
**Deliverables:** Gradient-check tests passing in CI.

## Action Plan 72 – Policy Distillation Pipeline
**Objective:** Compress complex policies into smaller models while retaining performance.
**Rationale:** Supports deployment and latency constraints.
**Prerequisites:** Teacher policy trained; student architecture defined.
**Implementation Steps:**
1. Implement distillation training loop minimizing KL divergence between teacher and student action distributions.
2. Evaluate student vs. teacher on validation/evaluation datasets.
3. Export distilled model and document performance trade-offs.
**Deliverables:** Distilled policy artifacts and comparison metrics.

## Action Plan 73 – TorchScript/ONNX Export Utility
**Objective:** Export trained policies for deployment in production or research replication.
**Rationale:** Facilitates integration with non-Python environments.
**Prerequisites:** Policy modules TorchScript/ONNX compatible; normalization state accessible.
**Implementation Steps:**
1. Implement `scripts/export_policy.py` generating TorchScript and ONNX variants.
2. Validate exports by running inference tests comparing outputs to original policy.
3. Document deployment instructions and limitations.
**Deliverables:** Export scripts with example artifacts stored under `results/exports/`.

## Action Plan 74 – Deterministic Policy Wrapper for Evaluation
**Objective:** Provide a wrapper enforcing deterministic action selection during evaluation.
**Rationale:** Required for fair comparison and reproducible backtests.
**Prerequisites:** Policy outputs stochastic distribution; evaluation script modular.
**Implementation Steps:**
1. Implement wrapper returning mean or argmax actions (depending on discrete/continuous space).
2. Integrate with evaluation CLI (`--deterministic` flag).
3. Compare deterministic vs. stochastic metrics and log differences.
**Deliverables:** Deterministic evaluation capability integrated into pipeline.

## Action Plan 75 – Lookback Change Rate Limiting
**Objective:** Restrict lookback adjustments per step to mimic realistic operational constraints.
**Rationale:** Prevents erratic behavior and aligns with trading system limitations.
**Prerequisites:** Action wrappers accessible; config specifies `max_delta`.
**Implementation Steps:**
1. Track previous lookback; clamp new value within ±`max_delta`.
2. Optionally apply penalty when clamp occurs; log in `info`.
3. Test scenarios with extreme action sequences to verify limiter effectiveness.
**Deliverables:** Rate-limited environment with tests covering boundary conditions.

## Action Plan 76 – Market Microstructure Safety Filters
**Objective:** Ensure selected lookbacks satisfy liquidity and data sufficiency constraints.
**Rationale:** Avoids decisions based on sparse/unreliable data.
**Prerequisites:** Precomputed microstructure metrics per window (volume, trades).
**Implementation Steps:**
1. Compute metrics during preprocessing; expose to environment per step.
2. Reject or penalize actions violating thresholds; provide informative logging.
3. Include safety compliance stats in evaluation outputs.
**Deliverables:** Safety layer integrated with environment and evaluation logs.

## Action Plan 77 – Reward/Feature Plugin Architecture
**Objective:** Allow external contributors to register custom reward or feature components without modifying core code.
**Rationale:** Enhances extensibility and collaborative experimentation.
**Prerequisites:** Reward/feature registries implemented; plugin interface defined.
**Implementation Steps:**
1. Create registry pattern using entry points or manual registration functions.
2. Document plugin API (expected inputs/outputs, configuration schema).
3. Provide sample plugin implementation (e.g., mutual information feature) demonstrating usage.
**Deliverables:** Plugin system with example extension and documentation.

## Action Plan 78 – Mutual Information Features & Rewards
**Objective:** Integrate MI-based metrics as alternative signals for observation or reward components.
**Rationale:** Captures non-linear dependencies beyond correlation/signature.
**Prerequisites:** Plugin system ready; MI estimator selected.
**Implementation Steps:**
1. Implement MI estimator (k-NN or histogram-based) for asset pairs.
2. Normalize outputs and expose as observation features or reward components via plugin registry.
3. Evaluate impact in ablation studies.
**Deliverables:** MI feature/reward module with performance analysis.

## Action Plan 79 – Automated Changelog Generation
**Objective:** Maintain up-to-date changelog highlighting RL module updates.
**Rationale:** Supports transparency and release management.
**Prerequisites:** Conventional commit style or tagging conventions.
**Implementation Steps:**
1. Integrate `git-cliff` or similar tool; configure to categorize RL-related changes.
2. Update release workflow to regenerate changelog per tag.
3. Store changelog under `CHANGELOG.md` with latest entries.
**Deliverables:** Automated changelog pipeline and documented release process.

## Action Plan 80 – Bilingual Documentation (Persian & English)
**Objective:** Provide localized documentation for diverse stakeholders.
**Rationale:** Enhances accessibility and community engagement.
**Prerequisites:** Key documents stable; translation resources available.
**Implementation Steps:**
1. Mirror essential docs (`README`, setup, quickstart) into `docs/i18n/en/` and `docs/i18n/fa/`.
2. Use templates or translation workflow to keep versions synchronized.
3. Indicate translation availability and contribution guidelines in README.
**Deliverables:** Bilingual docs with version tracking and update procedures.

## Action Plan 81 – Workflow Automation via Makefile/Invoke
**Objective:** Simplify recurring tasks (install, preprocess, train, evaluate, report).
**Rationale:** Encourages consistent execution and reduces manual error.
**Prerequisites:** Commands for each stage defined; Hydra overrides documented.
**Implementation Steps:**
1. Create `Makefile` (or Invoke tasks) with targets: `setup`, `lint`, `test`, `train`, `evaluate`, `report`, `hpo`.
2. Ensure tasks accept environment variables/arguments for config overrides.
3. Document usage in README and CI.
**Deliverables:** Automation script with sample commands and integration into CI/pipeline runner.

## Action Plan 82 – Full Pipeline Orchestration Script
**Objective:** Provide single command executing end-to-end workflow (preprocess → train → evaluate → report).
**Rationale:** Facilitates reproducibility and automated experiments.
**Prerequisites:** Individual stages operational; Makefile tasks available.
**Implementation Steps:**
1. Implement `scripts/run_full_pipeline.py` orchestrating stages with logging and error handling.
2. Accept Hydra overrides propagated to subcommands.
3. Store combined artifacts in timestamped directories; log summary at completion.
**Deliverables:** Orchestration script with documented CLI usage and sample run log.

## Action Plan 83 – Preset Configuration Library
**Objective:** Offer curated presets (crypto daily, equities hourly, sandbox) for rapid experimentation.
**Rationale:** Lowers barrier to entry and standardizes experiment descriptions.
**Prerequisites:** Hydra config structure mature; data sources organized.
**Implementation Steps:**
1. Add preset YAMLs under `configs/presets/` referencing environment, data, and agent defaults.
2. Document usage (`python training/train_rl.py +preset=crypto_daily`).
3. Ensure presets cover training, evaluation, and reporting steps.
**Deliverables:** Preset library with documentation and tested commands.

## Action Plan 84 – Experiment Comparison CLI
**Objective:** Compare metrics across multiple experiment runs for quick insights.
**Rationale:** Simplifies post-hoc analysis and presentation preparation.
**Prerequisites:** MLflow/W&B logs accessible; metrics standardized.
**Implementation Steps:**
1. Implement `scripts/compare_runs.py` pulling data from MLflow or saved CSVs.
2. Output tables/plots showing metric differences; support filters (tags, date ranges).
3. Allow export to Markdown/PDF for documentation.
**Deliverables:** Comparison CLI with sample outputs demonstrating functionality.

## Action Plan 85 – Bootstrap Confidence Intervals for Metrics
**Objective:** Quantify statistical confidence in evaluation metrics via bootstrapping.
**Rationale:** Strengthens empirical claims in publications.
**Prerequisites:** Evaluation metrics per episode available; statistical libraries accessible.
**Implementation Steps:**
1. Implement bootstrap resampling for rewards/PnL; compute mean, variance, and confidence intervals.
2. Include intervals in summary tables and plots.
3. Document methodology and default bootstrap sample counts.
**Deliverables:** Bootstrap-enhanced evaluation outputs.

## Action Plan 86 – Regime-Aware Evaluation
**Objective:** Analyze performance per market regime (bull, bear, sideways).
**Rationale:** Reveals strengths/weaknesses across conditions.
**Prerequisites:** Regime labeling method defined (e.g., volatility/trend filters).
**Implementation Steps:**
1. Tag historical data with regime labels during preprocessing.
2. Evaluate policies separately per regime; log metrics to MLflow.
3. Visualize regime-specific performance in reports/notebooks.
**Deliverables:** Regime-aware evaluation artifacts and commentary.

## Action Plan 87 – Interactive Streamlit Dashboard
**Objective:** Provide real-time, interactive visualization of experiment metrics.
**Rationale:** Accelerates exploratory analysis and stakeholder demos.
**Prerequisites:** MLflow/W&B data accessible; Streamlit dependency installed.
**Implementation Steps:**
1. Build `dashboard/app.py` loading metrics and artifacts; display reward curves, lookback histograms, anomalies, resource usage.
2. Include filters (experiment, preset, regime) for flexible exploration.
3. Document launch instructions in README (`streamlit run dashboard/app.py`).
**Deliverables:** Streamlit app with screenshots and usage guide.

## Action Plan 88 – Replay Logger & Inspector
**Objective:** Capture rollouts for offline analysis and debugging.
**Rationale:** Enables granular inspection of agent behavior and manual auditing.
**Prerequisites:** Environment exposes observations/actions/info; storage path defined.
**Implementation Steps:**
1. Implement `rl/utils/replay_logger.py` storing trajectories (Parquet/JSON) with metadata.
2. Hook logger into training/evaluation (configurable frequency).
3. Build `evaluation/inspect_rollout.py` to visualize or summarize stored rollouts.
**Deliverables:** Replay logging system with example outputs and visualization scripts.

## Action Plan 89 – Reward Snapshot Testing
**Objective:** Freeze reference reward outputs for deterministic scenarios to detect unintended changes.
**Rationale:** Safeguards against silent regressions in reward logic.
**Prerequisites:** Synthetic scenarios defined; reward components deterministic given seeds.
**Implementation Steps:**
1. Generate snapshot data for representative scenarios; store under `tests/data/reward_snapshots/`.
2. Implement tests comparing current rewards to snapshots within tolerances.
3. Provide script to regenerate snapshots when intentional changes occur (with review).
**Deliverables:** Snapshot tests integrated into CI.

## Action Plan 90 – Training Fail-Safe Mechanisms
**Objective:** Monitor metrics during training and abort on catastrophic divergence.
**Rationale:** Saves compute resources and alerts engineers to issues quickly.
**Prerequisites:** Logging of metrics per update; callback system.
**Implementation Steps:**
1. Implement fail-safe callback monitoring rewards, lookback magnitudes, entropy; set thresholds for anomalies (NaNs, spikes).
2. On breach, gracefully stop training, dump diagnostics, and alert via logs.
3. Test fail-safe with simulated anomalies.
**Deliverables:** Fail-safe system with documented thresholds and behavior.

## Action Plan 91 – Gymnasium Compliance Testing
**Objective:** Run Gymnasium `check_env` to ensure API adherence.
**Rationale:** Guarantees compatibility with third-party algorithms and wrappers.
**Prerequisites:** Environment implemented; dependencies installed.
**Implementation Steps:**
1. Add unit test invoking `gymnasium.utils.env_checker.check_env(LeadLagEnv)`.
2. Resolve reported issues (dtype mismatches, missing `terminated/truncated`).
3. Keep test in CI to catch future regressions.
**Deliverables:** Passing compliance test documented in CI logs.

## Action Plan 92 – AsyncVectorEnv Integration
**Objective:** Support asynchronous vectorized environments for higher throughput.
**Rationale:** Improves sample collection speed on multi-core machines.
**Prerequisites:** Environment picklable; action/observation serialization safe.
**Implementation Steps:**
1. Implement factory for `AsyncVectorEnv` creation in training script; ensure unique seeds per worker.
2. Handle shared analyzer resources (copy or reinitialize per process).
3. Benchmark performance vs. synchronous `SubprocVecEnv`; log results.
**Deliverables:** Async environment support with documented speedups.

## Action Plan 93 – Cross-Framework Compatibility (RLlib, CleanRL)
**Objective:** Ensure environment works with alternative RL frameworks.
**Rationale:** Broadens adoption and validates API design.
**Prerequisites:** Gymnasium compatibility confirmed; wrappers for masks and info ready.
**Implementation Steps:**
1. Implement compatibility layer adapting observation/action spaces for RLlib (e.g., Dict spaces, masks).
2. Provide sample scripts showing RLlib and CleanRL usage.
3. Document known limitations and adjustments in README.
**Deliverables:** Cross-framework examples and documentation.

## Action Plan 94 – Meta-Learning (MAML/PEARL) Experiments
**Objective:** Explore fast adaptation approaches for new regimes using meta-RL.
**Rationale:** Addresses non-stationarity by enabling rapid policy updates.
**Prerequisites:** Task definitions (regime-specific datasets); meta-learning frameworks.
**Implementation Steps:**
1. Implement meta-learning training loop (MAML or PEARL) interfacing with environment.
2. Train on multiple regime tasks; evaluate adaptation speed on unseen tasks.
3. Document results and compare with standard PPO.
**Deliverables:** Meta-learning experiment scripts and analysis.

## Action Plan 95 – Thompson Sampling Exploration Wrapper
**Objective:** Incorporate uncertainty-driven exploration inspired by Thompson sampling.
**Rationale:** Balances exploitation and exploration beyond entropy bonuses.
**Prerequisites:** Reward logging per action; ability to sample from estimated distributions.
**Implementation Steps:**
1. Maintain posterior-like statistics (e.g., beta/normal approximations) per action or state cluster.
2. Adjust action selection probabilities based on sampled parameters.
3. Evaluate exploration efficiency vs. baseline; log metrics.
**Deliverables:** Thompson-style exploration module with evaluation results.

## Action Plan 96 – Mutual Information Reward Component
**Objective:** Reward actions producing high mutual information between lead/lag assets.
**Rationale:** Encourages discovery of non-linear dependencies.
**Prerequisites:** MI estimator implemented; reward registry extensible.
**Implementation Steps:**
1. Compute MI for relevant asset pairs; aggregate into scalar reward with normalization.
2. Configure weighting via reward registry; run ablations.
3. Document effect on performance relative to correlation-based rewards.
**Deliverables:** MI reward component integrated and evaluated.

## Action Plan 97 – Backward Compatibility Regression Tests
**Objective:** Ensure analyzer updates remain compatible with existing trained models.
**Rationale:** Prevents silent breakage of archived policies.
**Prerequisites:** Archive of representative models; deterministic evaluation pipeline.
**Implementation Steps:**
1. Store baseline policies and evaluation metrics (frozen).
2. After analyzer changes, re-run evaluations; assert metrics within tolerance of archived values.
3. Alert when retraining required due to significant deviations.
**Deliverables:** Regression tests guarding analyzer–policy compatibility.

## Action Plan 98 – Bilingual README and Quickstart
**Objective:** Provide synchronized Persian and English README sections focusing on RL workflow.
**Rationale:** Enhances accessibility for the target audience.
**Prerequisites:** Bilingual documentation framework from Plan 80.
**Implementation Steps:**
1. Split README into language-specific sections or separate files (`README.md`, `README.fa.md`).
2. Keep RL quickstart instructions synchronized; automate checks if possible.
3. Highlight translation status badges at top of README.
**Deliverables:** Updated README(s) with bilingual content.

## Action Plan 99 – Deployment Artifact Packaging
**Objective:** Bundle policy, normalizer state, configs, reproducibility manifest, and inference scripts for deployment.
**Rationale:** Ensures ready-to-use packages for production or academic distribution.
**Prerequisites:** Export utilities, reproducibility validator, normalization state available.
**Implementation Steps:**
1. Implement `scripts/package_artifact.py` assembling policy (TorchScript/ONNX), normalizer, configs, reproducibility report, inference README.
2. Validate package via smoke test script executing deterministic rollout.
3. Store packaged artifacts under `results/packages/` and document structure.
**Deliverables:** Deployable artifact bundles with validation logs.

## Action Plan 100 – Continuous Deployment & Release Workflow
**Objective:** Define release process automating artifact packaging, changelog updates, and documentation refresh.
**Rationale:** Streamlines dissemination of new research milestones and ensures consistency.
**Prerequisites:** Artifact packaging, changelog automation, documentation pipeline ready.
**Implementation Steps:**
1. Create release script (or GitHub Action) executing tests, packaging artifacts, updating changelog, publishing docs.
2. Tag releases with semantic versioning; attach artifacts and reports.
3. Document release checklist including reproducibility validation and distribution channels.
**Deliverables:** Formal release workflow with automation scripts and documentation.

