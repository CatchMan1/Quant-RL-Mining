# AlphaSAGE: Structure-Aware Alpha Mining with GFlowNets

An end-to-end research framework for **automatically discovering quantitative
alpha factors** with **Generative Flow Networks (GFlowNets)**. Instead of
hand-crafting factor expressions, the system learns a generative policy that
samples symbolic factor expressions (in Reverse Polish Notation) proportional to
their downstream predictive quality, while a **structure-aware graph encoder**
(RGCN) and a **self-supervised consistency reward** encourage the discovery of a
*structurally diverse* factor pool rather than a single repeated pattern.

The project targets a single-asset, high-frequency setting (15-minute K-line
bars) and covers the full loop: data loading → GFlowNet training → factor-pool
management → factor export for downstream backtesting.

---

## Table of Contents

- [Motivation](#motivation)
- [Key Contributions](#key-contributions)
- [Method Overview](#method-overview)
  - [Factor Discovery as a GFlowNet](#factor-discovery-as-a-gflownet)
  - [State and Action Space](#state-and-action-space)
  - [Structure-Aware Encoder](#structure-aware-encoder)
  - [Multi-Component Reward](#multi-component-reward)
  - [Trajectory Balance with Entropy Regularization](#trajectory-balance-with-entropy-regularization)
  - [Mask Dropout and Grammatical Validity](#mask-dropout-and-grammatical-validity)
  - [Alpha Pool Management](#alpha-pool-management)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Format](#data-format)
- [Usage](#usage)
  - [1. Train a factor pool](#1-train-a-factor-pool)
  - [2. Resume / incremental training](#2-resume--incremental-training)
  - [3. Export factor values](#3-export-factor-values)
- [Configuration and Hyperparameters](#configuration-and-hyperparameters)
- [Training Outputs](#training-outputs)
- [Acknowledgements](#acknowledgements)

---

## Motivation

Classical symbolic factor mining relies on manual design or reinforcement
learning with a single-point reward. Standard RL objectives tend to suffer from
**mode collapse**: the policy repeatedly returns one high-reward expression and
the resulting factor library lacks diversity, which is exactly what a
multi-factor portfolio needs. Conversely, purely correlation-based filters are
blind to the *semantic structure* of an expression and may retain many
near-duplicate formulas written in different ways.

This project addresses both issues:

1. **GFlowNets** replace the reward-maximizing objective with *reward-proportional
   sampling* `P(τ) ∝ R(τ)`, so the model explores a broad set of good
   expressions instead of committing to one mode.
2. A **structure-aware encoder** learns embeddings of expression graphs and a
   **self-supervised consistency loss** ties reward to structural semantics,
   promoting semantically meaningful and mutually diverse factors.

## Key Contributions

- A discrete GFlowNet environment over RPN factor expressions with strict
  *grammatical validity masks*, guaranteeing that every sampled expression is a
  well-formed tree.
- A **structure-aware RGCN encoder** that converts an RPN token sequence into a
  typed directed graph (6 edge relations) and pools it into a fixed-size
  embedding, used both for action selection and for structural rewards.
- A **triple reward** combining predictive power (`IC`), **novelty** (decorrelation
  with the existing pool), and **structural self-consistency (SSL)** via k-NN in
  embedding space.
- **Entropy-regularized Trajectory Balance** training on top of `torchgfn`, with
  learnable `logZ` and a scheduled decay of the auxiliary reward weights.
- A production-style training loop with TensorBoard logging, checkpointing
  (model + pool + optimizer + `logZ` + pool embeddings) and seamless
  **incremental training / resume**.

## Method Overview

### Factor Discovery as a GFlowNet

A factor is a symbolic expression built from price/volume features, arithmetic
operators and rolling-window operators, e.g.
`Sub(TsMean($close,16),Ref($close,4))`. Each expression is generated token by
token as a **trajectory** `τ = (s₀ → … → s_f)` in a discrete environment. The
GFlowNet forward policy `P_F(τ)` learns to sample trajectories with probability
proportional to the terminal reward:

```
P_F(τ)  ∝  R(τ)
```

so that high-quality expressions are more likely, yet the model still assigns
non-zero probability mass to a *diversity* of good expressions.

### State and Action Space

- **State**: a fixed-length integer sequence of length `MAX_EXPR_LENGTH = 30`,
  starting with `BEG_TOKEN` and padded with `-1`.
- **Action vocabulary** (74 actions in total):

  | Block | Count | Examples |
  |-------|------:|----------|
  | `BEG_TOKEN` | 1 | start-of-sequence marker |
  | Operators | 35 | `Abs, Inv, Sign, Log, SLog1p`, `Add, Sub, Mul, Div, Pow, Greater, Less`, `Ref, TsMean, TsStd, TsSkew, TsKurt, TsEMA …`, `TsCov, TsCorr` |
  | Features | 32 | `$open, $close, $high, $low, $volume, $vwap, $amount, $twap, $top_amount, $amplitude_*, $successive_*, $max_ret, …` |
  | Delta times | 5 | `4, 8, 16, 32, 64` (15-min bars → 1h, 2h, 4h, 8h, 16h) |
  | `SEP_TOKEN` | 1 | end-of-sequence / exit action |

- **Transition**: non-exit actions are appended to the current sequence; the
  `SEP` action jumps directly to the sink state `s_f`.
- **Validity masks**: at every state `GFNEnvCore.update_masks` calls
  `ExpressionBuilder.validate` for each candidate token, enforcing RPN arity and
  type constraints (e.g., a rolling operator requires a delta-time argument plus
  a featured operand). `SEP` is allowed on a complete, single-tree stack;
  at the length limit it is the only allowed action, even for incomplete
  expressions (which receive the minimum reward). The 30 slots include `BEG`,
  so sampling takes at most 29 appended tokens plus one exit step.
  Each non-initial, non-sink state has exactly one backward action: remove its
  last token. Initial and sink states have no backward actions.
  Illegal actions receive probability 0.

### Structure-Aware Encoder

`SequenceEncoder` provides three interchangeable backbones selected by
`--encoder_type`:

- `transformer` — `nn.TransformerEncoder` with sinusoidal positional encoding.
- `lstm` — 2-layer LSTM, last valid hidden state is used.
- `gnn` *(default)* — an **R-GCN** over the expression's syntax graph.

For the GNN path, `_build_graph_from_rpn` reconstructs the expression tree from
the RPN sequence and emits a typed directed graph with **6 edge relations**:

| Relation | Meaning |
|---------:|---------|
| 0 | unary operator → operand |
| 1 | commutative binary operator (Add, Mul) |
| 2 | non-commutative left operand |
| 3 | non-commutative right operand |
| 4 | feature operand of a rolling operator |
| 5 | time-series operand of a rolling operator |

Two layers of `RGCNConv` followed by `global_mean_pool` yield the expression
embedding. The same encoder is shared by the forward/backward policies, and the
embedding additionally drives the SSL reward and k-NN retrieval.

### Multi-Component Reward

For each terminal expression `expr` with structural embedding `e`, the pool
returns three reward terms:

- **IC reward** — Pearson correlation between the rolling-normalized factor
  values and the forward return target `Ref($close,-1)/$close − 1`.
- **Novelty reward** — `1 − max_j |corr(f, f_j)|` over the current pool, rewarding
  factors that are uncorrelated with existing ones.
- **SSL reward** — pick the `k` nearest pool factors in embedding space, compute
  a distance-weighted consistency loss `L = Σ w_i · MSE(f, f_i)` where
  `w = softmax(−‖e − e_i‖² / τ)`, and return `R_SSL = exp(−L)`. Structurally
  similar factors are thus encouraged to make *consistent* predictions, which
  forces the encoder to learn meaningful representations.

These are combined into the terminal reward:

```
R = R_IC + w_ssl · R_SSL + w_nov · R_novelty      (floored at exp(-10))
```

The auxiliary weights `w_ssl` and `w_nov` are decayed over training by a
`WeightScheduler` (linear / exponential / polynomial), so the policy gradually
focuses on the pure IC signal once a diverse pool has been seeded.

### Trajectory Balance with Entropy Regularization

Training uses the **Trajectory Balance** objective (`EntropyTBGFlowNet`):

```
L_TB = ( logZ + log P_F(τ) − log P_B(τ) − log R )²
```

where `logZ` is a learnable scalar (log partition function) and `P_B` is a
backward policy that supports arbitrary reversal of the generation order. An
**entropy regularization** term computed from the raw policy logits is subtracted
from the loss to sustain exploration:

```
L = mean( L_TB ) − mean( β · Σ_t H_t )
```

The forward head outputs `n_actions` logits; the backward head outputs
`n_actions − 1` (it never predicts the exit action).

### Mask Dropout and Grammatical Validity

To bias the policy toward *short, interpretable* expressions, `update_masks`
applies a length-dependent dropout on legal actions: with probability

```
p_mask = mask_dropout_prob · (current_length / MAX_EXPR_LENGTH)
```

all legal non-exit actions are disabled, leaving `SEP` as the only option. Longer
(likely redundant) expressions are therefore cut off earlier.

### Alpha Pool Management

`AlphaPoolGFN` keeps the top-`capacity` factors with a diversity constraint:

- an expression is admitted only if **both train IC > 0 and validation IC > 0**;
- its maximum absolute correlation with existing factors must not exceed
  `ic_mut_threshold` (default `0.3`);
- when the pool is full, a new factor replaces the current weakest factor only if
  its IC is higher.

Each stored factor keeps its expression tree, rolling-normalized values, IC,
pairwise correlation matrix and structural embedding. Factor values are
normalized with a fast cumsum-based **rolling z-score** (window = 200 bars) so
that IC is computed on stationary series.

## Project Structure

```
RL-mining/
├── train_gfn.py                 # Entry point: GFlowNet alpha-mining training loop
├── export_factors.py            # Entry point: export pool factors to a CSV
├── requirements.txt             # Python dependencies
├── config/
│   └── qcm_config/              # Baseline configs (FQF / IQN / QRDQN) for reference
├── data/                        # K-line CSV + training logs (NOT committed)
├── src/
│   ├── alpha_gfn/               # GFlowNet alpha-mining core
│   │   ├── config.py            # Hyperparameters + operator/feature/delta action space
│   │   ├── env/core.py          # GFNEnvCore: discrete env, masks, reward
│   │   ├── modules.py           # SequenceEncoder / GNNEncoder (R-GCN) / PositionalEncoding
│   │   ├── alpha_pool.py        # AlphaPoolGFN: pool mgmt, k-NN, SSL consistency
│   │   ├── gflownet.py          # EntropyTBGFlowNet: TB loss + entropy reg.
│   │   └── preprocessors.py     # IntegerPreprocessor (integer states for embeddings)
│   ├── alphagen/                # Symbolic expression system (from AlphaGen)
│   │   ├── data/                # Token / expression-tree / operators (tokens, tree, expression)
│   │   ├── models/alpha_pool.py # AlphaPoolBase: normalization, IC, ensemble
│   │   ├── rl/env/              # Reference gym environment + action↔token mapping
│   │   └── utils/               # correlation / ranking utilities
│   ├── alphagen_qlib/
│   │   └── stock_data.py        # StockData: CSV loader, feature indexing, slicing
│   ├── alphagen_generic/        # Generic feature/operator shorthands
│   └── gan/                     # Auxiliary GAN-based factor generation utilities
└── gfn/                         # Vendored torchgfn core (env, samplers, modules, GFlowNets)
```

The `gfn/` package is a local copy of the [torchgfn](https://github.com/GFNOrg/torchgfn)
building blocks used by the training loop (`Sampler`, `DiscretePolicyEstimator`,
`DiscreteEnv`, `TBGFlowNet`, …), kept in-tree for reproducibility.

## Installation

The code targets **Python ≥ 3.11**.

```bash
pip install -r requirements.txt
```

PyTorch must be installed separately to match your CUDA version, e.g.:

```bash
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121
```

`torch-geometric` is required for the default R-GCN encoder.

## Data Format

Training reads a single K-line CSV (default `data/kline_im.csv`). Each row is one
15-minute bar with at minimum the following columns:

| Column | Description |
|--------|-------------|
| `DateTime` | Bar timestamp (parseable by `pandas.to_datetime`) |
| `open, close, high, low` | OHLC prices |
| `vol` | Volume |
| `vwap` | Volume-weighted average price |
| `amount, twap, top_amount` | Turnover-derived features |
| `amplitude_max/mean/std`, `successive_*`, `*_ret*`, `*_kline_ratio`, `vwap_ema_*` | Engineered features (32 in total) |
| `pct` | Bar return (used by `export_factors.py`) |

The 32 model features and their CSV column mapping are defined in
[stock_data.py](file:///d:/Job-project/RL-mining/src/alphagen_qlib/stock_data.py)
(`FeatureType` / `_FEATURE_TO_COL`).

The dataset is split chronologically to avoid look-ahead leakage:

| Split | Period |
|-------|--------|
| Train | 2022-10-10 → 2024-12-31 |
| Validation | 2025-01-01 → 2025-07-31 |
| Test | 2025-08-01 → 2026-03-09 |

> Raw market data is **not** committed to the repository; place your own
> `data/kline_im.csv` before running. See [.gitignore](file:///d:/Job-project/RL-mining/.gitignore).

## Usage

### 1. Train a factor pool

```bash
python train_gfn.py \
    --seed 0 \
    --instrument im \
    --name run \
    --pool_capacity 400 \
    --n_episodes 10000 \
    --n_trajectories 32 \
    --encoder_type gnn \
    --entropy_coef 0.01 \
    --mask_dropout_prob 0.5 \
    --ssl_weight 1.0 \
    --nov_weight 0.3 \
    --weight_decay_type linear \
    --final_weight_ratio 0.0
```

`--encoder_type` selects `transformer`, `lstm` or `gnn`. Useful knobs:

- `--pool_capacity` — maximum number of factors kept in the pool.
- `--n_episodes` — number of sampled trajectories (episodes) to train for.
- `--n_trajectories` — batch size of trajectories sampled per episode.
- `--ssl_weight` / `--nov_weight` — initial weights of the SSL / novelty rewards.
- `--weight_decay_type` + `--final_weight_ratio` — decay schedule of those weights.
  Linear and quadratic polynomial schedules support a final ratio of zero.
  Exponential decay requires `0 < final_weight_ratio <= 1` (for example `0.01`).
  All schedules stop decaying at the configured final ratio.

### 2. Resume / incremental training

Pass a checkpoint directory (the latest `model*.pt` is picked automatically) or a
specific model file:

```bash
python train_gfn.py --resume data/gfn_logs/pool_400/run
```

On resume the loader restores the model, the backward head, the optimizer,
`logZ`, the factor pool **and** its structural embeddings. Hyperparameters that
are tightly coupled to the checkpoint (`encoder_type`, `pool_capacity`,
`entropy_coef`, `ssl_weight`, `nov_weight`, decay settings, …) are inherited from
the checkpoint unless you override them on the command line.

### 3. Export factor values

Write each pool factor's time series (plus the `pct` return column) to a CSV for
downstream combination / backtesting:

```bash
python export_factors.py \
    --pool data/gfn_logs/pool_400/run/pool.json \
    --csv data/kline_im.csv \
    --output data/gfn_logs/pool_400/run/pool_factors.csv
```

Factors that fall outside the data range are exported as `NaN` columns so the
output shape always matches the pool size.

## Configuration and Hyperparameters

Model / search hyperparameters live in
[config.py](file:///d:/Job-project/RL-mining/src/alpha_gfn/config.py):

| Constant | Value | Meaning |
|----------|-------|---------|
| `MAX_EXPR_LENGTH` | 30 | Maximum tokens per expression |
| `HIDDEN_DIM` | 128 | Encoder / embedding dimension |
| `NUM_ENCODER_LAYERS` | 2 | Encoder depth (Transformer / LSTM / R-GCN) |
| `NUM_HEADS` | 4 | Attention heads (Transformer) |
| `DROPOUT` | 0.1 | Dropout rate |
| `LEARNING_RATE` | 1e-4 | Adam learning rate |
| `OPERATORS` | 36 | Available operators |
| `FEATURES` | 32 | Available features |
| `DELTA_TIMES` | 4, 8, 16, 32, 64 | Rolling window lengths |

Command-line defaults (`train_gfn.py`):

| Argument | Default | Meaning |
|----------|--------:|---------|
| `--pool_capacity` | 10 | Max factors in the pool |
| `--n_episodes` | 256 | Training episodes |
| `--n_trajectories` | 32 | Trajectories per episode |
| `--encoder_type` | gnn | Encoder backbone |
| `--entropy_coef` | 0.01 | Entropy regularization coefficient |
| `--entropy_temperature` | 1.0 | Entropy temperature |
| `--mask_dropout_prob` | 0.5 | Max legal-action dropout probability |
| `--ssl_weight` | 0.5 | Initial SSL reward weight |
| `--nov_weight` | 0.5 | Initial novelty reward weight |
| `--weight_decay_type` | linear | Weight decay schedule |
| `--final_weight_ratio` | 0.0 | Final/initial weight ratio |
| `--update_freq` | 1 | Episodes per optimizer step |
| `--log_freq` | 100 | Episodes per logging round |
| `--resume` | None | Checkpoint dir/file to resume from |

## Training Outputs

Logs and checkpoints are written to
`data/gfn_logs/pool_{capacity}/{name}/`:

- `model.pt` — forward/backward weights, optimizer state, `logZ`, args and pool
  embeddings.
- `pool.json` — factor expressions (as strings), IC values, ensemble weights,
  best IC and the reward record.
- `pool_factors.csv` — exported factor time series (after running
  `export_factors.py`).
- TensorBoard `events.*` — pool size, best single-factor IC, validation/test
  ensemble IC, training loss, `logZ`, and the current SSL / novelty weights.

Launch TensorBoard with:

```bash
tensorboard --logdir data/gfn_logs
```

During training the console periodically prints every factor in the pool with its
train / validation / test IC so that quality can be monitored in real time.

## Acknowledgements

This project builds on and adapts several open-source works:

- **[AlphaGen](https://github.com/RL-MLDM/alphagen)** — the symbolic expression
  system, token/operator definitions and alpha-pool foundation.
- **[torchgfn](https://github.com/GFNOrg/torchgfn)** — GFlowNet environments,
  samplers, estimators and the Trajectory Balance objective.
- **[torch-geometric](https://github.com/pyg-team/pytorch_geometric)** — R-GCN
  layers used by the structure-aware encoder.

The work is intended for research and educational purposes.

### Regression checks

```bash
PYTHONPATH=src python -m pytest -q
```

The tests cover expression graph token IDs, parent masks, the length limit,
GNN sampling and gradients, entropy with/without stored logits, checkpoint
`logZ` persistence, and reward weight schedules. They use a fixed reward fixture
and do not require market data or run a full training experiment.
