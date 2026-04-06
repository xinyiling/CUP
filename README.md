# CUP: Conversation Uncertainty-Aware Planning

CUP is a framework for goal-oriented conversational systems that uses uncertainty as a planning signal for multi-turn decision making. It integrates language models with Monte Carlo Tree Search to balance information acquisition and target commitment.

## Overview

The system maintains a belief state over candidate items and uses Expected Information Gain (EIG) to guide action selection. At each turn, an LLM proposes feasible actions, MCTS evaluates their long-term impact, and the selected action is verbalized into natural language.

We evaluate on four conversational benchmarks — [Inspired](https://github.com/sweetpeach/Inspired), [Beauty, Fashion, and Home](https://github.com/jeon185/LaViC).

**Workflow**:

1. **Belief & Uncertainty Modeling** — Maintains a probability distribution over candidates using SBERT similarity, quantifies uncertainty via entropy, and computes Expected Information Gain for candidate actions (`belief.py`, `similarity.py`)

2. **Uncertainty-Guided Planning** — An LLM proposes feasible actions grounded against candidates distributions, then MCTS with EIG-based priors evaluates their long-term impact over simulated trajectories (`action_proposer.py`, `mcts.py`)

3. **Language-Grounded Action Execution** — The selected action is verbalized by the LLM, presented to the user simulator, and the belief state is updated via Bayesian multiplicative updates (`environment.py`, `simulator.py`)

## Usage

Run CUP on each dataset:
```bash
python run.py --dataset inspired
python run.py --dataset lavic --category all_beauty
python run.py --dataset lavic --category amazon_fashion
python run.py --dataset lavic --category amazon_home
```

## Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--max-turns` | 5 | Maximum conversation turns (T) |
| `--system-model` | Llama-3.1-8B | LLM for action proposal and verbalization |
| `--sim-model` | Llama-3.2-3B | LLM for user simulation |
| `--entropy-threshold` | 0.5 | Normalized entropy ratio for commitment |
| `--belief-threshold` | 0.8 | Max belief probability for commitment |
| `--bayesian-alpha` | 1.0 | Exponent for Bayesian belief update |
| `--num-simulations` | 50 | MCTS search budget |

## Requirements

Python 3.10+, PyTorch, Transformers, sentence-transformers, tqdm, numpy.
