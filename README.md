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
python run.py --dataset ${dataset} --system-model ${model}
```

<code>${dataset}</code> specifies the dataset to be evaluated on, selected from `inspired` / `beauty` / `fashion` / `home`

<code>${model}</code> specifies the backbone system model from huggingface.


## Key Parameters

| Parameter | Description |
|-----------|-------------|
| `--dataset` | `inspired` / `beauty` / `fashion` / `home` |
| `--T` | Maximum conversation turns (T) |
| `--system-model` | Backbone language model for action proposal, verbalization, and refined commitment |
| `--sim-model` | Separate LLM that simulates user response |
| `--model-device` | Device for all models |
| `--epsilon` | Normalized entropy threshold for commitment trigger |
| `--theta` | Max belief probability threshold for commitment |
| `--K` | Search budget: number of MCTS simulations per turn |

## Requirements

Python 3.10+, PyTorch, Transformers, sentence-transformers, tqdm, numpy.


## Citation
```bibtex
@article{ling2026uncertainty,
  title={Uncertainty as a Planning Signal: Multi-Turn Decision Making for Goal-Oriented Conversation},
  author={Ling, Xinyi and Liu, Ye and Averly, Reza and Ning, Xia},
  journal={arXiv preprint arXiv:2604.03924},
  year={2026}
}
```