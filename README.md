# Learning LeWorldModel

An implementation of **LeWorldModel (LeWM)** adapted to the `LunarLander-v3` environment.

LeWM is a Joint Embedding Predictive Architecture (JEPA) that learns a latent world model
directly from raw pixel observations — no reconstruction losses, no reward signals, no pre-trained
encoders. It was introduced in:

> *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels*
> Maes, Le Lidec, Scieur, LeCun, Balestriero — arXiv:2603.19312

## What it does

The model learns two things from offline pixel trajectories:

- **Encoder** — maps raw frames into a compact latent representation
- **Predictor** — models environment dynamics by predicting the next latent state given the current one and an action

Training uses only two loss terms: an MSE prediction loss and a SIGReg regularizer that prevents
representation collapse by encouraging latent embeddings to follow an isotropic Gaussian distribution.

At inference time, the model plans action sequences in latent space using the Cross-Entropy Method (CEM)
inside a Model Predictive Control (MPC) loop.

## Project stages

1. Encoder
2. Predictor
3. SIGReg + training loop
4. Latent planning (CEM + MPC)
5. Closed-loop action execution
6. Comparison with baseline algorithms (A2C, PPO)

## Setup

```bash
uv sync
```

## Run baseline

```bash
uv run baseline_algorithms/run_baseline_example.py
```
