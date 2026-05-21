# LeWorldModel — LunarLander Adaptation

## Project Goal

Adapt the **LeWorldModel (LeWM)** algorithm to the `LunarLander-v3` gymnasium environment.
LeWM is a Joint Embedding Predictive Architecture (JEPA) that learns a latent world model from
raw pixel observations end-to-end, without reconstruction losses, reward signals, or pre-trained
encoders. We will implement it from scratch, then compare it against standard baselines.

---

## The LeWorldModel Algorithm

**Paper:** *LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels*
(Maes, Le Lidec, Scieur, LeCun, Balestriero — arXiv:2603.19312)

### Core Idea

LeWM is a JEPA: instead of reconstructing pixels, it learns to predict future **latent embeddings**.
The key challenge with JEPAs is representation collapse (encoder maps everything to the same vector).
LeWM solves this with a single regularization term (SIGReg) that encourages embeddings to follow
an isotropic Gaussian distribution — no stop-gradient, no EMA, no frozen encoders.

### Architecture

Two learned components (~15M params total, trains on a single GPU):

**Encoder** (`enc_θ`) — ViT-tiny (~5M params)
- Input: raw pixel frame `o_t` → Output: latent embedding `z_t`
- Patch size 14, 12 layers, 3 attention heads, hidden dim 192
- Uses the `[CLS]` token of the last layer, followed by a 1-layer MLP + BatchNorm projection
- BatchNorm is required because the final ViT LayerNorm blocks the SIGReg gradient

**Predictor** (`pred_φ`) — Transformer (~10M params)
- Input: history of N latent embeddings + action `a_t` → Output: predicted next embedding `ẑ_{t+1}`
- 6 layers, 16 attention heads, 10% dropout
- Action conditioning via Adaptive Layer Normalization (AdaLN) at each layer, initialized to zero
- Followed by the same 1-layer MLP + BatchNorm projector as the encoder
- Uses causal masking (autoregressive, no peeking at future embeddings)

### Training Objective

```
L_LeWM = L_pred + λ · SIGReg(Z)
```

**Prediction loss** (MSE, teacher-forcing):
```
L_pred = || ẑ_{t+1} - z_{t+1} ||²₂
```
The encoder is incentivized to produce embeddings that the predictor can track.

**SIGReg** (anti-collapse regularizer):
- Projects the batch of embeddings `Z ∈ R^{N×B×d}` onto M random unit-norm directions
- Applies the Epps–Pulley univariate normality test to each 1D projection
- Averages the test statistics across projections
- By the Cramér–Wold theorem, matching all 1D marginals = matching the full joint Gaussian

```
SIGReg(Z) = (1/M) Σ_m T(Z u^(m))
```

Default hyperparameters: M=1024 projections, **λ=0.1** (the only hyperparameter that matters).
λ can be tuned via bisection search (log complexity), unlike PLDM which has 6 hyperparameters.

**No stop-gradient, no EMA, no auxiliary losses.** All gradients flow through all components.

### Training Data

Fully offline and reward-free: trajectories of `(o_{1:T}, a_{1:T})` raw pixel observations + actions.
Data can come from any behavior policy (random, expert, or mixed) as long as it covers the dynamics.

### Latent Planning (Inference)

At test time, planning is done via **Model Predictive Control (MPC)** with the
**Cross-Entropy Method (CEM)**:

1. Encode current observation: `z_1 = enc_θ(o_1)`
2. Encode goal observation: `z_g = enc_θ(o_g)`
3. Sample candidate action sequences, roll out latent states autoregressively via the predictor
4. Minimize terminal cost: `C(ẑ_H) = || ẑ_H - z_g ||²₂`
5. CEM iteratively refines the action distribution using top-k plans
6. Execute only the first K actions, then replan from the new observation

Planning horizon H trades off lookahead quality vs. error accumulation.
LeWM plans up to **48× faster** than foundation-model-based world models (DINO-WM).

---

## LunarLander-v3 Environment

**Observations** — 8-dimensional continuous vector:
| Index | Quantity |
|-------|----------|
| 0 | x position |
| 1 | y position |
| 2 | x velocity |
| 3 | y velocity |
| 4 | angle |
| 5 | angular velocity |
| 6 | left leg contact (bool) |
| 7 | right leg contact (bool) |

**Actions** — depends on variant:
- `LunarLander-v3` (discrete): 4 actions — do nothing, fire left engine, fire main engine, fire right engine
- `LunarLanderContinuous-v3` (continuous): 2-dimensional vector — main engine throttle `[-1, 1]`, lateral engine throttle `[-1, 1]`

**Reward:** computed each step as a delta of a shaping function plus fuel penalties and terminal bonuses.

Shaping function (computed from current state `s`):
```
shaping(s) = -100 * sqrt(x² + y²)        # penalize distance from landing pad
           - 100 * sqrt(vx² + vy²)        # penalize speed
           - 100 * |angle|                 # penalize tilt
           + 10 * leg_left                 # reward ground contact
           + 10 * leg_right
```

Step reward:
```
r_t = shaping(s_t) - shaping(s_{t-1})     # improvement in shaping
    - 0.30 * main_engine_fired             # fuel cost
    - 0.03 * side_engine_fired             # fuel cost
```

Terminal reward (replaces step reward on final step):
- Crash or out of bounds: **-100**
- Successful landing (lander comes to rest): **+100**

**Key implication:** the step reward depends on *both* `s_{t-1}` and `s_t`, not just the current state.
The reward predictor therefore needs `(prev_obs, obs, action)` as input.

**Approximate value ranges:**
- Typical step reward: `[-10, +5]`
- Terminal: `±100`
- Full episode total: typically `[-500, +300]` depending on policy quality

---

## Project TODO

Note: architectures will differ from the paper — LunarLander has different observation/action
characteristics than the original paper's environments.

- [x] Stage 1 — Encoder
- [x] Stage 2 — Predictor
- [x] Stage 3 — SIGReg + full training loop
- [x] Stage 4 — Latent planning (CEM + MPC)
- [x] Incorporate reward model into the world (encoder input = [obs, signed_log(reward)]; decoder output = [obs, reward_log])
- [ ] Closed-loop action execution:
        init world model NNs, and policy pi
        while True:
            data = collect(pi)
            wm = learnWM(data)
            pi = learnPolicy(wm)
- [ ] Baseline comparison

---

## Ablation Studies

- [ ] Latent dimension size

---

## Running Python Files

Use `uv run python3` to run Python files, e.g. `uv run python3 -m lewm.train_world_model`

---

## Code Conventions

- All imports at the top of the file, never inside functions
- Maximum line length: 100 characters
- Two empty lines between functions
- No empty lines inside functions
- No comments unless the reason is non-obvious
- Each function does exactly one thing; break down multi-purpose functions
- Use as little code as possible — prefer built-ins and library calls over manual loops
- Be precise and focused — no speculative abstractions
- **Test after every stage** before moving to the next
