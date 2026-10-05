# MolRL-Folding

Reinforcement learning for **DNA origami staple design**: an agent learns to build the staple layout of a target nanostructure, and is rewarded by the robustness of the finished design.

<p align="center">
  <img src="docs/episode.svg" width="760" alt="Animation of one episode: staples are placed one by one on a routed scaffold, then the complete design is scored.">
</p>

<p align="center"><em>One episode (illustrative). 18 staples on a small L-shaped sheet; real designs use roughly 200.</em></p>

## The task in one paragraph

A DNA origami is built from one long **scaffold** strand (typically ~7,000 nucleotides, e.g. M13mp18) and many short **staples** (typically 30–40 nt) that bind the scaffold at chosen positions and pull distant regions together. Given a target shape and a routed scaffold, the staple layout decides whether the shape forms and how robust it is. Today this layout comes from rule-based design tools plus manual tweaks. Here, an agent places staples one at a time and receives a single reward once the full design is evaluated.

## Research question

> Can an RL agent learn staple-placement policies that produce more robust DNA origami designs than rule-based baselines, and do these policies generalize to target shapes unseen during training?

Sub-questions:

1. **Credit assignment.** With one terminal reward, can learned value estimates of *partial* designs guide staple placement?
2. **Sample efficiency.** Evaluating a design is expensive. How many evaluations are needed to beat the baselines?
3. **Proxy validity.** Do cheap robustness scores agree with higher-fidelity simulation and with published experimental results?
4. **Generalization.** Does a policy trained on some shapes transfer to others?

## Background in five lines

- A **helix** is a stretch of double-stranded DNA. A DNA origami sheet is several helices side by side.
- The **scaffold** snakes through all helices as a single strand, turning around at the ends.
- A **staple** binds the scaffold on one helix, crosses to a neighboring helix at a **crossover**, and binds the scaffold there too.
- Staples hold neighboring helices together, which gives the sheet its rigidity and its shape.
- In the lab all staples are mixed with the scaffold at once and annealed.

## MDP formulation (draft)

| Component | Definition |
|---|---|
| **Fixed per episode** | Target shape (helix grid and lengths), routed scaffold and its base sequence |
| **State** | Target shape and coverage map, two boolean channels on an `h_max × l_max` canvas (1 cell = 1 helix × 1 nt). The scaffold routing is implied by the shape |
| **Action** | Place one two-domain staple: `MultiDiscrete [row, start, crossover, end, side]`, where `side` selects the neighbouring helix (below or above). Illegal actions are overlaps, lengths out of range, and impossible crossovers |
| **Transition** | Deterministic: the chosen staple is added to the plan |
| **Termination** | The scaffold is fully covered |
| **Truncation** | Step limit (e.g. twice the expected staple count). Handled as truncation, not termination, when bootstrapping values |
| **Reward** | Sparse, terminal: robustness score of the complete design (next section). No intermediate reward in v1 |

The placement order is a *construction order* for the plan, not a physical order. In the tube all staples are present together. An episode therefore produces a complete design, not a sequence of physical steps.

## Environment (v0.1)

- **Canvas:** `h_max = 16` helices by `l_max = 128` nt. Shapes smaller than the canvas are padded, and cells outside the shape are never legal.
- **Staple length:** `|x_c − x_0| + |x_e − x_c| + 2`, bounded by `l_staple_min` and `l_staple_max` (20 to 60 nt). Optional constraints: minimum domain length (`l_domain_min`) and crossover spacing (`crossover_period`). Both are off by default.
- **Shapes:** generated from a seed. Each helix is one interval, and consecutive helices share the end where the scaffold turns, so a serpentine routing always exists. Shapes are identified by a hash (`shape_id`) so that train/test splits can be made by shape.
- **End of episode:** the shape is fully covered, or no legal staple remains (dead end). Truncation at `max_steps_factor × (cells // l_staple_min)` steps.
- **Reward:** 0 until the end, then a pluggable `reward_fn(shape, covered, staples, cfg)`. The default is the covered fraction, a placeholder for the robustness score.

## Measuring robustness (draft)

There is no single standard metric, so the score is built from several levels of fidelity, trading accuracy against cost.

| Level | Method | What it measures | Cost | Role |
|---|---|---|---|---|
| 1. Thermodynamic | Nearest-neighbor models (e.g. NUPACK) | Stability of each staple–scaffold duplex, with emphasis on the weakest link; mis-hybridization between staples | milliseconds | Training reward |
| 2. Coarse mechanical | Tools such as CanDo or mrDNA | Deviation from the target shape, flexibility, and **defect tolerance** (remove random staples, re-evaluate) | seconds to minutes | Training reward or selective evaluation |
| 3. Coarse-grained dynamics | oxDNA | Stability of the simulated structure | hours | Validation of the best designs only |

These scores are **proxies** for experimental yield. Checking how well they track published results is part of the research.

## Methods and baselines

- **Random valid policy**: samples a feasible crossover uniformly, then random domain lengths and directions (implemented).
- **Rule-based heuristic**: fixed staple length and regular crossover spacing, following standard origami design conventions. A first greedy sweep (cover the first free cell with a staple of about 40 nt) is implemented; a convention-based version is still to do.
- **PPO with action masking**: policy-gradient baseline.
- **Discrete value-based agent with action masking** (e.g. Double DQN). PPO also handles discrete actions, so this comparison is policy-gradient versus value-based.

Metrics: robustness score, defect tolerance, shape deviation, number of evaluations needed to reach a target score, and performance on held-out shapes.

## Scope and assumptions

- **Scaffold routing is fixed** in v1. Learning the routing is a possible extension.
- Targets start as **2D sheets** built from parallel helices. More complex shapes come later.

## Open design decisions

- Action parameterization: one flat discrete index, or an autoregressive choice of start, end, and crossover.
- Reward composition and weighting between levels 1 and 2.
- Whether to allow an explicit stop action that can leave regions uncovered.
- Reward for a dead end (currently the covered fraction, like any other ending).
- Values of `l_domain_min` and `crossover_period` (both unconstrained for now).
- Benchmark shapes and the scaffold sequence.
- Evaluation budget per training run.

## Repository structure

```
src/molrl_folding/
    config.py      EnvConfig: canvas size, staple constraints, shape generator settings
    shapes.py      seeded shape generator, shape hash
    design.py      Staple, legality checks, dead-end detection
    rewards.py     reward functions (placeholder: covered fraction)
    env.py         FoldingEnv (Gymnasium)
    baselines.py   random valid policy, greedy sweep
    render.py      top-view drawing of a design
scripts/           evaluate_baselines.py, render_episode.py
tests/             pytest suite
docs/              episode.svg
```

## Quick start

```bash
pip install -e ".[dev]"
pytest
python scripts/evaluate_baselines.py --episodes 100
python scripts/render_episode.py --policy greedy --seed 3 --out outputs/episode.png
```

```python
import gymnasium as gym
import molrl_folding

env = gym.make("MolRLFolding-v0")
obs, info = env.reset(seed=0)   # obs: float32 array of shape (2, 16, 128)
```

## Technologies

- Python 3.10+, NumPy, matplotlib (rendering)
- Gymnasium environment (`MolRLFolding-v0`)
- RL library to be decided
- Thermodynamic and mechanical evaluation backends to be decided; oxDNA for validation

## Roadmap

- [ ] Define the benchmark: small 2D sheet shapes first, then larger ones
- [x] Implement the environment: state, termination and truncation
- [ ] Add action masking for the policy
- [ ] Implement the level-1 (thermodynamic) reward
- [x] Implement first baselines: random valid policy and greedy sweep
- [ ] Add a rule-based baseline following standard design conventions
- [ ] Train PPO with action masking
- [ ] Train a discrete value-based agent
- [ ] Add the level-2 (mechanical) reward and defect-tolerance test
- [ ] Evaluate generalization to held-out shapes
- [ ] Validate the best designs with oxDNA

## Results

Coming soon.