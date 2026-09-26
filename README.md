# MolRL-Folding

MolRL-Folding is a reinforcement learning project focused on DNA nanostructure self-assembly.  
The core idea is to model DNA staple placement as a **Markov decision process (MDP)** so an agent can learn folding policies that minimize thermodynamic free energy.

## Motivation

DNA nanotechnology can encode highly structured assemblies, but finding efficient folding pathways remains difficult because each strand addition can alter the global energy landscape.  
This project explores whether reinforcement learning can discover better assembly sequences than hand-crafted heuristics.

## Project Overview

The folding process is represented as a sequential decision problem:

- **State**: current partial structure, available staple strands, and thermodynamic context
- **Action**: choose which staple strand to add next (discrete action space)
- **Transition**: update the structure after staple addition and evaluate resulting state
- **Reward**: encourage moves that reduce free energy and discourage destabilizing additions
- **Objective**: learn a policy that reaches stable target structures through low-energy trajectories

## Methods

This repository is designed to support and compare:

- **Discrete reinforcement learning** methods for staple selection
- **Proximal Policy Optimization (PPO)** as a policy-gradient baseline

The long-term goal is to benchmark which learning setup best balances stability, convergence speed, and final structure quality.

## MDP Formulation (Planned)

- **Observation space**
  - Encoded representation of partial DNA nanostructure
  - Thermodynamic descriptors (e.g., free-energy-related features)
  - Remaining candidate staples and placement constraints
- **Action space**
  - Select one valid staple placement at each decision step
- **Reward design**
  - Positive reward for favorable free-energy reduction
  - Penalty for invalid or destabilizing additions
  - Terminal bonus for reaching a stable folded target
- **Episode termination**
  - Target structure achieved
  - No valid actions remain
  - Maximum assembly steps reached

## Technologies

- Python
- Reinforcement learning tooling (planned)
- Numerical simulation stack (planned)

## Roadmap

- [ ] Formalize environment and state encoding
- [ ] Implement DNA folding simulator interface
- [ ] Train baseline discrete RL agent
- [ ] Train PPO agent
- [ ] Compare policies on energy minimization and folding success

## Results

Coming soon.
