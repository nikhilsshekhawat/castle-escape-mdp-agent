# Escape the Castle – Reinforcement Learning Agent

An autonomous **Reinforcement Learning (RL) agent** trained to navigate and survive in a stochastic 5×5 gridworld full of hidden guards.  
The environment is modeled as a **Markov Decision Process (MDP)**, and the agent learns optimal behavior through **Model-Based estimation** and **Q-learning**.

---

## Project Summary

This project implements an intelligent agent capable of learning to **escape a guarded castle** by exploring, hiding, and fighting through probabilistic interactions.  
The problem combines **uncertainty, sequential decision-making, and delayed rewards**, making it an ideal testbed for reinforcement learning algorithms.

---

## Core Components

| File | Description |
|------|--------------|
| `mdp_gym.py` | Defines the MDP environment and transition dynamics |
| `vis_gym.py` | PyGame-based visualizer for the agent’s interactions |
| `reconstruct_MDP.py` | Model-based estimation of guard victory probabilities |
| `Q_learning.py` | Implementation of the Q-learning algorithm |
| `MFMC.py` | Integrates components and saves the trained Q-table |
| `Q_table.pickle` | Serialized learned policy (state–action values) |

---

## Reinforcement Learning Framework

### State Space
A state `s = (x, y, health, guard)` where:
- `(x, y)` ∈ grid positions `[0–4]`
- `health` ∈ {Full, Injured, Critical}
- `guard` ∈ {G1, G2, G3, G4, None}

### Action Space
`A = {UP, DOWN, LEFT, RIGHT, HIDE, FIGHT}`

### Rewards

| Event | Reward |
|--------|---------|
| Reach Goal | +10,000 |
| Win Fight | +10 |
| Lose Fight | -1,000 |
| Defeat (Critical loss) | -1,000 |

---

## Q-learning Algorithm

For every transition `(s, a, r, s')`, the Q-table is updated as:

\[
\eta = \frac{1}{1 + N(s,a)}
\]

\[
Q_{new}(s, a) = (1 - \eta) Q(s, a) + \eta \Big[r + \gamma \max_{a'} Q(s', a')\Big]
\]

Where:
- `γ` is the discount factor  
- `η` decays as updates accumulate  
- `N(s,a)` tracks the number of times a state–action pair is updated  

### Exploration Strategy

The epsilon value decays after each episode to balance exploration and exploitation.
