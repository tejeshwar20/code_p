# Lunar Lander Policy Optimization

This project implements an optimization-based controller for the LunarLander environment using reinforcement‐learning–inspired black-box search techniques. The agent policy is represented as a linear mapping from observations to action logits and is optimized using:

* Particle Swarm Optimization (PSO)
* Tabu Search refinement
* Hill Climbing fine-tuning

The implementation uses **Gymnasium** for environment simulation and **NumPy** for numerical computation.

---

## 1. Environment

The project uses the `LunarLander-v3` environment from Gymnasium.
The observation space has 8 dimensions and the action space has 4 discrete actions.

---

## 2. Policy Representation

The policy is a linear model:

```
action = argmax( observation · W + b )
```

where

* `W` is an 8 × 4 weight matrix
* `b` is a 4-dimensional bias vector
* parameter vector size = 32 + 4 = 36

---

## 3. Core Components

### Policy Evaluation

* single episode evaluation
* multi-episode averaged evaluation

### Optimization Algorithms

1. **Particle Swarm Optimization**
2. **Tabu Search** refinement of PSO output
3. **Hill Climbing** final local search

### Additional Features

* Parameter saving and loading (`.npy`)
* Render mode for visual evaluation
* Command-line interface for training and playing

---

## 4. Requirements

Install required dependencies:

```bash
pip install gymnasium
pip install gymnasium[box2d]
pip install numpy
```

Note: Box2D support must be installed for LunarLander.

---

## 5. How to Run

### Train Policy

This trains using PSO → Tabu Search → Hill Climbing and saves best policy.

```bash
python main.py --train
```

(optional arguments)

```
--population_size
--generations
--episodes
--filename
```

Example:

```bash
python main.py --train --population_size 60 --generations 200 --filename best.npy
```

---

### Play / Render Best Policy

```bash
python main.py --play --filename best_policy.npy
```

This loads parameters and renders multiple episodes.

---

## 6. Files Produced

| File              | Description                   |
| ----------------- | ----------------------------- |
| `best_policy.npy` | Saved optimized weight vector |

---

## 7. Project Structure

```
policy_action()      Linear policy
evaluate_episode()   Single episode evaluation
evaluate_policy()    Multi-episode evaluation
pso_optimize()       Global optimization
tabu_search()        Local refinement
hill_climbing()      Final tuning
train_and_save()     Full pipeline and save
load_policy()        Load saved model
play_policy()        Render policy in environment
```
