import gymnasium as gym
import numpy as np
import argparse
import os
OBSERVATION_SIZE = 8
ACTION_SIZE = 4
PARAM_SIZE = OBSERVATION_SIZE * ACTION_SIZE + ACTION_SIZE  

def policy_action(params, observation):
    """Compute action using a linear policy."""
    W = params[:OBSERVATION_SIZE * ACTION_SIZE].reshape(OBSERVATION_SIZE, ACTION_SIZE)
    b = params[OBSERVATION_SIZE * ACTION_SIZE:].reshape(ACTION_SIZE)
    logits = np.dot(observation, W) + b
    return np.argmax(logits)

# Evaluate a single episode
def evaluate_episode(params, render=False):
    """Evaluate a policy for a single episode."""
    env = gym.make("LunarLander-v3", render_mode="human" if render else "rgb_array")
    observation, _ = env.reset()
    episode_reward = 0.0
    done = False
    while not done:
        action = policy_action(params, observation)
        observation, reward, terminated, truncated, _ = env.step(action)
        episode_reward += reward
        done = terminated or truncated
    env.close()
    return episode_reward

# Sequential evaluation of multiple episodes
def evaluate_policy(params, episodes=35, render=False):
    """Evaluate a policy over multiple episodes."""
    rewards = [evaluate_episode(params, render) for _ in range(episodes)]
    return np.mean(rewards)

# Particle Swarm Optimization (PSO) with Dynamic Parameters
def pso_optimize(population_size=80, generations=300, episodes=35):
    """Particle Swarm Optimization with adaptive learning rates."""
    swarm = np.random.randn(population_size, PARAM_SIZE) * 0.1
    velocity = np.random.randn(population_size, PARAM_SIZE) * 0.1
    personal_best = swarm.copy()
    personal_best_scores = np.array([evaluate_policy(p, episodes=episodes) for p in swarm])
    global_best = personal_best[np.argmax(personal_best_scores)]
    global_best_score = max(personal_best_scores)

    for gen in range(generations):
        w = 0.9 - (0.5 * gen / generations)  
        c1 = 1.5 + (0.5 * gen / generations)  # Increase cognitive component
        c2 = 2.5 - (0.5 * gen / generations)  # Decrease social component

        for i in range(population_size):
            r1, r2 = np.random.rand(PARAM_SIZE), np.random.rand(PARAM_SIZE)
            velocity[i] = (
                w * velocity[i]
                + c1 * r1 * (personal_best[i] - swarm[i])
                + c2 * r2 * (global_best - swarm[i])
                + np.random.randn(PARAM_SIZE) * 0.1  # Increased noise for better exploration
            )
            swarm[i] += velocity[i]
            score = evaluate_policy(swarm[i], episodes=episodes)
            if score > personal_best_scores[i]:
                personal_best[i] = swarm[i]
                personal_best_scores[i] = score
            if score > global_best_score:
                global_best = swarm[i]
                global_best_score = score

        print(f"Generation {gen+1}: Best Reward = {global_best_score:.2f}")
    return global_best

# Tabu Search refinement
def tabu_search(best_params, max_iterations=100, tabu_size=50, episodes=35):
    """Refine the best solution found by PSO using Tabu Search."""
    tabu_list = []
    best_solution = best_params.copy()
    best_score = evaluate_policy(best_solution, episodes=episodes)

    for _ in range(max_iterations):
        candidates = [
            best_solution + np.random.normal(0, 0.1, size=PARAM_SIZE) for _ in range(5)
        ]
        candidate_scores = [evaluate_policy(c, episodes=episodes) for c in candidates]

        sorted_candidates = sorted(zip(candidates, candidate_scores), key=lambda x: -x[1])

        for candidate, score in sorted_candidates:
            if score > best_score:
                best_solution = candidate
                best_score = score
                tabu_list.append(tuple(candidate))
                if len(tabu_list) > tabu_size:
                    tabu_list.pop(0)

    return best_solution

# Hill Climbing refinement
def hill_climbing(best_params, max_iterations=100, step_size=0.05, episodes=35):
    """Further refine the solution using Hill Climbing."""
    best_solution = best_params.copy()
    best_score = evaluate_policy(best_solution, episodes=episodes)

    for _ in range(max_iterations):
        candidate = best_solution + np.random.uniform(-step_size, step_size, size=PARAM_SIZE)
        candidate_score = evaluate_policy(candidate, episodes=episodes)
        
        if candidate_score > best_score:
            best_solution = candidate
            best_score = candidate_score
    
    return best_solution

# Train and save best policy
def train_and_save(filename, population_size=80, generations=300, episodes=35):
    """Train the policy using PSO + Tabu Search + Hill Climbing and save it."""
    best_pso_params = pso_optimize(population_size=population_size, generations=generations, episodes=episodes)
    refined_params = tabu_search(best_pso_params, max_iterations=100, episodes=episodes)
    final_params = hill_climbing(refined_params, max_iterations=100, episodes=episodes)
    np.save(filename, final_params)
    print(f"Best policy saved to {filename}")
    return final_params

# Load best policy
def load_policy(filename):
    """Load a trained policy."""
    if not os.path.exists(filename):
        print(f"File {filename} does not exist.")
        return None
    return np.load(filename)

# Play using the best policy
def play_policy(best_params, episodes=35):
    """Evaluate and render the best policy."""
    test_reward = evaluate_policy(best_params, episodes=episodes, render=True)
    print(f"Average reward of best policy: {test_reward:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or play best policy for Lunar Lander using PSO with Tabu Search and Hill Climbing.")
    parser.add_argument("--train", action="store_true", help="Train the policy using PSO and save it.")
    parser.add_argument("--play", action="store_true", help="Load the best policy and play.")
    parser.add_argument("--filename", type=str, default="best_policy.npy", help="Filename to save/load the best policy.")
    parser.add_argument("--population_size", type=int, default=80, help="Population size for PSO.")
    parser.add_argument("--generations", type=int, default=300, help="Number of generations for PSO.")
    parser.add_argument("--episodes", type=int, default=35, help="Number of episodes for evaluation.")
    args = parser.parse_args()
    
    if args.train:
        best_params = train_and_save(args.filename, population_size=args.population_size, generations=args.generations, episodes=args.episodes)
    elif args.play:
        best_params = load_policy(args.filename)
        if best_params is not None:
            play_policy(best_params, episodes=args.episodes)
    else:
        print("Please specify --train to train and save a policy, or --play to load and play the best policy.")