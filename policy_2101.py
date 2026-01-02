import numpy as np

def policy_action(params, observation):
    expected_size = 32 + 4
    if isinstance(params, np.ndarray) and params.size == 1:
        params = params.item()
    if params.size != expected_size:
        raise ValueError(f" ERROR: Expected {expected_size} parameters, but got {params.size}.")

    W = params[:32].reshape(8, 4)
    b = params[32:]

    observation = np.clip(observation, -1, 1)

    logits = np.dot(observation, W) + b

    max_logit = np.max(logits)
    exp_logits = np.exp(logits - max_logit)
    probabilities = exp_logits / np.sum(exp_logits)

    action = np.random.choice(4, p=probabilities)

    return action
