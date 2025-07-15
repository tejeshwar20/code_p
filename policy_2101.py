import numpy as np

def policy_action(params, observation):
    """Selects an action using a softmax policy based on a linear model."""

    # Ensure parameter size is correct (32 weights + 4 biases)
    expected_size = 32 + 4
    if isinstance(params, np.ndarray) and params.size == 1:
        params = params.item()  # Extract stored object if necessary
    if params.size != expected_size:
        raise ValueError(f" ERROR: Expected {expected_size} parameters, but got {params.size}.")

    # Reshape: First 32 elements → weights (8x4), Last 4 elements → biases
    W = params[:32].reshape(8, 4)
    b = params[32:]

    # Clip observation values to [-1, 1] for stability
    observation = np.clip(observation, -1, 1)

    logits = np.dot(observation, W) + b

    max_logit = np.max(logits)
    exp_logits = np.exp(logits - max_logit) 
    probabilities = exp_logits / np.sum(exp_logits)

    action = np.random.choice(4, p=probabilities)

    return action