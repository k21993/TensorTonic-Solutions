import numpy as np

def epsilon_greedy(q_values, epsilon, rng=None):
    """
    Returns: action index (int)
    """
    # Q: (num_actions, )
    if not rng:
        rng = np.random.default_rng()

    if rng.random() < epsilon:
        action = rng.integers(0, len(q_values))
    else:
        action = np.argmax(q_values)

    return action
        
