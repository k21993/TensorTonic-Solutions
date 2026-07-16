import numpy as np

def value_iteration_step(values, transitions, rewards, gamma):
    """
    Perform one step of value iteration and return updated values.
    """
    # Write code here
    #values: [num_states, ]
    # rewards: [num_states, num_actions]
    # transitions: [num_states, num_actions, num_states]
    values, transitions, rewards = np.array(values), np.array(transitions), np.array(rewards)
    out = np.zeros_like(values)
    Q_sa = rewards + gamma* np.sum(transitions * values, axis=-1)
    out = np.max(Q_sa, axis=-1)

    return out.tolist()
    
    