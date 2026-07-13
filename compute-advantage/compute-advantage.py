import numpy as np

def compute_advantage(states, rewards, V, gamma):
    """
    Returns: A (NumPy array of advantages)
    """
    # Gt = rt + gamma*Gt+1
    Gt = [rewards[-1]]
    for r in rewards[-2::-1]:
        Gt.append(r + gamma*Gt[-1])

    Gt = Gt[::-1]
    At = np.array(Gt) - np.array(V)
    return At
    
