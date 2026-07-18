import numpy as np
def policy_gradient_loss(log_probs, rewards, gamma):
    """
    Compute REINFORCE policy gradient loss with mean-return baseline.
    """
    # Write code here
    G = [rewards[-1]]
    for r in rewards[-2::-1]:
        G.append(r + gamma*G[-1])
    G = G[::-1]
    
    G = np.asarray(G)
    log_probs = np.asarray(log_probs)
    G_mean = np.mean(G)

    adv = G - G_mean
    loss = -np.mean(adv * log_probs)

    return float(loss)

        
    