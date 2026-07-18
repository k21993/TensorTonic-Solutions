def gae(rewards, values, gamma, lam):
    """
    Compute Generalized Advantage Estimation.
    """
    # delta_t = rt + gamma*(V(st+1) - V(st))
    deltas = []
    for idx in range(len(values) - 1):
        deltas.append(rewards[idx] + gamma*values[idx+1] - values[idx])
    # deltas.append(rewards[-1] - gamma*values[-1])

    #At = delta_t + gamma*lambda*At+1
    adv = [deltas[-1]]
    for d in deltas[-2::-1]:
        adv.append(d + gamma*lam*adv[-1])
    adv = adv[::-1]
    return adv
    