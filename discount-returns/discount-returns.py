def discount_returns(rewards, gamma):
    """
    Compute the discounted return at every timestep.
    """
    # rewards: (N,), returns: (N, ) both are 1 per time step
    Gt1 = 0
    G = []
    for r in rewards[::-1]:
        Gt = r + gamma*Gt1
        G.insert(0, Gt)
        Gt1 = Gt

    return G
    