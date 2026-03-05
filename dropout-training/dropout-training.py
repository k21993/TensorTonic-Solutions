import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code here
    x = np.array(x)
    rng = np.random if rng is None else rng

    probs = rng.random(x.shape)
    mask = probs < (1-p)
    dropout_pattern = mask/(1-p)
    x = x * dropout_pattern

    return x, dropout_pattern
    