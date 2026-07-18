import numpy as np

def bernoulli_pmf_and_moments(x, p):
    """
    Compute Bernoulli PMF and distribution moments.
    """
    #mean=p, std = p(1-p)
    mu = p
    std = p*(1-p)
    pmf = np.asarray([p if i==1 else 1-p for i in x ] )

    return pmf, mu, std
    
    