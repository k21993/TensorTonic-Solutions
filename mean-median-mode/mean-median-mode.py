import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    # Write code here
    n = len(x)
    mean = np.sum(x)/n
    x = sorted(x)
    median = x[n//2] if n%2!=0 else 0.5* (x[n//2-1]+  x[n//2])
    freqs = Counter(x)
    mode = freqs.most_common(1)[0][0]

    return mean, median, mode