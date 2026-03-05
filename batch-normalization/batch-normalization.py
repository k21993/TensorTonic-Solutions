import numpy as np

def batch_norm_forward(x, gamma, beta, eps=1e-5):
    """
    Forward-only BatchNorm for (N,D) or (N,C,H,W).
    """
    # Write code here
    #bn: norm across all axes except feature or channel.

    x = np.array(x)
    gamma = np.array(gamma)
    beta = np.array(beta)
    
    is_2d = len(x.shape) == 2

    if is_2d:
        x = x[:, :, None, None] #(N,D) -> (N, D, 1, 1)

    mu = np.mean(x, axis=(0, 2, 3), keepdims=True)
    var = np.var(x, axis=(0, 2, 3), keepdims=True)
    #standardoze -> scale and shift
    x = (x-mu)/(np.sqrt(var + eps))

    #gamma and beta are (D,) or (C,). expand dims to broadcast
    gamma = gamma.reshape(1,-1, 1, 1)
    beta = beta.reshape(1, -1, 1, 1)

    y = x*gamma + beta
    if is_2d:
        y = y.squeeze(-1).squeeze(-1)
    return y.tolist()

    