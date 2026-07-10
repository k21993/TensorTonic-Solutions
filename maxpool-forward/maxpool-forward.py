import numpy as np

def maxpool_forward(X, pool_size, stride):
    """
    Compute the forward pass of 2D max pooling.
    """
    # Write code here
    #out_dim = [(d - p)/s] + 1
    X = np.array(X)
    h, w = X.shape
    h_out = (h - pool_size) // stride + 1
    w_out = (w - pool_size) // stride + 1
    out = np.zeros((h_out, w_out))

    for i in range(h_out):
        for j in range(w_out):
            row_start = i*stride
            col_start = j*stride
            row_end = row_start + pool_size
            col_end = col_start + pool_size

            out[i][j] = np.max(X[row_start: row_end, col_start: col_end])


    return out.tolist()
            
    