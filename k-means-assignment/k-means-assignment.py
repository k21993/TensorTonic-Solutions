import numpy as np
def k_means_assignment(points, centroids):
    """
    Assign each point to the nearest centroid.
    """
    # Write code here
    points = np.array(points) #(N, d)
    centroids = np.array(centroids) #(C, d)
    #we need (n, c, d) since we need dist for each centroid (C)from each point (N)
    diff = points[:, None, :] - centroids[None, :, :] # (N, C, d)
    dist = np.linalg.norm(diff, axis=-1) #(N, C)
    nearest_centroid = np.argmin(dist, axis=-1) #(N)

    return nearest_centroid.tolist()
    
    