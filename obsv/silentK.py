import numpy as np
from obsv.MDAVwMSE import MDAVwMSE

def silentK(X:np.ndarray,k):
    Xout = np.zeros(X.shape)
    # Create Clusters
    cluster_mapping, num_clusters, num_dp_per_cluster = MDAVwMSE(X, k)

    for cluster_id in range(num_clusters):
        # Find ids in cluster
        cluster_dp_ids = np.nonzero(cluster_mapping==cluster_id)[0]
        # Replace points by mean
        Xout[cluster_dp_ids] = np.mean(X[cluster_dp_ids], axis=0)

    return Xout

if __name__ == '__main__':
    print('Starting SILENTK')
    import scipy.io
    np.random.seed(0)
    X:np.ndarray = scipy.io.loadmat('source/SrcSoria/CASCrefmicrodataOriginal.mat')['CASCrefmicrodataOriginal']
    X = X[:10]
    print(X)
    r = silentK(X,3,3)
    print(r)