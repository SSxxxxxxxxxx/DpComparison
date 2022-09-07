import numpy as np
from obsv.MDAVwMSE import MDAVwMSE
from obsv.boundSampledElements import boundSampledElements

def soria(X:np.ndarray,epsilon,k):
    Xout = np.zeros(X.shape)
    # Calculating Sensitivity
    DeltaAttribute = 1.5 * np.abs(np.max(X, axis=0) - np.min(X, axis=0))
    MappingXtoXm,kappa,nQAtQ = MDAVwMSE(X,k)

    for IndexCluster in range(kappa):
        ClusterRecordsIDs = np.nonzero(MappingXtoXm==IndexCluster)[0]
        Cluster = X[ClusterRecordsIDs]
        # Compute Centroid
        Centroid = np.mean(Cluster,axis=0)
        #Add Noise
        Xout[ClusterRecordsIDs] = Centroid+np.random.laplace(0, np.sum(DeltaAttribute)/(k*epsilon),X.shape[1])
    return boundSampledElements(Xout,X)

def soriaByNumCluster(X:np.ndarray,eps,num_clusters):
    return soria(X,eps,int(X.shape[0]/num_clusters))

if __name__ == '__main__':
    print('Starting SORIA')
    '''import scipy.io
    X:np.ndarray = scipy.io.loadmat('source/SrcSoria/CASCrefmicrodataOriginal.mat')['CASCrefmicrodataOriginal']
    r = soria(X,3,100)'''
    import numpy as np
    import os
    print(os.listdir('.'))
    data = np.load('evaluatingDPML/dataset/housing_features.p',allow_pickle=True)
    assert data.shape == np.unique(data, axis=1).shape

    pert_data = soria(data,1000,40)

    print(data.shape)
    print(np.unique(pert_data,axis=0).shape)