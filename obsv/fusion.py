import numpy as np
from obsv.MDAVwMSE import MDAVwMSE
from obsv.boundSampledElements import boundSampledElements

def SamplingGaussian_m_equal_1(numCluster:int,mean:np.ndarray,variance:np.ndarray):
    return np.random.normal(mean,variance,(numCluster,len(mean)))

def fusion(X:np.ndarray,epsilon,k):
    Xout = np.zeros(X.shape)
    NumAttributes = X.shape[1]
    # Calculating Sensitivity
    DeltaAttribute = 1.5 * np.abs(np.max(X, axis=0) - np.min(X, axis=0))
    MappingXtoXm,kappa,nQAtQ = MDAVwMSE(X,k)

    for IndexCluster in range(kappa):
        ClusterRecordsIDs = np.nonzero(MappingXtoXm==IndexCluster)[0]
        Cluster = X[ClusterRecordsIDs]

        # Compute the actual mean and variance of each attribute of Cluster
        Means=np.mean(Cluster,axis=0)
        Variances =np.var(Cluster,axis=0)

        # Add noise to Means
        ProtectedMeans = Means+ np.random.laplace(0, DeltaAttribute*(2*NumAttributes/k/epsilon))

        # Add noise to Variances
        ProtectedVariances = Variances+np.random.laplace(0,DeltaAttribute**2 *(2*NumAttributes/k/epsilon))

        # Sample from the mean and variance
        ProtectedVariances = np.maximum(0,ProtectedVariances)
        Xout[ClusterRecordsIDs] = SamplingGaussian_m_equal_1(len(ClusterRecordsIDs),ProtectedMeans,ProtectedVariances)
    return boundSampledElements(Xout,X)

if __name__ == '__main__':
    print('Starting FUSION')
    '''import scipy.io
    np.random.seed(0)
    X:np.ndarray = scipy.io.loadmat('source/SrcSoria/CASCrefmicrodataOriginal.mat')['CASCrefmicrodataOriginal']
    r = fusion(X,3,100)'''
    import numpy as np
    import os

    print(os.listdir('.'))
    data = np.load('evaluatingDPML/dataset/housing_features.p', allow_pickle=True)
    fusion(data,10,40)
    print()
