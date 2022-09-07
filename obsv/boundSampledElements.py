import numpy as np

def boundSampledElements(Xout,X):
    Xout = np.maximum(Xout,np.min(X,axis=0))
    Xout = np.minimum(Xout,np.max(X,axis=0))
    return Xout