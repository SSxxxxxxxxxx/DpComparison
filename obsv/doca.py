import os.path

import numpy as np
from typing import Optional,List

def doca(X:np.ndarray,eps,delay_constraint=1000,beta=50,mi=100, inplace=False):
    """
    This is a fast, numpy-utilising, static implementation of doca

    This does not aim at beeing transparent, but short and fast

    :param X: inputdata
    :param eps: privacy budget
    :param delay_constraint: maximum number of "time" a datapoint "stays" in the process
    :param beta: maximum number of clusters kept in memory at the same time
    :param mi: number of clusters used to update tau
    :return: perturbed outputdata
    """
    num_attributes = X.shape[1]

    sensitivity = 1.5*(np.max(X)-np.min(X)) #Is this done right

    clusters:List[List[int]] = []
    clusters_final:List[List[int]] = []
    #Cluster attribute minimum/maximum
    mn_c:List[np.ndarray] = []
    mx_c:List[np.ndarray] = []
    #Global Attribute Minimum/Maximum
    mn:np.ndarray = np.full(X.shape[1],np.inf)
    mx:np.ndarray = np.full(X.shape[1],-np.inf)
    # losses saved for tau
    losses = []
    tau = lambda: np.mean(losses[-mi:]) if losses else 0
    div0 = np.vectorize( lambda a,b:np.divide(a, b, out=np.zeros_like(b), where=b != 0))

    # Create Output structure
    if inplace:
        output = X
    else:
        output = np.zeros(X.shape)


    for clock, data_point in enumerate(X):
        # Update min/max
        mn= np.minimum(mn,data_point)
        mx = np.maximum(mx,data_point)
        dif = mx-mn

        #Find best Cluster
        best_cluster = None
        if clusters:
            #Calculate enlargement (the value is not yet divided by the number of attributes!)
            enlargement = [(np.sum(div0(np.maximum(0, data_point - mxc) - np.minimum(0, data_point - mnc), dif))) for mnc, mxc in zip(mn_c, mx_c)]
            min_enlarge = min(enlargement)

            ok_clusters = []
            min_clusters = []
            for c, enl in enumerate(enlargement): # Search for clusters with minimal enlargement (and additionally with smaller than tau overall loss)
                if enl == min_enlarge:
                    min_clusters.append(c)
                    if (enl+ np.sum(div0(mx_c[c]-mn_c[c],dif)))/num_attributes <= tau():
                        ok_clusters.append(c)

            if ok_clusters:# Choose cluster from the best set according to cluster size
                best_cluster = min( ok_clusters, key=lambda x: len(clusters[c]))
            elif len(clusters) >= beta: #If no new Cluster can be made
                best_cluster = min(min_clusters, key=lambda x: len(clusters[c]))

        if best_cluster is None:
            # Add new Cluster
            clusters.append([clock])
            # Set Min/Max of new Cluster
            mn_c.append(data_point)
            mx_c.append(data_point)
        else:
            # Add point to new cluster
            clusters[best_cluster].append(clock)
            # Update min/max
            mn_c[best_cluster] = np.minimum(mn_c[best_cluster],data_point)
            mx_c[best_cluster] = np.maximum(mx_c[best_cluster],data_point)

        overripe_cluster = [c for c, cl in enumerate(clusters) if clock-delay_constraint in cl]
        assert len(overripe_cluster)<=1, "Every datapoint should only be able to be in one cluster!?"
        if overripe_cluster:
            #print('Throw out cluster of len',len(clusters[overripe_cluster[0]]))
            c = overripe_cluster[0]
            losses.append(np.sum(div0((mx_c[c]-mn_c[c]), dif))/num_attributes)
            clusters_final.append(clusters[c])
            del clusters[c]
            del mn_c[c]
            del mx_c[c]
    clusters_final.extend(clusters)

    for cs in clusters_final:
        print(np.sum(sensitivity)/(len(cs)*eps))
        output[cs] = np.mean(X[cs],axis=0)+np.random.laplace(0,np.sum(sensitivity)/(len(cs)*eps),X.shape[1])
    return output

if __name__ == '__main__':
    import pandas as pd
    data = pd.read_csv('../evaluatingDPML/dataset/adult.csv')
    data = pd.read_csv(os.path.join(os.path.dirname(__file__),'..','evaluatingDPML','dataset','adult.csv'))
    # normalized_df: pd.DataFrame = data / np.std(data, axis=0)
    # normalized_df: np.ndarray = normalized_df.to_numpy()
    # doca(normalized_df,10000,beta=60)
    doca(data.to_numpy()[:400],10000)