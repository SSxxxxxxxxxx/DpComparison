import numpy as np

def MDAVwMSE(data:np.ndarray, K: int):
    '''


    :param data: data array of features
    :param K: K
    :return:
    '''
    # XAmt = Number of data points
    # DimAmt = Number of attributes
    XAmt, DimAmt = data.shape
    QAmt = int(XAmt/K) # Number of clusters
    QAtX = np.zeros(XAmt,dtype=np.int) # Mapping from dp to Cluster
    Is = np.ones(XAmt,dtype=np.bool)
    IAmt = XAmt
    G = 0

    while IAmt>=2*K:
        X0 = np.sum(data[Is], axis=0)/IAmt
        Ds = np.linalg.norm(data[Is]-X0,axis=1)
        # Ds =np.sum((data[Is]-X0)**2,axis=1)

        for _ in range(2):
            # Find index of point which is the furthest away from X0 (or X1)
            IFar = np.argmax(Ds)
            # Actual point
            X1 = data[Is][IFar]
            # Build new distances
            Ds = np.linalg.norm(data[Is]-X1,axis=1)

            # Find K nearest Points
            IsSort = np.argpartition(Ds,kth=K-1)[:K]

            # Assign Groups
            #QAtX[Is][IsSort]=G
            QAtX[np.nonzero(Is)[0][IsSort]] = G
            G+=1

            # Remove assigned points from considered point and distances
            Is[np.nonzero(Is)[0][IsSort]] = False
            # Is[Is][IsSort]=False
            Ds = np.delete(Ds,IsSort)
            # Adjust remaining-point-count
            IAmt = IAmt-K
    QAtX[Is] = QAmt-1
    nQAtQ = np.tile(K,QAmt) #Number of points in each Cluster
    nQAtQ[-1]=XAmt-K*(QAmt-1)
    return QAtX, QAmt, nQAtQ # Mapping dp to cluster, Number of clusters, Mapping cluster to number of dp in cluster

def MDAVwMSE_readable(data:np.ndarray, K: int):
    '''


    :param data: data array of features
    :param K: K
    :return:
    '''

    total_num_datapoints, num_attributes = data.shape
    total_num_clusters = int(total_num_datapoints/K) # Number of clusters
    cluster_mapping = np.zeros(total_num_datapoints,dtype=np.int) # Mapping from dp to Cluster
    unvisited_datapoints = np.ones(total_num_datapoints,dtype=np.bool)
    num_remaining_datapoints = total_num_datapoints
    current_cluster_id = 0

    while num_remaining_datapoints>=2*K:
        remaining_centroid = np.sum(data[unvisited_datapoints], axis=0)/num_remaining_datapoints
        distances_to_remaining_centroid = np.linalg.norm(data[unvisited_datapoints]-remaining_centroid,axis=1)
        # Ds =np.sum((data[Is]-X0)**2,axis=1)

        for _ in range(2):
            # Find index of point which is the furthest away from X0 (or X1)
            current_datapoint_index = np.argmax(distances_to_remaining_centroid)
            # Actual point
            X1 = data[unvisited_datapoints][current_datapoint_index]
            # Build new distances
            distances_to_remaining_centroid = np.linalg.norm(data[unvisited_datapoints]-X1,axis=1)

            # Find K nearest Points
            current_cluster_datapoint_ids = np.argpartition(distances_to_remaining_centroid,kth=K-1)[:K]

            # Assign Groups
            cluster_mapping[np.nonzero(unvisited_datapoints)[0][current_cluster_datapoint_ids]] = current_cluster_id
            current_cluster_id+=1

            # Remove assigned points from considered point and distances
            unvisited_datapoints[np.nonzero(unvisited_datapoints)[0][current_cluster_datapoint_ids]] = False
            distances_to_remaining_centroid = np.delete(distances_to_remaining_centroid,current_cluster_datapoint_ids)
            # Adjust remaining-point-count
            num_remaining_datapoints = num_remaining_datapoints-K
    cluster_mapping[unvisited_datapoints] = total_num_clusters-1
    num_datapoints_per_cluster = np.tile(K,total_num_clusters) #Number of points in each Cluster
    num_datapoints_per_cluster[-1]=total_num_datapoints-K*(total_num_clusters-1)
    return cluster_mapping, total_num_clusters, num_datapoints_per_cluster # Mapping dp to cluster, Number of clusters, Mapping cluster to number of dp in cluster
