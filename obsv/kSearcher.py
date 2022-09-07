import datetime

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from math import floor,ceil
import matplotlib.pyplot as plt
from typing import List,Tuple,Union,Optional,Dict


def optimal_clustering_K_by_silhouette (X:np.ndarray, num_cluster_options:List[int]=None, plot=False, **kwargs):
    if num_cluster_options is None:
        num_cluster_options = [2,5,10,20,40,60,80,100,150,200,250,300,400,500,700,900,1000,1200,1400,1600,1800,2000]
    mean_silhouettes = []
    for num_cluster in num_cluster_options:
        print(f'Clustering with {num_cluster} clusters')
        start = datetime.datetime.now()
        result:KMeans = KMeans(n_clusters=num_cluster, random_state=0).fit(X)
        print('KMeans took',datetime.datetime.now()-start)
        start = datetime.datetime.now()
        mean_silhouettes.append(silhouette_score(X,result.labels_))
        print('Score took ',datetime.datetime.now()-start)
        print(num_cluster, mean_silhouettes[-1])
    print(num_cluster_options,mean_silhouettes)
    max_cluster_id_size_before_steady_rise=max((i for i in range(len(num_cluster_options)-1) if mean_silhouettes[i]>mean_silhouettes[i+1]),default=None)
    if max_cluster_id_size_before_steady_rise is not None:
        y, x = max(list(zip(mean_silhouettes, num_cluster_options))[:max_cluster_id_size_before_steady_rise])
    if plot:
        plotShilouette(num_cluster_options,mean_silhouettes,mark=None if max_cluster_id_size_before_steady_rise is None else (x,y), **kwargs)
    if max_cluster_id_size_before_steady_rise is None:
        raise Exception('Could not find a best value for k')
    return x

def plotShilouette(xs,ys,mark:Tuple[int,int]=None, name=None,**kwargs):
        plt.plot(xs, ys,'bx-')
        plt.xlabel('Values of K')
        plt.ylabel('Silhouette score')
        plt.title('Silhouette analysis For Optimal k' + ('' if name is None else f' for {name}'))
        if mark is not None:
            x,y = mark
            plt.annotate(str(x), xy=(x, y), xytext=(1, 4), color='red', textcoords='offset points')
            plt.plot(x,y,'rx')
        plt.show()

def get_cluster_sizes_expo(num_data, num_lables, target_num_options):
    max_k = int(num_data/num_lables)
    return np.unique(np.round(np.logspace(0,np.log10(max_k),target_num_options))).astype(int)
def get_cluster_sizes_expo_by_cluster_num(num_data, num_lables, target_num_options): # Equivalent with get_cluster_sizes_expo
    # print(np.logspace(np.log10(num_lables),np.log10(num_data),target_num_options))
    return np.unique(np.round(num_data/np.logspace(np.log10(num_lables),np.log10(num_data),target_num_options)))

def get_cluster_sizes_lin(num_data, num_lables, target_num_options):
    max_k = int(num_data/num_lables)
    return np.unique(np.round(np.linspace(0,max_k,target_num_options)))

if __name__ == '__main__':
    # adult [0.2639591211441367, 0.29901940004914634, 0.28071293272781056, 0.29555369576228113, 0.3107547634049323, 0.3165957586996981, 0.30459574987480825, 0.29389575288404385, 0.29931839385519626, 0.2970918927363058, 0.29735192692384055, 0.30959004256768435, 0.30732031862208586, 0.32293295689175405, 0.33541703069527873, 0.3516974994562655, 0.3582956149191953, 0.3764205421286009, 0.38562816246556236, 0.39579956141474454, 0.4048391847647498, 0.4123155643593658]
    # irish [0.23987263559295582, 0.2938993399689734, 0.41018570363599594, 0.5071581995189279, 0.43379694526018525, 0.3989782079542836, 0.38889840606619636, 0.38481060855766713, 0.40166434752434177, 0.43338917293842744, 0.4649287628134337, 0.48382621346391336, 0.5139542190089794, 0.5332700992897018, 0.5519468694943742, 0.5693351637885947, 0.5920749638138998, 0.6098052000255741, 0.6352239025973941, 0.6727225825514103, 0.717141447303924, 0.7659921431256798]

    import pandas as pd
    data:pd.DataFrame = pd.read_csv('../evaluatingDPML/dataset/housing.csv')
    drop_fields = ['ocean_proximity']
    log_fields=['total_rooms','total_bedrooms','population','households']

    # Preprocessing#
    data = data.drop_duplicates()
    data = data.dropna()
    if drop_fields:
        data = data.drop(columns=drop_fields)
    for c in list(data.columns):
        if c in log_fields:
            data[c] = np.log(1+data[c])

    normalized_df:np.ndarray = (data - np.mean(data, axis=0))/np.std(data,axis=0)
    print('Shape:',normalized_df.shape)
    # housing
    #normalized_df: np.ndarray = normalized_df.to_numpy()
    #optimal_clustering_K_by_silhouette(normalized_df, num_cluster_options=[2,5,10,20,40,60,80,100,150,200,250,300,400,500], name='housing', plot=True)
    #[2, 5, 10, 20, 40, 60, 80, 100, 150, 200, 250, 300, 400, 500] [0.20120664875327302, 0.23661318943430512, 0.19301234450949714, 0.1795426457837274, 0.1676716189298121, 0.15871419320270233, 0.15677075472343138, 0.1513664455336867, 0.14975813149684664, 0.14605359120400901, 0.1441115445598599, 0.14254043827338583, 0.1435753107496716, 0.14258100904718862]
    optimal_clustering_K_by_silhouette(normalized_df, num_cluster_options=list(range(2,11)), name='housing', plot=True)
    #[2, 3, 4, 5, 6, 7, 8, 9, 10] [0.20120664875327302, 0.23224719086545703, 0.23098501727212012, 0.23661318943430512, 0.22605910960410366, 0.19678163780896105, 0.20407511677842224, 0.19315107277984064, 0.19301234450949714]
    # Best k seems to be 5 :(

    # adults unique
    #optimal_clustering_K_by_silhouette(normalized_df, num_cluster_options=[2,5,10,20,40,60,80,100,150,200,250,300,400,500], name='adult', plot=True)
    #[2, 5, 10, 20, 40, 60, 80, 100, 150, 200, 250, 300, 400, 500] [0.21941534743745936, 0.2494358579432236, 0.275866992131926, 0.27621801256723744, 0.26562481552044515, 0.2621114940147862, 0.26530622168316126, 0.25516602910710234, 0.2541497256881973, 0.2439703317103089, 0.24019147903144009, 0.24284687383891465, 0.24236233041428695, 0.24428818664884355]
    #optimal_clustering_K_by_silhouette(normalized_df, num_cluster_options=list(range(5,101,5)), name='adult', plot=True)
    #[5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100] [0.2494358579432236, 0.275866992131926, 0.26089827774738233, 0.27621801256723744, 0.265657224329223, 0.26363818913961273, 0.26116031555009656, 0.26562481552044515, 0.2641439006341046, 0.26553564935407054, 0.2643960297692998, 0.2621114940147862, 0.26712445428843534, 0.26173298568302517, 0.25567166534173474, 0.26530622168316126, 0.26326931475597626, 0.26086100913262134, 0.25211935009921793, 0.25516602910710234]
    #optimal_clustering_K_by_silhouette(normalized_df, num_cluster_options=list(range(5,31,1)), name='adult', plot=True)
    #[5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30] [0.2494358579432236, 0.2419151520757351, 0.2629759059727314, 0.26147237908429255, 0.2729684534771858, 0.275866992131926, 0.25381252502130863, 0.27850741103607285, 0.2481627603484711, 0.2590989703731599, 0.26089827774738233, 0.2722245784114261, 0.26753664663976107, 0.2744970217136446, 0.26949054920960636, 0.27621801256723744, 0.25982716933452293, 0.26432625520789915, 0.26644082197967045, 0.2634750826775752, 0.265657224329223, 0.26443544247230893, 0.26143015401250524, 0.26438631679866725, 0.2630699236432354, 0.26363818913961273]
    #The best k is 12. Yet, 10 and 20 are close competitors. Sooo, I'll just take 20 as this significantly lowers the chance of randomly guessing right.



    # Old runs with duplicates
    #[2, 5, 10, 20, 40, 60, 80, 100, 150, 200, 250, 300, 400, 500] [0.2639591211439875, 0.29901940004881467, 0.2807129327272068, 0.2955536957604389, 0.31075476340168956, 0.31656884457382783, 0.30458764847743197, 0.29560140941592616, 0.29517087704966133, 0.29734550270257165, 0.29722542326669804, 0.3096187717442295, 0.30926911915412997, 0.32299097174342745]
    #optimal_clustering_K_by_silhouette(normalized_df, num_cluster_options=[30,40,45,50,55,58,60,62,65,70,80,90], name='adult', plot=True)
    #[30, 40, 45, 50, 55, 58, 60, 62, 65, 70, 80, 90] [0.311383410325172, 0.31075476340168956, 0.2968954925181276, 0.30824925254267527, 0.30773655394142113, 0.30894692786400313, 0.31656884457382783, 0.30315062987688185, 0.31781981308223217, 0.30535836368803787, 0.30458764847743197, 0.2939220084510083]
    #optimal_clustering_K_by_silhouette(normalized_df, num_cluster_options=[58,59,60,61,62,63,64,65,67,68,69], name='adult', plot=True)
    #[58, 59, 60, 61, 62, 63, 64, 65, 67, 68, 69] [0.30894692786400313, 0.3095766171790946, 0.31656884457382783, 0.3015116770698989, 0.30315062987688185, 0.30131701881514655, 0.3067716122647048, 0.31781981308223217, 0.3092039108240519, 0.29742013243973775, 0.3063778670206241]

    exit()
    data = pd.read_csv('../evaluatingDPML/dataset/irishn.csv')
    normalized_df: pd.DataFrame = (data - np.mean(data, axis=0))/np.std(data,axis=0)
    normalized_df: np.ndarray = normalized_df.to_numpy()
    #optimal_clustering_K_by_silhouette(normalized_df, num_cluster_options=[2,5,10,20,40,60,80,100,150,200,250,300,400,500], name='irishn', plot=True)
    #[2, 5, 10, 20, 40, 60, 80, 100, 150, 200, 250, 300, 400, 500] [0.23987263559612776, 0.2938993399788257, 0.4101857036536929, 0.5071581995690181, 0.4337969454042583, 0.3989894673289123, 0.3899085509586262, 0.384003808240228, 0.4021096568344952, 0.43188117471002435, 0.4626328442776322, 0.4820347946425681, 0.5127654422247623, 0.5334992133886277]
    #optimal_clustering_K_by_silhouette(normalized_df, num_cluster_options=[10,15,18,20,22,25,30,35,40], name='irishn', plot=True)
    #[10, 15, 18, 20, 22, 25, 30, 35, 40] [0.4101857036536929, 0.46979477219884, 0.505433232246344, 0.5071581995690181, 0.5119100805266695, 0.5001945900037729, 0.4803467469703103, 0.4671431903388189, 0.4337969454042583]
    #optimal_clustering_K_by_silhouette(normalized_df, num_cluster_options=[18,19,20,21,22,23,24,25,26], name='irishn', plot=True)
    #[18, 19, 20, 21, 22, 23, 24, 25, 26] [0.505433232246344, 0.5056329945500254, 0.5071581995690181, 0.5090417784010125, 0.5119100805266695, 0.5011101654657855, 0.4949202773685152, 0.5001945900037729, 0.49080483868894664]
