import csv,pandas,pickle,os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize


def combineCSVs(output, *inp_files):
    with open(output,'w', newline='') as f:
        writer = csv.writer(f)
        header =None
        for file in inp_files:
            reader = csv.reader(open(file))
            data = next(reader)
            if header is None:
                header = data
                writer.writerow(header)
            print(data)
            for data in reader:
                writer.writerow(data)
                #print(data)
        f.close()

def createDatasetFromCSV(file:str,dataname=None,log_fields=None, drop_fields = None, showPlots = False,num_lables=100):
    fileDir = os.path.dirname(file)
    if dataname is None:
        dataname = os.path.basename(file).split('.')
        if dataname[-1] == 'csv':
            del dataname[-1]
        dataname = '.'.join(dataname)
    log_fields = [] if log_fields is None else log_fields
    drop_fields = [] if drop_fields is None else drop_fields
    print('-' * 10, 'Converting and labeling data', '-' * 10)
    data = pandas.read_csv(file)
    # Make dataset unique
    data = data.drop_duplicates()
    data = data.dropna()

    if drop_fields:
        data = data.drop(columns=drop_fields)

    for c in list(data.columns):
        if c in log_fields:
            data[c] = np.log(1+data[c])

    if showPlots:
        for c in list(data.columns):
            data[c].plot.hist(title=c,bins=40)
            plt.show(title=c)

    print('Shuffel')
    data.sample(frac=1).reset_index(drop=True)

    print('Normalise')
    # normalized_df: pandas.DataFrame = data/np.std(data,axis=0) # TODO Normalize by standart diviation
    normalized_df: pandas.DataFrame = (data - np.mean(data, axis=0))/np.std(data,axis=0)
    normalized_df: np.ndarray = normalized_df.to_numpy()
    pickle.dump(normalized_df, open(os.path.join(fileDir,f'{dataname}_features.p'), 'wb'))

    print('Labeling...')
    X = KMeans(n_clusters=num_lables, random_state=0).fit(normalized_df)
    pickle.dump(X.labels_, open(os.path.join(fileDir, f'{dataname}_labels.p'), 'wb'))
