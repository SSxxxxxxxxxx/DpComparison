import ast
import datetime
import os
import pickle,csv, argparse
import re
import sys
import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from createDataset import combineCSVs,createDatasetFromCSV

import sklearn,random
'''from sklearn.model_selection import train_test_split

#sklearn.random.seed(0)
np.random.seed(0)
print(train_test_split(np.arange(8),random_state=np.random.randint(0,2**32 - 1)))

#sklearn.random.seed(0)
np.random.seed(0)
print(train_test_split(np.arange(8),random_state=np.random.randint(0,2**32 - 1)))
'''



'''for s in ['adult','diabetes','irishn']:
    combineCSVs(f'evaluatingDPML/dataset/{s}.csv',
                f'evaluatingDPML/dataset/{s}_test.csv',
                f'evaluatingDPML/dataset/{s}_train.csv')'''

#createDatasetFromCSV('evaluatingDPML/dataset/adult.csv',log_fields=['capital-gain'])
#createDatasetFromCSV(f'evaluatingDPML/dataset/diabetes.csv',log_fields=['insulin'])
#createDatasetFromCSV(f'evaluatingDPML/dataset/irishn.csv')


'''a = pickle.load(open('evaluatingDPML/dataset/purchase_100_labels.p', 'rb'))
print(a[:10])'''
'''
from paramsearch.sigmaSearcher import get_advcmp_sigma,get_advcmp_eps
from paramsearch.sigmaSearcher import get_zcdp_sigma,get_zcdp_eps
from paramsearch.sigmaSearcher import get_gdp_sigma,get_gdp_eps
from paramsearch.sigmaSearcher import get_rdp_sigma,get_rdp_eps
eps_range = [0.01,0.05,0.1,0.5,1,2,5,10,50,100,500,1000]

sigmas = []

print('advcmp')
print('#'*20)
for eps in eps_range:
    sigma = get_advcmp_sigma(eps,50000,300,60,1e-5)
    sigmas.append(sigma)
    #eps2 = get_advcmp_eps(sigma,50000,300,60,1e-5)
    #print(f'{eps}: {eps2:.4f} ({abs(eps-eps2):.4f}/{abs(eps-eps2)/eps*100:.4f}%)')
print('\n'*4)

print('zcdp')
print('#'*20)
for eps in eps_range:
    sigma = get_zcdp_sigma(eps,50000,300,60,1e-5)
    sigmas.append(sigma)
    #eps2 = get_zcdp_eps(sigma,50000,300,60,1e-5)
    #print(f'{eps}: {eps2:.4f} ({abs(eps-eps2):.4f}/{abs(eps-eps2)/eps*100:.4f}%)')
print('\n'*4)

print('gdp')
print('#'*20)
for eps in eps_range[:-1]:
    sigma = get_gdp_sigma(eps,50000,300,60,1e-5)
    sigmas.append(sigma)
    #eps2 = get_gdp_eps(sigma,50000,300,60,1e-5)
    #print(f'{eps}: {eps2:.4f} ({abs(eps-eps2):.4f}/{abs(eps-eps2)/eps*100:.4f}%)')
print('\n'*4)


print('rdp')
print('#'*20)
for eps in eps_range:
    sigma = get_rdp_sigma(eps,50000,300,60,1e-5)
    sigmas.append(sigma)
    #eps2 = get_rdp_eps(sigma,50000,300,60,1e-5)
    #print(f'{eps}: {eps2:.4f} ({abs(eps-eps2):.4f}/{abs(eps-eps2)/eps*100:.4f}%)')
print('\n'*4)


print(sorted(sigmas))'''

'''from evaluatingDPML.core.constants import rdp_noise_multiplier
from paramsearch.sigmaSearcher import get_rdp_sigma
for eps in rdp_noise_multiplier[100]:
    print('Running')
    sigma = get_rdp_sigma(eps,10000,100,200,1e-5)
    print(eps,sigma,rdp_noise_multiplier[100][eps],sep='\t')'''

from multiprocessing import Process, Lock, Pool
import os
def printLorem(i):
    print('Lorem ipsum',i)

'''def f(i):
    return 
    sys.stdout = open(os.path.join(os.path.dirname(__file__),'logs-debug',f'std-out-{i}.log'), 'w',buffering=1)
    sys.stderr = open(os.path.join(os.path.dirname(__file__),'logs-debug',f'std-err-{i}.log'), 'w',buffering=1)
    time.sleep(1)
    printLorem(i)
    time.sleep(1)
    printLorem(-2*i)


if __name__ == '__main__':
    processes = []
    for num in range(10):
        processes.append(Process(target=f, args=(num,)))
        processes[-1].start()
    for p in processes:
        p.join()'''

'''def f(x,y):
    key = os.getenv('Mark',None)
    if key is None:
        os.environ['Mark']=str(random.random())
    print(f'Starting {x}, {y} => {os.getenv("Mark",None)}')
    start = datetime.datetime.now()
    time.sleep((6-x)/2)
    print(f'Finish {x}, {y} after {datetime.datetime.now()-start} => {os.getenv("Mark",None)}')
    return {'Hiiiiii':x,'Moin':y}

if __name__ == '__main__':
    with Pool(3,maxtasksperchild=1) as p:
        print(p.starmap(f, [(1,2), (2,3), (3,4), (4,5), (5,6), (6,7)]))
        print('Finished loading')
    print('EXIT')'''
'''if __name__ == '__main__':
    with Pool(3) as p:
        print(p.map(f, [1, 2, 3, 4, 5, 6]))
        print('Finished loading')
    print('EXIT')'''
'''def f(*args):
    print(args)
    time.sleep(1)
    return args

def apf(x):
    ind,x =x
    time.sleep(random.random())
    print(f'Starting {ind}',x,sep='\n',end='\n\n')
    time.sleep(6)
    print(f'Finished {ind}')
    return {'Hi': x['A'] + 2 + x['B']}

if __name__ == '__main__':
    data = pd.DataFrame({'A':[1,2,3,4],'B':[5,6,7,8]})
    with Pool(3) as p:#,maxtasksperchild=1
        print(pd.DataFrame(p.imap(apf,data.iterrows()),index=data.index))
        #print(p.starmap(f, data.iterrows()))
        print('Finished loading')
    print('EXIT')'''
import multiprocessing
def dummy(t):
    print(f'Start {t}')
    A = np.random.rand(10000, 10000)
    inv = np.linalg.inv(A)
    ret =np.linalg.norm(inv)
    from scipy.spatial import distance
    a = distance.cdist(np.random.rand(10_000,100),np.random.rand(10_000,100))
    print(f'Finish {t}',a.shape)
    return {f'Hii':f'Moin {t}',2:3,9:2*t}


if __name__ == "__main__":
    fig, axs = plt.subplots(1, 2, figsize=(4, 2), constrained_layout=True)
    axs[0].plot(np.arange(10))
    lines = axs[1].plot(np.arange(10), label='This is a plot')
    labels = [l.get_label() for l in lines]
    leg = fig.sub.legend(lines, labels, loc='center left',
                     bbox_to_anchor=(0.8, 0.5), bbox_transform=axs[1].transAxes)
    fig.show()
    exit()
    from pool import Pool2
    with multiprocessing.Pool(5) as pool:
        print(pd.DataFrame(pool.imap(dummy,range(100))))
        processes = []
        '''for i in range(10):
            processes.append(pool.apply_async(dummy,(i,)))
        print('Finishied INIT')
        for p in processes:
            p.wait()
            print(p.get())'''
        #print(pool.map(dummy, range(20)))
    '''pool = Pool2(processes=2,sleep_time=2)
    for i in range(20):
        pool.apply_async(dummy,(i,))
    pool.wait()'''
    print('Fnish all')
'''import shutil
folder = os.path.join('evaluatingDPML','evaluating_dpml','results')
for f in os.listdir(folder):
    print(f)
    if not f.startswith('adult'):
        continue
    for ff in os.listdir(os.path.join(folder,f)):
        for filename in os.listdir(os.path.join(folder,f,ff)):
            file = os.path.join(folder,f,ff,filename)
            if os.path.isfile(file):
                continue
            if not re.match(r'\d+',filename):
                print('ERROR',file)
                continue
            print('Deleting', file)
            shutil.rmtree(file, ignore_errors=True)'''