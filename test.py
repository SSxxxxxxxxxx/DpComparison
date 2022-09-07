import os, sys, datetime,pickle
import numpy as np
import pandas as pd
from scipy.spatial import distance


def loadProcessedData(folder:str):
    a = np.load(f'evaluatingDPML/evaluating_dpml/data/{folder}/target_data.npz')
    return [a['arr_%d' % i] for i in range(len(a))]
def loadProcessedData(folder:str):
    return np.load(f'evaluatingDPML/dataset/{folder}_features.p',allow_pickle=True)

def getClosestTo(base:np.ndarray, domain:np.ndarray):
    from scipy.spatial import distance
    return np.argmin(distance.cdist(base, domain), axis=1)

def isDiagonal(arr:np.ndarray):
    return arr==np.arange(len(arr))

def canBeLinked(base:np.ndarray, domain:np.ndarray):
    closest_idex = getClosestTo(base,domain)
    return np.logical_or(closest_idex==np.arange(len(closest_idex)), np.isclose(base,domain[closest_idex]).all(axis=1))

'''
data = []
for obsc in ['soria','fusion']:
    for eps in [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]:
        for k in [1,2,4,9,19,40,83,172,360,751]:
            folder = f'adult_{obsc}_{float(eps)}_{k}'

            a3 = loadProcessedData('adult')
            b3 = loadProcessedData(folder)
            print(folder)

            r = canBeLinked(b3,a3)
            print('Link back',sum(r)/len(r))
            data.append({'obsc':obsc,'eps':eps,'k':k,'way':'back','acc':sum(r)/len(r)})

            r = canBeLinked(a3,b3)
            print('Link forw',sum(r)/len(r))
            data.append({'obsc':obsc,'eps':eps,'k':k,'way':'forw','acc':sum(r)/len(r)})
            print()
            pd.DataFrame(data).to_pickle('linkageAcc.pcl')
'''
'''import matplotlib.pyplot as plt
data: pd.DataFrame = pd.read_pickle('linkageAcc.pcl')
data = data[data['obsc']=='soria']
#data.loc[((data['obsc']=='soria') & (data['eps']==1000.0) & (data['way']=='back'))].plot(x='k',y='acc',kind='line',logy=True,logx=True)
#data.groupby(['obsc','eps','way'])[['k','acc']].plot(x='k',y='acc',legend=True)

fig, ax = plt.subplots(figsize=(16,12))

for label, df in data.groupby(['obsc','eps','way']):
    df.plot(x='k',y='acc',kind="line", ax=ax, label=label,logx=True,logy=True)
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
          fancybox=True, shadow=True, ncol=3)
fig.subplots_adjust(bottom=0.2)
plt.show()'''

'''def createFunc(args):
    return (lambda a,b:lambda X:f'{X}: {a}, {b}')(args.get("a",1),args.get("b",2))

args = {'a':'A','b':'B'}
f1 = createFunc(args)
args['a']='ACA'
f2 = createFunc(args)
print(f1('1-1'))
print(f2('2-2'))'''

def testAgg(data:pd.DataFrame,*args,**kwargs):
    #print('testAgg')
    #print('data',data)
    #print(args,kwargs)
    return data.nlargest(1,'B')

data = pd.DataFrame({'A':['a','a','a','a','b','b','b','c','c'],
                     'B':[1  ,  4,  2,  3,102, 99,202, -5, -9],
                     'C':[  1,  2,  3,  4,  5,  6,  7,  8,  9]})

'''print(data)
print('#'*10)
print(data.groupby(['A']).apply(testAgg).reset_index(drop=True))'''
print(data.apply(lambda x: {'AA':x['B'] * 3,'BB': x['C'] * -2},result_type='expand', axis=1))
data2 = data.join(data.apply(lambda x: {'AA':x['B'] * 3,'BB': x['C'] * -2},result_type='expand', axis=1))
#data.apply(lambda x:(x['B']*3,x['C']*-2),result_type='expand', axis=1)
print(data2)
print(data.pipe(lambda d:d.join(d.apply(lambda x: {'AA':x['B'] * 3,'BB': x['C'] * -2},result_type='expand', axis=1))))


'''
a1, a2, a3, a4 = loadProcessedData('adult')
b1, b2, b3, b4 = loadProcessedData('adult_soria_50.0_360')

r = isDiagonal(getClosestTo(b3,a3))
print(sum(r)/len(r))
r = isDiagonal(getClosestTo(a3,b3))
print(sum(r)/len(r))
'''

'''

with open(f'evaluatingDPML/dataset/{folder}_features.p','rb') as f:
    a = pickle.load(f)
'''

'''
a = np.array([[1,2,3], [2,3,1], [3,1,2]])
b = np.array([[7,0,5], [6,500,0], [0,7,6]])


print(getClosestTo(a,b))
print(isDiagonal(getClosestTo(a,b)))
print(getClosestTo(b,a))
print(isDiagonal(getClosestTo(b,a)))
'''
#print(c)