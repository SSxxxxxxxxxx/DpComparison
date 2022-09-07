import pickle, sys, os,re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import List,Union,Optional
FILEDIR = os.path.dirname(__file__)

if __name__ == '__main__':
    sys.path.append('..')

FIELD_MODEL = 'model'
FIELD_METH = 'method'
FIELD_EPS = 'eps'
FIELD_RUN = 'run'
FIELD_PRIV = 'privacy'
FIELD_DATA = 'dataset'

from DPMLadapter.ArgsObject import ArgsObject
from runExperiments import adoptArgsToPerturbedData

def getDataPath(args:ArgsObject):
    args = adoptArgsToPerturbedData(args)
    folder = os.path.join(FILEDIR,'..','evaluatingDPML','evaluating_dpml','results',args.train_dataset)
    for filename in os.listdir(folder):
        if re.search(fr'{args.target_model}_(no_privacy)_\d+(\.\d+(e-\d+)?)?_{args.run}\.results\.p',filename):
            return os.path.join(folder,filename)
    return None

def getDataFromFile(filename, params):
    data = np.load(filename, allow_pickle=True)
    train_acc, test_acc, train_loss, was_in_training_data, shokri_mem_adv, shokri_mem_confidence, yeom_mem_adv, per_instance_loss, yeom_attr_adv, pred_membership_all, features = data
    assert shokri_mem_adv != np.NaN
    return {**params, 'train_acc':train_acc, 'test_acc':test_acc, 'shokri_mem_adv':shokri_mem_adv, 'yeom_mem_adv':yeom_mem_adv, 'yeom_attr_adv':yeom_attr_adv,}

def getDataFromMethodsFolder(folder, params):
    for filename in os.listdir(folder):
        res =re.match(fr'([a-zA-Z]+)_(?:no_privacy|grad_pert)_\d+(?:\.\d+(?:e-\d+)?)?_(\d+)\.results\.p',filename)
        if res:
            model, run,*_ = res.groups()
            yield getDataFromFile(os.path.join(folder,filename),{**params, FIELD_MODEL:model, FIELD_RUN:int(run)})
        else:
            raise f'Could not interpret {filename} in {folder}'

def getDataFromStandardFolder(folder,params):
    for filename in os.listdir(folder):
        res =re.match(fr'([a-zA-Z]+)_(?:no_privacy|grad_pert)_\d+(?:\.\d+(?:e-\d+)?)?_(\d+)\.results\.p',filename)
        if res:
            model, run,*_ = res.groups()
            yield getDataFromFile(os.path.join(folder,filename),{**params, FIELD_MODEL:model, FIELD_RUN:int(run)})
            continue
        res =re.match(fr'([a-zA-Z]+)_(?:no_privacy|grad_pert)_([a-zA-Z]+|adv_cmp)_(\d+(?:\.\d+(?:e-\d+)?)?)_(\d+)\.results\.p',filename)
        if res:
            model, privacy_definition, eps, run,*_ = res.groups()
            yield getDataFromFile(os.path.join(folder,filename),{**params, FIELD_MODEL:model, FIELD_PRIV:privacy_definition, FIELD_EPS:float(eps), FIELD_RUN:int(run)})
        else:
            raise f'Could not interpret {filename} in {folder}'

def stuff():
    results_folder = os.path.join(FILEDIR,'..','evaluatingDPML','evaluating_dpml','results')
    folders = os.listdir(os.path.join(FILEDIR,'..','evaluatingDPML','evaluating_dpml','results'))
    for foldername in folders:# Allows for up to 8 number-Parameters
        if '_' not in foldername:
            yield from getDataFromStandardFolder(os.path.join(results_folder, foldername),{FIELD_DATA:foldername})
        res = re.match(r'([a-zA-Z]+)_([a-zA-Z]+)_(\d+(?:\.\d+(?:e-\d+)?)?)(?:_(\d+(?:\.\d+(?:e-\d+)?)?)(?:_(\d+(?:\.\d+(?:e-\d+)?)?))?(?:_(\d+(?:\.\d+(?:e-\d+)?)?))?(?:_(\d+(?:\.\d+(?:e-\d+)?)?))?(?:_(\d+(?:\.\d+(?:e-\d+)?)?))?(?:_(\d+(?:\.\d+(?:e-\d+)?)?))?(?:_(\d+(?:\.\d+(?:e-\d+)?)?))?(?:_(\d+(?:\.\d+(?:e-\d+)?)?))?)?',foldername)
        if res:
            dataset, method, eps, *args = res.groups()
            if method in ['soria', 'fusion']:
                yield from getDataFromMethodsFolder(os.path.join(results_folder, foldername), {FIELD_METH: method, FIELD_DATA: dataset, FIELD_EPS: float(eps),'k':int(args[0])})
            elif method=='doca':
                yield from getDataFromMethodsFolder(os.path.join(results_folder, foldername), {FIELD_METH: method,
                                                                 FIELD_DATA: dataset,
                                                                 FIELD_EPS: float(eps),
                                                                 'doca_delay_constraint':int(args[0]),
                                                                 'doca_beta':int(args[1]),
                                                                 'doca_mi':int(args[2])})
        else:
            print(foldername)

data = pd.DataFrame(stuff())
print(data[FIELD_METH].unique())
a = data.groupby([c for c in data.columns if not (c in ['train_acc', 'test_acc', 'shokri_mem_adv', 'yeom_mem_adv', 'yeom_attr_adv'])])
print('A:',a.indices,sep='\n')
data = data[data['dataset']=='adult']
#data[data['privacy'].notnull()].plot.scatter(x="shokri_mem_adv", y="test_acc", s=5)
#plt.show()


'''plt.boxplot([[1,2,3,3.3,4],[2,3,4,4.5,5],[3,4,5,5.5,6]], positions=[2,4,5.5])
plt.boxplot([[3,4,5,5.5,6]], positions=[2])
plt.boxplot([[1,10,2,5,2,2,2,5,3,7]], positions=[3.5])'''

#['method', 'dataset', 'eps', 'k', 'model', 'run', 'train_acc', 'test_acc', 'shokri_mem_adv', 'yeom_mem_adv', 'yeom_attr_adv', 'doca_delay_constraint', 'doca_beta', 'doca_mi', 'privacy']

print(list(data.columns))

result_fields = ['train_acc', 'test_acc', 'shokri_mem_adv', 'yeom_mem_adv', 'yeom_attr_adv']
method_params_fields = ['k', 'doca_beta', 'doca_mi', 'doca_delay_constraint']

data = data[data['model']=='nn']
advs = []
accs = []
cols = []
trad_data = data[data['privacy'].notnull()]
'''for _,scenario_data in trad_data.groupby([c for c in data.columns if not (c in [*result_fields, *method_params_fields, 'run'])], dropna=False):
    print(_)
    advs.append(scenario_data['yeom_mem_adv'].mean())
    accs.append(scenario_data['test_acc'].mean())
    cols.append('r')'''

for meth, c in zip(['soria','fusion','doca'],['green', 'blue','yellow','pink']):
    meth_data = data[data['method']==meth]
    for _, scenario_data in meth_data.groupby([c for c in data.columns if not (c in [*result_fields, *method_params_fields, 'run'])], dropna=False):
        print(_)
        mx = -100
        mx_acc = -100
        corr_adv =0
        best_params = None
        for _, fancy_data in scenario_data.groupby([c for c in data.columns if not (c in [*result_fields, 'run'])], dropna=False):
            if scenario_data['test_acc'].mean()>mx:
                mx = scenario_data['test_acc'].mean()
                mx_acc = scenario_data['test_acc'].mean()
                corr_adv = scenario_data['eps'].mean()
                best_params = _
        print('Best:',best_params)
        advs.append(corr_adv)
        accs.append(mx_acc)
        cols.append(c)
[
{'A':1, 'B': 1, 'run':1, 'val':1},
{'A':1, 'B': 1, 'run':2, 'val':2},
{'A':1, 'B': 1, 'run':3, 'val':3},

{'A':1, 'B': 2, 'run':1, 'val':2},
{'A':1, 'B': 2, 'run':2, 'val':4},
{'A':1, 'B': 2, 'run':3, 'val':6},

{'A':2, 'B': 1, 'run':1, 'val':2},
{'A':2, 'B': 1, 'run':2, 'val':4},
{'A':2, 'B': 1, 'run':3, 'val':6},
]
print('=>',*(tuple(x) for x in zip(advs,accs)))
print()
plt.scatter([np.log(x) for x in advs],accs,c=cols)
#plt.scatter(advs,accs,c=cols)
plt.show()
#print(getDataPath(ArgsObject('diabetes',method='soria',target_epsilon=0.1,k=1,run=1)))


'''data = np.load('../evaluatingDPML/evaluating_dpml/results/',allow_pickle=True)
print(data)'''