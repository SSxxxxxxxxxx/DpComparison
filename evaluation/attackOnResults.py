import datetime
import multiprocessing

from sklearn.metrics import roc_curve
import sys, os,pickle

from typing import Union,Tuple,Hashable,Any,List,Optional
from scipy.spatial import distance as ssdistance
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
FILEDIR = os.path.dirname(__file__)

'''

'''
#ATTACKER_RESULTS_FILE = os.path.join(FILEDIR, '..', 'savedResultsIrish.pcl.gz')
ATTACKER_RESULTS_FILE_PURCHASE = os.path.join(FILEDIR, '..', 'savedResultsPurchase_100.pcl.gz')
ATTACKER_RESULTS_FILE_ADULT = os.path.join(FILEDIR, '..', 'savedResultsAdult.pcl.gz')
ATTACKER_RESULTS_FILE_HOUSING = os.path.join(FILEDIR, '..', 'savedResultsHousing.pcl.gz')
#ATTACKER_RESULTS_FILE = os.path.join(FILEDIR, '..', 'savedResults.pcl.gz')
ATTACKER_RESULTS_FILE = ATTACKER_RESULTS_FILE_ADULT

SIGMA_RESULTS_FILE = os.path.join(FILEDIR, '..', 'savedSigmas.pcl.gz')
ATTACKS_RESULTS_FILE = os.path.join(FILEDIR, '..', 'savedAttacks.pcl.gz')
AGG_RESULTS_FILE = os.path.join(FILEDIR, '..', 'savedAggregation.pcl.gz')
FINAL_EXP_FILE = os.path.join(FILEDIR, '..', 'savedFinalExp.pcl.gz')

KEY = ['base_dataset', 'run', 'method', 'k', 'target_epsilon', 'target_model', 'target_dp','target_privacy']
REDUNDANT = ['use_cpu','save_model','save_data','n_shadow']
MODEL_META = ['target_learning_rate','target_batch_size','target_n_hidden','target_epochs','target_l2_ratio','target_clipping_threshold']
ATTACKER_META = ['attack_model', 'attack_learning_rate', 'attack_batch_size', 'attack_n_hidden', 'attack_epochs', 'attack_l2_ratio']

if __name__ == '__main__':
    print('Adding to path')
    sys.path.append(os.path.join(FILEDIR,'..'))
from utils import FullPandasPrinter
from evaluatingDPML.core.attack import load_data, yeom_membership_inference
from DPMLadapter.ArgsObject import ArgsObject

def mergeUpdate(data:pd.DataFrame,addition:pd.DataFrame,on):
    uncomcols = [c for c in list(data.columns) if c not in on and c in list(addition.columns)]
    data = pd.merge(data,addition,on=on,how='left')
    for c in uncomcols:
        data[c]=data[f'{c}_y'].fillna(data[f'{c}_x'])
        data.drop([f'{c}_x', f'{c}_y'], axis=1,inplace=True)
    return data



def loadModel(path):
    import tensorflow.compat.v1 as tf1
    import tensorflow as tf
    loaded = tf.saved_model.load(path)
    class LayerFromSavedModel(tf1.keras.layers.Layer):
        def __init__(self):
            super(LayerFromSavedModel, self).__init__()
            self.vars = loaded.variables

        def call(self, inputs):
            return loaded.signatures['serving_default'](inputs)

    input = tf1.keras.Input(shape=(loaded.variables.variables[0].shape[0],))
    return tf1.keras.Model(input, LayerFromSavedModel()(input))

def _getScore(real, pred):
    fpr, tpr, thresholds = roc_curve(real, pred, pos_label=1,drop_intermediate=False)
    return tpr[1]-fpr[1]

def getScore(real, pred, threshold = 1, invert = False):
    if invert:
        pred = -pred
    fpr, tpr, thresholds = roc_curve(real, pred, pos_label=1,drop_intermediate=False)
    score = tpr-fpr
    if invert:
        threshold = -threshold
    if threshold is None:
        return score, thresholds
    for thr, sc in reversed(list(zip(thresholds,score))):
        if thr>=threshold:
            return sc
    return 0

def getScoreDict(real, pred, invert = False):
    if invert:
        pred = -pred
    fpr, tpr, thresholds = roc_curve(real, pred, pos_label=1,drop_intermediate=False)
    fpr, tpr, thresholds = fpr[1:], tpr[1:], thresholds[1:]
    score = tpr - fpr
    if invert:
        thresholds = -thresholds
    return dict(zip(thresholds,score))

def loadResult(file):
    data:dict = pickle.load(open(file,'rb'))
    if 'base_dataset' not in data['args'].keys():
        data['args']['base_dataset'] = data['args']['train_dataset']
    data['args'] = ArgsObject(**data['args'])
    return data

def loadResultData(results_folder= None, include_fields=None)->pd.DataFrame:
    results_folder = os.path.join(FILEDIR, '..', 'evaluatingDPML', 'evaluating_dpml', 'results') if results_folder is None else results_folder
    folders = os.listdir(os.path.join(FILEDIR, '..', 'evaluatingDPML', 'evaluating_dpml', 'results'))

    include_fields = ['train_loss','real_train_loss',
                      'test_loss','real_test_loss',
                      'train_acc','real_train_acc',
                      'test_acc','real_test_acc'] if include_fields is None else include_fields

    full_data = []

    for datafoldername in folders:
        datafolder = os.path.join(results_folder, datafoldername)
        for resultsfoldername in os.listdir(datafolder):
            resultfolder = os.path.join(datafolder, resultsfoldername)
            if os.path.isdir(resultfolder):
                resultfolder = os.path.join(datafolder, resultsfoldername)
                file = os.path.join(resultfolder, 'results.p')
                if not os.path.exists(file):
                    continue
                data = loadResult(file)
                full_data.append({**data['args'],
                                  **{k:v for k,v in data.items() if k in include_fields},
                                  'results_file':file, 'results_foldername':resultsfoldername, 'result_folder':resultfolder})
            else:
                file =resultfolder
                data = loadResult(file)
                full_data.append({**data['args'],
                                  **{k: v for k, v in data.items() if k in include_fields},
                                  'results_file': file, 'results_foldername': resultsfoldername,
                                  'result_folder': datafolder})
    return pd.DataFrame(full_data)

def loadDataIncReal(results):
    train_x, train_y, test_x, test_y = load_data('target_data.npz',
                                                 results['args'],
                                                 PATH_TO_PWD=os.path.join(FILEDIR,
                                                                          '..',
                                                                          'evaluatingDPML',
                                                                          'evaluating_dpml/'))

    real_train_x, real_train_y, real_test_x, real_test_y = load_data('target_data.npz',
                                                                     results['args'].withChange(
                                                                         train_dataset=results['args']['base_dataset']
                                                                     ),
                                                                     PATH_TO_PWD=os.path.join(FILEDIR,
                                                                                              '..',
                                                                                              'evaluatingDPML',
                                                                                              'evaluating_dpml/'))

    true_x = np.vstack((train_x, test_x))
    true_y = np.append(train_y, test_y)

    real_true_x = np.vstack((real_train_x, real_test_x))
    real_true_y = np.append(real_train_y, real_test_y)

    assert true_y.shape == real_true_y.shape and (true_y == real_true_y).all()
    return true_x, real_true_x

@DeprecationWarning
def yeom_mi_attack_old(results:dict,inplace=False):
    ret = results if inplace else {}
    train_x, train_y, test_x, test_y = load_data('target_data.npz', results['args'],PATH_TO_PWD='../evaluatingDPML/evaluating_dpml/')

    true_x = np.vstack((train_x, test_x))
    true_y = np.append(train_y, test_y)
    real_train_x, real_train_y, real_test_x, real_test_y = load_data('target_data.npz', results['args'].withChange(train_dataset=results['args']['base_dataset']),PATH_TO_PWD='../evaluatingDPML/evaluating_dpml/')
    real_true_x = np.vstack((real_train_x, real_test_x))
    real_true_y = np.append(real_train_y, real_test_y)
    assert (true_y==real_true_y).all()

    pred_membership: np.ndarray = yeom_membership_inference(results['per_instance_loss'], results['membership'], results['train_loss'])
    real_pred_membership: np.ndarray = yeom_membership_inference(results['real_per_instance_loss'], results['membership'], results['real_train_loss'])

    ret['naive_yeom_mi'] = getScoreDict(results['membership'], pred_membership)[1] # Originally proposed attack
    ret['naive_yeom_mi_max'] = max(getScoreDict(results['membership'], results['per_instance_loss'], invert=True).values())

    ret['yeom_mi'] = getScoreDict(results['membership'], real_pred_membership)[1] # Just on training-data
    ret['yeom_mi_max'] = max(getScoreDict(results['membership'], results['real_per_instance_loss'], invert=True).values())

    if results['args'].get('method',None) is not None and False:
        distances = ssdistance.cdist(real_true_x, true_x)

        linkage = np.argmax(distances, axis=0)  # find the closes original point to the perturbed
        adv2_pred_membership = np.zeros(real_pred_membership.shape)
        adv2_pred_membership[linkage[np.where(real_pred_membership)]] = 1
        ret['back_linked_yeom_mi'] = getScoreDict(results['membership'], adv2_pred_membership)[1]

        closest_dist = np.min(distances[:, real_pred_membership.astype(bool)], axis=1)

        ret['adapted_yeom_mi_max'] = max(getScoreDict(results['membership'], closest_dist, invert=True).values())
        ret['adapted_yeom_mi_median'] = getScoreDict(results['membership'], closest_dist, invert=True)[np.median(closest_dist)]


    return ret

def yeom_mi_attack(data:Union[pd.DataFrame,Tuple[Hashable,pd.Series]],distances=None,real_true_x=None,true_x=None):
    if isinstance(data,tuple):
        data:pd.DataFrame = data[1]
    print('Starting', data['results_file'])
    start = datetime.datetime.now()
    try:
        results = loadResult(data['results_file'])
        ret = {'attack_version':1}
        if real_true_x is None or true_x is None:
            l_true_x, l_real_true_x, *_ = loadDataIncReal(results)
            del _

            if true_x is None:
                true_x = l_true_x

            if real_true_x is None:
                real_true_x = l_real_true_x
            '''train_x, train_y, test_x, test_y = load_data('target_data.npz', results['args'],
                                                         PATH_TO_PWD=os.path.join(FILEDIR, '..', 'evaluatingDPML',
                                                                                  'evaluating_dpml/'))

            real_train_x, real_train_y, real_test_x, real_test_y = load_data('target_data.npz', results['args'].withChange(
                train_dataset=results['args']['base_dataset']), PATH_TO_PWD=os.path.join(FILEDIR, '..', 'evaluatingDPML',
                                                                                         'evaluating_dpml/'))
            if true_x is None:
                true_x = np.vstack((train_x, test_x))
            true_y = np.append(train_y, test_y)

            if real_true_x is None:
                real_true_x = np.vstack((real_train_x, real_test_x))
            real_true_y = np.append(real_train_y, real_test_y)
            assert true_y.shape == real_true_y.shape and (true_y == real_true_y).all()'''

        pred_membership: np.ndarray = yeom_membership_inference(results['per_instance_loss'], results['membership'], results['train_loss'])
        real_pred_membership: np.ndarray = yeom_membership_inference(results['real_per_instance_loss'], results['membership'], results['real_train_loss'])

        ret['naive_yeom_mi'] = getScoreDict(results['membership'], pred_membership)[1]  # Just on training-data
        naive_opt_thresh, ret['naive_yeom_mi_max'] = max(getScoreDict(results['membership'], results['per_instance_loss'], invert=True).items(), key=lambda x: x[1])
        opt_pred_membership: np.ndarray = (results['per_instance_loss'] <= naive_opt_thresh).astype(int)

        ret['yeom_mi'] = getScoreDict(results['membership'], real_pred_membership)[1]  # Originally proposed attack
        opt_thresh, ret['yeom_mi_max'] = max(getScoreDict(results['membership'], results['real_per_instance_loss'], invert=True).items(), key=lambda x: x[1])
        opt_real_pred_membership = (results['real_per_instance_loss'] <= opt_thresh).astype(int)

        if results['args'].get('method', None) is not None:
            if distances is None:
                distances = ssdistance.cdist(real_true_x, true_x)

            back_linkage = np.argmax(distances, axis=0)  # find the closes original point to the perturbed
            fwd_linkage = np.argmax(distances, axis=1)  # find the closes perturbed point to the original

            def doExtendedAttacks(pred: np.ndarray, prefix=''):
                adv2_pred_membership = np.zeros(pred.shape)
                adv2_pred_membership[back_linkage[np.where(pred)]] = 1
                ret[prefix + 'back_linked_yeom_mi'] = getScoreDict(results['membership'], adv2_pred_membership)[1]

                closest_dist = np.min(distances[:, pred.astype(bool)], axis=1)

                ret[prefix + 'adapted_yeom_mi_max'] = max(
                    getScoreDict(results['membership'], closest_dist, invert=True).values())

                ret[prefix + 'adapted_yeom_mi_median'] = getScoreDict(results['membership'], closest_dist, invert=True)[
                    np.median(np.percentile(closest_dist, (100 / (1 + results['args']['target_test_train_ratio'])),
                                            interpolation='nearest'))]

            doExtendedAttacks(pred_membership)
            doExtendedAttacks(opt_pred_membership, 'opt_')
            del distances
    finally:
        print('Finishing', data['results_file'], datetime.datetime.now()-start)
    return ret

def addYeomMi(data:pd.DataFrame, multiprocessing:Union[bool,int] = True,**kwargs):
    if multiprocessing:
        from multiprocessing import Pool
        with Pool(3) as p:  # , maxtasksperchild=1
            return data.join(pd.DataFrame(p.imap(yeom_mi_attack,data.iterrows()),index=data.index))
    ass = data.apply(yeom_mi_attack, result_type='expand', axis=1,**kwargs)
    return data.join(ass)

def addYeomMiToGroup(data:Union[Tuple[Any,pd.DataFrame],pd.DataFrame]):
    if isinstance(data,tuple):
        data = data[1]
    # Load data once
    results = loadResult(data.iloc[0]['results_file'])
    true_x, real_true_x, *_ = loadDataIncReal(results)
    distances = ssdistance.cdist(real_true_x, true_x)
    print('#### Start',data['data_key'].unique()[0])

    return addYeomMi(data,
                     multiprocessing=False,
                     distances=distances,
                     real_true_x=real_true_x,
                     true_x=true_x)

def addYeomMiEfficient(data:pd.DataFrame, multiprocessing:Union[bool,int] = True,**kwargs):
    if multiprocessing:
        from multiprocessing import Pool
        with Pool(3) as p:  # , maxtasksperchild=1
            data['data_key'] = data.apply(lambda x: x['result_folder'].split('/')[-2],axis=1)
            data = pd.concat(p.imap(addYeomMiToGroup,iter(data.groupby(by=['data_key'], dropna=False))))
            data.drop(columns=['data_key'],inplace=True)
            return data
    ass = data.apply(yeom_mi_attack, result_type='expand', axis=1)
    return data.join(ass)

def addYeomMiToDF(data:pd.DataFrame, min_attack_version=1, multiprocessing:Union[bool,int] = True,**kwargs):
    # Load known values
    if os.path.exists(ATTACKS_RESULTS_FILE):
        data_attacks:pd.DataFrame = pd.read_pickle(ATTACKS_RESULTS_FILE)
        assert 'attack_version' in list(data_attacks.columns)
    else:
        data_attacks:pd.DataFrame =pd.DataFrame(columns=KEY+['attack_version']).astype({"attack_version": float}, errors='raise')


    org_data_column = list(data.columns)

    # Insert the highest known results
    data = mergeUpdate(data,
                       (data_attacks[data_attacks['attack_version'] >= min_attack_version]
                        .groupby(KEY, dropna=False)
                        .apply(lambda group: group.nlargest(1, columns='attack_version'))
                        .reset_index(drop=True)
                        ),
                       on=KEY)

    # Adding missing
    if data['attack_version'].isna().any():
        # generate new values
        new_res:pd.DataFrame = addYeomMiEfficient(data[data['attack_version'].isna()][org_data_column],**kwargs)
        new_res = new_res[KEY+[c for c in list(new_res.columns) if c not in org_data_column]]
        # add missing values to data
        data = mergeUpdate(data,new_res,on=KEY)
        assert (data['attack_version']>=min_attack_version).all()
        # add new results to known results
        data_attacks = pd.concat([data_attacks,new_res], ignore_index=True)
        data_attacks.to_pickle(ATTACKS_RESULTS_FILE)

    return data



def reduceToMostSuccessfull(data: pd.DataFrame,key:str='real_test_acc')->pd.DataFrame:
    return (data
            .groupby(['base_dataset', 'run', 'method', 'target_epsilon', 'target_model', 'target_dp'], dropna=False)
            .apply(lambda group: group.nlargest(1, columns=key))
            .reset_index(drop=True)
            )

def stat_bestK(data:pd.DataFrame, key:str='real_test_acc')->pd.DataFrame:
    return (reduceToMostSuccessfull(data,key=key)
            .dropna(subset=['method'])
            .groupby(['base_dataset', 'method', 'target_epsilon', 'target_model', 'target_dp'], dropna=False)
            .aggregate({'k': ['min', 'median', 'max', 'mean', 'std']})
            .dropna(subset=[('k', 'min')])
            .reset_index()
            )




'''results_folder = os.path.join(FILEDIR,'..','evaluatingDPML','evaluating_dpml','results')
folders = os.listdir(os.path.join(FILEDIR,'..','evaluatingDPML','evaluating_dpml','results'))

FIELD_TO_MAXIMIZE ='test_acc'
remember = {}
for foldername in folders:# Allows for up to 8 number-Parameters
    folder = os.path.join(results_folder, foldername)
    for filename in os.listdir(folder):
        file = os.path.join(folder,filename)
        data = loadResult(file)
        if data['args'].get('method', None) != 'fusion':
            continue
        if data['args'].get('run',0)!=1:
            continue
        if data['args'].get('target_model', None) != 'nn':
            continue
        if data['args']['target_epsilon'] not in remember:
            remember[data['args']['target_epsilon']] = data
            continue
        if data[FIELD_TO_MAXIMIZE]>remember[data['args']['target_epsilon']][FIELD_TO_MAXIMIZE]:
            remember[data['args']['target_epsilon']] = data

for eps, results in sorted(remember.items(),key=lambda x:x[0]):
    print(eps,results['args']['k'],results['real_test_acc'],results['test_acc'])
    print(yeom_mi_attack(results))
'''


def reIntrodiceMODELMETA(attack_file):
    data:pd.DataFrame = pd.read_pickle(AGG_RESULTS_FILE)
    data_in: pd.DataFrame = pd.read_pickle(attack_file)

    com_col = [f for f in list(data.columns) if
               (f in KEY or f not in list(data_in.columns)) and f not in ATTACKER_META and f not in REDUNDANT]
    data_out: pd.DataFrame = pd.merge(data_in, data[com_col], on=KEY)
    assert all(f in list(data_out.columns) for f in list(data_in))
    assert len([f for f in list(data_out.columns) if f not in list(data_in.columns)]) in [0, 6,
                                                                                          7], f'Increase by {len([f for f in list(data_out.columns) if f not in list(data_in.columns)])} ({[f for f in list(data_out.columns) if f not in list(data_in.columns)]})'
    assert data_in.shape[0] == data_out.shape[0]
    data_out.to_pickle(attack_file)





def addSigma(data):
    raise Exception('Sigma should not be retrained!')
    if data['target_privacy'] != 'grad_pert':
        return np.nan
    dp = data['target_dp']
    epochs = data['target_epochs']
    delta = data['target_delta']
    epsilon = data['target_epsilon']
    n = data['target_data_size']
    batch_size = data['target_batch_size']

    if dp == 'adv_cmp':
        sigma = np.sqrt(epochs * np.log(2.5 * epochs / delta)) * (
                np.sqrt(np.log(2 / delta) + 2 * epsilon) + np.sqrt(np.log(2 / delta))) / epsilon
    elif dp == 'zcdp':
        sigma = np.sqrt(epochs / 2) * (np.sqrt(np.log(1 / delta) + epsilon) + np.sqrt(np.log(1 / delta))) / epsilon
    elif dp == 'rdp':
        from paramsearch.sigmaSearcher import get_rdp_sigma
        sigma = get_rdp_sigma(epsilon, n, epochs, batch_size, delta)
        # sigma = rdp_noise_multiplier[epochs][epsilon]
    elif dp == 'gdp':
        from paramsearch.sigmaSearcher import get_gdp_sigma
        sigma = get_gdp_sigma(epsilon, n, epochs, batch_size, delta)
        # sigma = gdp_noise_multiplier[epochs][epsilon]
    else:  # if dp == 'dp'
        sigma = epochs * np.sqrt(2 * np.log(1.25 * epochs / delta)) / epsilon

    print(dp, epsilon, delta, epochs, n, batch_size, '=>', sigma)
    return sigma

def addSigmaLine(line):
    i, data = line
    data['sigma'] = addSigma(data)
    return data

def calcSigmaForDF(data, multi_process = True, **kwargs):
    # Actually calculate Sigma
    if multi_process:
        with multiprocessing.Pool(30) as p:
            data = pd.DataFrame(p.imap(addSigmaLine, data.iterrows()))
    else:
        data['sigma'] = data.apply(addSigma, axis=1)
    return data

def addSigmaToDF(data, recalc_sigma = False,**kwargs):
    KS = ['target_privacy','target_dp','target_epochs','target_delta','target_epsilon','target_data_size','target_batch_size']
    if os.path.exists(SIGMA_RESULTS_FILE) and not recalc_sigma:
        sigmas:pd.DataFrame = pd.read_pickle(SIGMA_RESULTS_FILE)
    else:
        sigmas:pd.DataFrame = pd.DataFrame(columns=KS + ['sigma'])

    needed_sigmas: pd.DataFrame = data[KS]
    needed_sigmas = needed_sigmas[needed_sigmas['target_privacy'] == 'grad_pert']
    needed_sigmas = needed_sigmas.drop_duplicates()
    needed_sigmas = pd.merge(needed_sigmas,sigmas,how='left',on=KS)
    needed_sigmas = needed_sigmas[needed_sigmas['sigma'].isna()]

    if not needed_sigmas.empty:
        raise Exception('Sigma should not be retrained!')
        sigmas = pd.concat([sigmas,calcSigmaForDF(needed_sigmas,**kwargs)], ignore_index=True)
        sigmas.to_pickle(SIGMA_RESULTS_FILE)

    data_out = pd.merge(data,sigmas,how='left',on=KS)
    assert data.shape[0]==data_out.shape[0]
    return data_out

def addSigmaToSavedDF(attack_file,**kwargs):
    data = pd.read_pickle(attack_file)
    data = addSigmaToDF(data,**kwargs)
    data.to_pickle(attack_file)


def getExpData(add_sigma=True,add_attacks=True,recollect=False,savedfile=None,**kwargs):
    savedfile = AGG_RESULTS_FILE if savedfile is None else savedfile
    if recollect or not os.path.exists(savedfile):
        print('Recalculating')
        data = loadResultData()
        data.to_pickle(savedfile)
    else:
        data = pd.read_pickle(savedfile)
    print('Loaded Aggregated data')
    # Drop unneccesary columns
    data = data.drop(columns=[*REDUNDANT,*ATTACKER_META])

    if add_attacks:
        print('Adding Attacks')
        data = addYeomMiToDF(data, **kwargs)

    if add_sigma:
        print('Adding Simgas')
        data = addSigmaToDF(data,**kwargs)

    return data

if __name__ == '__main__':
    start = datetime.datetime.now()
    data = getExpData()
    data.to_pickle(FINAL_EXP_FILE)
    print(data.shape)

    '''start = datetime.datetime.now()
    data =pd.read_pickle(FINAL_EXP_FILE)
    print(data.shape)'''

    print(datetime.datetime.now()-start)
    #data.to_pickle(FINAL_EXP_FILE)
    exit()
    '''for attack_file in [ATTACKER_RESULTS_FILE_HOUSING,ATTACKER_RESULTS_FILE_PURCHASE,ATTACKER_RESULTS_FILE_ADULT]:
        print()
        print('-'*100)
        print(attack_file)
        print('-'*100)
        addSigmaToSavedDF(attack_file)
    exit()'''
    exit(0)
    getExpData(recalc=True).to_pickle(ATTACKER_RESULTS_FILE)
    print('Finished')
    exit(0)

    EXPLANATION = {
        'naive_yeom_mi':'Score of the yeom MIA attempting to retrieve perturbed training data',
        'naive_yeom_mi_max':'Score of the yeom MIA attempting to retrieve perturbed training data knowing optimal threshold',
        'yeom_mi':'Score of the yeom MIA attempting to retrieve original training data',
        'yeom_mi_max':'Score of the yeom MIA attempting to retrieve original training data knowing optimal threshold'
    }

    data  = pd.read_pickle(ATTACKER_RESULTS_FILE)
    data = data[data['target_model'] == 'nn']
    #data = data[(data['method']=='soria')]#|(data['method']=='silentK')
    def plotKVsColumn(data,COLUMN):
        data = (data.groupby(['base_dataset', 'target_epsilon', 'method', 'target_model', 'target_privacy', 'target_dp','k'],dropna=False)
                .aggregate(['max','mean','std'])
                .reset_index())
        #data = data[data['target_epsilon']<=1]
        for d in data.groupby(['base_dataset', 'target_epsilon', 'method', 'target_model', 'target_privacy', 'target_dp'],dropna=False):
            with FullPandasPrinter(cntRows=False):
                print(d[0])
                print(d[1])
                plt.plot(d[1][['k']].to_numpy(),d[1][[(COLUMN,'mean')]].to_numpy(),label=(d[0][2],d[0][1]),linewidth=3 if d[0][2]=='silentK' else 1)
                #d[1].plot(x='k',y=(COLUMN,'mean'),legend=True,xlabel='k',ylabel=COLUMN)
        plt.legend()
        #plt.yscale('log')
        plt.xlabel('k')
        plt.ylabel(COLUMN)
        plt.title(f'{COLUMN} evolving over different values of k for different pert-algo and epsilon')
        plt.show()

    #plotKVsColumn(data[(data['method']=='fusion')|(data['method']=='silentK')],COLUMN='naive_yeom_mi')
    #exit()
    #data = data[(data['method'] == 'soria')]# | data['method'].isna()
    '''data = data[data['method'] != 'soria']
    data = data[data['method'] != 'fusion']
    data = data[data['method'] != 'doca']
    data = data[data['method'] != 'silentK']
    data = data[data['target_privacy']=='no_privacy']
    with FullPandasPrinter(cntRows=False):
        print(data)
    exit()'''
    def genPlotUnPerturbed(data, column='naive_yeom_mi_max'):
        data = data[data['method'].isna()]
        print(data['target_dp'].unique())
        data = data[data['target_privacy']=='grad_pert']
        for dp in data['target_dp'].unique():
            print('=>',dp)
            dt = data[data['target_dp'] == dp]
            dt: pd.DataFrame = (dt
                                .groupby(['base_dataset', 'method', 'target_epsilon', 'target_model', 'target_dp'],
                                         dropna=False)
                                .aggregate(['max', 'mean', 'std'])
                                .reset_index())

            dt = dt.set_index('target_epsilon')
            dt[(column, 'mean')].plot(  # logy=True,
                logx=True,
                xlabel='epsilon',
                ylabel=f'{column} score',
                label=dp)
        plt.title(f'{column} evolving over epsilon for different notions of privacy')
        plt.legend()
        plt.show()

    #genPlotUnPerturbed(data,'naive_yeom_mi_max')
    #exit()

    def addplot(data, column = 'yeom_mi', print_out=False, label = None):

        data: pd.DataFrame = (data
                .groupby(['base_dataset', 'method', 'target_epsilon', 'target_model', 'target_dp'],dropna=False)
                .aggregate(['max','mean','std'])
                .reset_index())


        data = data.set_index('target_epsilon')
        data[(column,'mean')].plot(#logy=True,
                                        logx=True,
                                        xlabel='epsilon',
                                        ylabel=f'{column}',
                                        label = label)
        if print_out:
            with FullPandasPrinter(cntRows=False):
                print(data)
    '''with FullPandasPrinter(cntRows=False):
        print(list(data[data['method']=='silentK'][['run','k','real_test_acc']].groupby(['k']).aggregate(['max','mean','std']).reset_index().columns))
        print(data[data['method']=='silentK'][['run','k','real_test_acc']].groupby(['k']).aggregate(['max','mean','std']).reset_index())
    data[data['method']=='silentK'][['run','k','real_test_acc']].groupby(['k']).aggregate(['max','mean','std']).reset_index()[[('k',''),('real_test_acc', 'mean')]].plot('k',('real_test_acc', 'mean'))
    plt.ylabel('real_test_acc')
    plt.title('Model accuracy without perturbation')
    plt.show()
    exit()'''
    data = reduceToMostSuccessfull(data,key='real_test_acc')

    #COLUMN='adapted_yeom_mi_max'
    #COLUMN='opt_adapted_yeom_mi_max'
    COLUMN='naive_yeom_mi'
    addplot(data = data[(data['method'] == 'fusion')],label='fusion',column=COLUMN,print_out=True)
    addplot(data = data[(data['method'] == 'soria')],label='soria',column=COLUMN,print_out=True)
    #addplot(data = data[(data['method'] == 'silentK')],label='silentK',column='opt_adapted_yeom_mi_max',print_out=True)
    plt.legend()
    plt.title(f'{COLUMN} evolving over epsilon for different pert-algos')
    plt.show()

    '''#data = data.set_index('target_epsilon')
    dd = data.groupby(['target_epsilon','method','target_dp'])
    ddd= dd['real_test_acc']
    print(dd.groups)
    ddd['mean'].plot(logy=True,
                                logx=True,
                                xlabel='epsilon',
                                ylabel='accuracy')
    
    plt.show()'''



    '''with FullPandasPrinter(cntRows=False):
        print(data)'''
        # print(results[['run','method','target_epsilon']])
    '''with FullPandasPrinter(cntRows=False):
        print(data)'''

