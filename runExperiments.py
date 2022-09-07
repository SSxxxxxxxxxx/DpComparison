import datetime, os, json, shutil,string, random, pickle,inspect
import time

import numpy as np

from DPMLadapter.ArgsObject import ArgsObject
from evaluatingDPML import chdir_to_evaluating, chdir_to_dataset, evaluatingDPML_path
from DPMLadapter.TrainingContext import TrainingContext
from obsv.fusion import fusion
from obsv.kSearcher import get_cluster_sizes_expo
from obsv.soria import soria
from obsv.doca import doca
from obsv.silentK import silentK
from paramsearch.hyperparamsearch import getBest as getBestMLHyperparam
from pool import Pool
from typing import List,Optional,Union

from utils import NpEncoder

TIMES = 5 # Number of times each configuration is run
KNOWN_LOG_FIELD = {'adult':['capital-gain'],'diabetes':['insulin'],'purchase_100':[], 'housing':['total_rooms','total_bedrooms','population','households']}
KNOWN_DROP_FIELD = {'adult':[],'diabetes':[],'purchase_100':[],'housing':['ocean_proximity']}
#PERTUBATION_METHODS = ['soria','fusion']

LABLE_COUNTCACHE = {'adult':20,'irishn':22,'diabetes':10,'housing':5,'purchase_100':100} # 12 is another option for adult
LOGFOLDER = 'logs-purchase_100-run'

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
# https://stackoverflow.com/questions/35911252/disable-tensorflow-debugging-information
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
def perturbedDataName(args):
    method = args.method
    if method == 'soria':
        return f'{args.train_dataset}_{method}_{args.target_epsilon}_{args.k}'
    elif method == 'fusion':
        return f'{args.train_dataset}_{method}_{args.target_epsilon}_{args.k}'
    elif method == 'silentK':
        return f'{args.train_dataset}_{method}_{args.k}'
    elif method == 'doca':
        return f'{args.train_dataset}_{method}_{args.target_epsilon}_{args.doca_delay_constraint}_{args.doca_beta}_{args.doca_mi}'
    else:
        raise Exception(f'Does not know method "{method}"')

def getLoggingFile(args:ArgsObject):
    if args.get('method',None) is None:
        if args.target_privacy == 'no_privacy':
            return f'{args.train_dataset}_{args.target_privacy}_{args.target_model}_{args.run}'
        return f'{args.train_dataset}_{args.target_dp}_{args.target_epsilon}_{args.target_model}_{args.run}'
    return f'{perturbedDataName(args)}_{args.target_model}_{args.run}'

def adoptArgsToPerturbedData(args:ArgsObject):
    method = args.get('method',None)
    if method is None:
        return args
    dataset = args.train_dataset
    args['base_dataset'] = dataset
    args['label_dataset'] = dataset
    args['source_dataset'] = dataset
    args['train_dataset'] = perturbedDataName(args)
    return args

def ensureBaseDataset(args:ArgsObject, **kwargs):
    dataset = args.train_dataset
    if not os.path.exists(os.path.join(evaluatingDPML_path, 'dataset', f'{dataset}_features.p')):
        if not os.path.exists(os.path.join(evaluatingDPML_path, 'dataset', f'{dataset}.csv')):
            raise FileNotFoundError(f'Could neither find the dataset "{dataset}_features.p" nor the "{dataset}.csv" to generate it.')

        # Create labels
        from createDataset import createDatasetFromCSV
        createDatasetFromCSV(os.path.join(evaluatingDPML_path, 'dataset', f'{dataset}.csv'),**kwargs)

    # Saving the data to npz
    if not os.path.exists(os.path.join(evaluatingDPML_path, 'evaluating_dpml', 'data', dataset, 'target_data.npz')):
        print('Creating labels and saving it in the right folder...')
        from evaluatingDPML.core.attack import save_data
        print('-' * 10, 'Saving the data', '-' * 10)
        chdir_to_evaluating()  # Adapting current working directory
        args2: ArgsObject = args.copy()
        args2['save_data'] = 1
        save_data(args2, perturbation_func=createPerturbeDataFunc(args2),create_shadow_dataset=False)

def createPerturbedData_old(args:ArgsObject):
    if args.get('method',None) is None:
        return
    print(f'Ensuring that PERTURBED dataset ({args.method}) is present in the right format at the right places...')
    if not os.path.exists(os.path.join(evaluatingDPML_path, 'dataset', f'{args.train_dataset}_features.p')):
        if not os.path.exists(os.path.join(evaluatingDPML_path, 'dataset', f'{args.base_dataset}_features.p')):
            raise FileNotFoundError(f'Cannnot find "{args.base_dataset}" to perturb.')
        X = pickle.load(open(os.path.join(evaluatingDPML_path,'dataset', f'{args.base_dataset}_features.p'), 'rb'))
        if args.method == 'fusion':
            X:np.ndarray = fusion(X,args.target_epsilon,args.k)
        elif args.method == 'soria':
            X:np.ndarray = soria(X,args.target_epsilon,args.k)
        elif args.method == 'silentK':
            X:np.ndarray = silentK(X,args.k)
        elif args.method == 'doca':#doca_delay_constraint=1000,doca_beta=50,doca_mi=100
            X:np.ndarray = doca(X,args.target_epsilon,
                                delay_constraint=args.doca_delay_constraint,
                                beta=args.doca_beta,
                                mi=args.doca_mi)
        else:
            raise ValueError(f'Does not recognise perturbation method "{args.method}"')
        with open(os.path.join(evaluatingDPML_path, 'dataset', f'{args.train_dataset}_features.p'), 'wb') as f:
            pickle.dump(X,f)

def createPerturbeDataFunc(args:ArgsObject):
    if args.get('method',None) is None:
        return lambda x:x
    elif args.method == 'fusion':
        return (lambda eps,k: lambda X:fusion(X,epsilon=eps,k=k))(args.target_epsilon,args.k)
    elif args.method == 'soria':
        return (lambda eps,k: lambda X:soria(X,epsilon=eps,k=k))(args.target_epsilon,args.k)
    elif args.method == 'silentK':
        return (lambda k: lambda X:silentK(X,k=k))(args.k)
    elif args.method == 'doca':#doca_delay_constraint=1000,doca_beta=50,doca_mi=100
        return (lambda eps,delay,beta,mi: lambda X: doca(X,
                                                         eps=eps,
                                                         delay_constraint=delay,
                                                         beta=beta,
                                                         mi=mi))(args.target_epsilon,
                                                                 args.doca_delay_constraint,
                                                                 args.doca_beta,
                                                                 args.doca_mi)
    else:
        raise ValueError(f'Does not recognise perturbation method "{args.method}"')

def createPerturbedData(args:ArgsObject):
    if args.get('method',None) is None:
        return
    from evaluatingDPML.core.attack import load_data
    print(f'Ensuring that PERTURBED dataset ({args.method}) is present in the right format at the right places...')
    out_file = os.path.join(evaluatingDPML_path,'evaluating_dpml', 'data', args.train_dataset, 'target_data.npz')
    if not os.path.exists(out_file):
        if not os.path.exists(os.path.join(evaluatingDPML_path,'evaluating_dpml', 'data', args.base_dataset, 'target_data.npz')):
            raise FileNotFoundError(f'Cannnot find "{args.base_dataset}" to perturb.')
        chdir_to_evaluating()
        train_x, train_y, test_x, test_y = load_data('target_data.npz',args.withChange(train_dataset=args['base_dataset']))

        pert_func = createPerturbeDataFunc(args)
        train_x = pert_func(train_x)
        test_x = pert_func(test_x)
        folder = os.path.join(evaluatingDPML_path,'evaluating_dpml', 'data', args.train_dataset)
        if not os.path.exists(folder):
            os.makedirs(folder)
        np.savez(os.path.join(evaluatingDPML_path,'evaluating_dpml', 'data', args.train_dataset, 'target_data.npz'), train_x, train_y, test_x, test_y)



def ensureDataset(args:ArgsObject,**kwargs):
    print('Ensuring that dataset is present in the right format at the right places...')

    # Create "labels.p"
    ensureBaseDataset(args, num_lables=args.num_lables, **kwargs)
    if args.get('method',None) is not None:
        # Change args to reflect
        adoptArgsToPerturbedData(args)
        createPerturbedData(args)


def getBestMLparams(dataset, model='nn'):
    print(f'Loading hyperparameters {dataset=} {model=}')
    file = os.path.join(PROJECT_FOLDER,'bestParams.json')
    if os.path.exists(file):
        bestMLargs = json.load(open(file))
    else:
        bestMLargs ={}

    if dataset not in bestMLargs:
        bestMLargs[dataset] = {}

    if model in bestMLargs[dataset]:
        return bestMLargs[dataset][model]
    bestMLargs[dataset][model] = getBestMLHyperparam(model=model,dataset_name=dataset)
    json.dump(bestMLargs,open(file,'w'))
    return bestMLargs[dataset][model]

def getDatasetSize(dataset:str):
    print('Find size of data for later calculations...')
    filename = os.path.join(evaluatingDPML_path, 'dataset', f'{dataset}_features.p')
    if os.path.exists(filename):
        data:np.ndarray = np.load(filename,allow_pickle=True)
        assert isinstance(data,np.ndarray), f'{filename} is no np.ndarray'
        return data.shape[0]
    filename = os.path.join(evaluatingDPML_path, 'dataset', f'{dataset}.csv')
    if os.path.exists(filename):
        # Load data
        import pandas as pd
        data:pd.DataFrame = pd.read_csv(filename)
        # Make dataset unique
        data = data.drop_duplicates()
        data = data.dropna()
        return data.shape[0]
    raise FileNotFoundError(f'Could not find the dataset "{dataset}"')

def myrun(args:ArgsObject,*other_args, **kwargs):
    import sys
    print('Starting:',getLoggingFile(args), datetime.datetime.now())
    if not os.path.exists(os.path.join(PROJECT_FOLDER,LOGFOLDER)):
        os.mkdir(os.path.join(PROJECT_FOLDER,LOGFOLDER))
    sys.stdout = open(os.path.join(PROJECT_FOLDER, LOGFOLDER, f'{getLoggingFile(args)}-stdout.log'), 'a',buffering=1)
    sys.stderr = open(os.path.join(PROJECT_FOLDER, LOGFOLDER, f'{getLoggingFile(args)}-stderr.log'), 'a',buffering=1)
    print('Starting with', args)
    from evaluatingDPML.evaluating_dpml.main import run_experiment
    ensureDataset(args)

    start = datetime.datetime.now()
    with TrainingContext(keepCWD=True,info=json.dumps(dict(args),cls=NpEncoder)):
        run_experiment(args, *other_args, **kwargs)
        pass
    print('Finished with', args,'after',datetime.datetime.now()-start)


def runExperimentForDataset(dataset:str,num_lables=None,log_fields=None,drop_fields=None, total_runs=5, start_runs_at=1, epsilon_options:List[float]=None):
    if num_lables is None:
        if dataset in LABLE_COUNTCACHE:
            num_lables = LABLE_COUNTCACHE[dataset]
        else:
            print('WARNING: Number of lables where not specified nor is there a value specified in the cache. Using 100 instead.')
            num_lables = 100
    print(f'Number of labels determined to be {num_lables}')

    if log_fields is None:
        log_fields = KNOWN_LOG_FIELD.get(dataset)
        print('Following field will be logged:',log_fields)

    if drop_fields is None:
        drop_fields = KNOWN_DROP_FIELD.get(dataset)
        print('Following field will be dropped:',drop_fields)

    # Calc datsize fto determin fitting ks
    num_datapoints = getDatasetSize(dataset)

    # Create Data for parametersearch
    assert dataset not in ['irishn','adult'] or num_datapoints<30_000
    ensureDataset(ArgsObject(dataset, num_lables=num_lables,target_data_size=int(10_000),target_test_train_ratio=1), log_fields=log_fields,drop_fields=drop_fields) #TODO num_datapoints/2

    # Load best ML-params
    nn_bestArgs = getBestMLparams(dataset,'nn')
    softmax_bestArgs = getBestMLparams(dataset,'softmax')


    print('Init Pool')
    pool = Pool(processes=30)
    if epsilon_options is None:
        epsilon_options = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]

    ARGS = {} # {'adapt':True,'genFeatures':True,'yeom_mi':True,'shokri_mi':True}

    def runAsycnWithDefaults(**kwargs):#target_l2_ratio=1e-5
        #myrun(ArgsObject(dataset,num_lables=num_lables, log_fields=log_fields,target_data_size=10_000,target_test_train_ratio=1, **kwargs),**ARGS)#int(num_datapoints/2)
        pool.apply_async(myrun,(ArgsObject(dataset,num_lables=num_lables, log_fields=log_fields,target_data_size=int(10_000),target_test_train_ratio=1, **kwargs),),ARGS) #TODO num_datapoints/2

    def runAsyncWithBothDefaults(**kwargs):
        runAsycnWithDefaults(target_model='softmax',**softmax_bestArgs,**kwargs)
        runAsycnWithDefaults(target_model='nn',**nn_bestArgs,**kwargs)


    for run in range(start_runs_at,total_runs+1):
        runAsyncWithBothDefaults(run=run)
        for eps in epsilon_options:
            for DP in ['dp', 'adv_cmp', 'rdp', 'zcdp']:
                runAsyncWithBothDefaults(target_privacy='grad_pert', target_dp=DP, target_epsilon=eps,run=run)
                pass
            for method in ['soria', 'fusion']:
                for k in get_cluster_sizes_expo(num_datapoints,num_lables,10):
                    runAsyncWithBothDefaults(method=method, target_epsilon=eps, k=k, run=run)
                    pass
            runAsyncWithBothDefaults(method='doca', target_epsilon=eps, doca_delay_constraint=1000,doca_beta=50,doca_mi=100, run=run)
        for k in get_cluster_sizes_expo(num_datapoints, num_lables, 10):
            runAsyncWithBothDefaults(method='silentK',k=k,run=run)
    print('Wait for finish....')
    pool.wait()
    print('Finished', datetime.datetime.now())

if __name__ == '__main__':
    runExperimentForDataset('housing', start_runs_at=6, total_runs=10)