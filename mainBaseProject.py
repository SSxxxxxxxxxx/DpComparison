import multiprocessing as mp
from pool import Pool
import time

from evaluatingDPML import chdir_to_evaluating, chdir_to_dataset, evaluatingDPML_path
from DPMLadapter.ArgsObject import ArgsObject
import datetime, os, json, shutil,string, random
import tensorflow as tf


def myrun(*args,**kwargs):
    import tensorflow as tf
    from evaluatingDPML.evaluating_dpml.main import run_experiment
    from evaluatingDPML.core.classifier import CHECKPOINT_DIR
    
    
    # INIT Dir
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    
    random.seed(time.time()*random.random())
    individual_CHECKPOINT_DIR = os.path.join(CHECKPOINT_DIR, ''.join(random.choice(string.ascii_letters+string.digits) for _ in range(20)))

    if not os.path.exists(individual_CHECKPOINT_DIR):
        os.makedirs(individual_CHECKPOINT_DIR)
    
    os.environ['CHECKPOINT_DIR'] = individual_CHECKPOINT_DIR

    # Seeding
    random.seed(0)
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.autograph.set_verbosity(2)
    tf.get_logger().setLevel('ERROR')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    run_experiment(*args,**kwargs)
    
    shutil.rmtree(individual_CHECKPOINT_DIR)
    
if __name__=="__main__":
    # Preprocessing data if necessary
    if not os.path.exists(os.path.join(evaluatingDPML_path,'dataset','transactions_dump.p')) or \
            not os.path.exists(os.path.join(evaluatingDPML_path,'dataset','purchase_100_labels.p')):
        from evaluatingDPML.extra import preprocess_purchase
        print('-' * 10, 'Preprocessing data', '-' * 10)
        chdir_to_dataset() # Adapting current working directory
        if not os.path.exists(os.path.join(evaluatingDPML_path, 'dataset', 'transactions_dump.p')):
            print('Populating...')
            preprocess_purchase.populate()
        print('Making dataset')
        preprocess_purchase.make_dataset()

    # Saving the data to npz
    if not os.path.exists(os.path.join(evaluatingDPML_path,'evaluating_dpml','data','purchase_100','target_data.npz')):
        from evaluatingDPML.core.attack import save_data
        print('-' * 10, 'Saving the data', '-' * 10)
        chdir_to_evaluating() # Adapting current working directory
        save_data(ArgsObject('purchase_100', save_data=1))

    # Running the actual experiments
    print('-' * 10, 'Running the experiments', '-' * 10)
    chdir_to_evaluating() # Adapting current working directory

    print('Check for temp-folder')
    if not os.path.exists('__temp_files'):
        os.makedirs('__temp_files')
    pool = Pool(processes=25)

    print('Change General Settings to limit')
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
    tf.autograph.set_verbosity(2)
    tf.get_logger().setLevel('ERROR')
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    ARGS = {'adapt':True,'genFeatures':False,'yeom_mi':False,'shokri_mi':False}

    pool.apply_async(myrun,(ArgsObject('purchase_100', target_model='softmax', target_l2_ratio=1e-5),),ARGS)
    pool.apply_async(myrun,(ArgsObject('purchase_100', target_model='nn', target_l2_ratio=1e-8),),ARGS)

    #key = None
    for run in range(1, 6):
        for eps in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]:
            for DP in ['dp', 'adv_cmp', 'rdp', 'zcdp']:
                pool.apply_async(myrun, (
                    ArgsObject('purchase_100', target_model='softmax', target_l2_ratio=1e-5, target_privacy='grad_pert', target_dp=DP, target_epsilon=eps,
                               run=run),),ARGS)
                '''print('DP =',DP)
                key = ('purchase_100',run,eps,DP,'softmax')
                if key not in done:
                    done.add(key)
                    setDone(done)
                    print('Starting',key)
                    run_experiment(ArgsObject('purchase_100',target_model='softmax', target_l2_ratio=1e-5,target_privacy='grad_pert', target_dp=DP,target_epsilon=eps, run = run))
                    done = getDone()'''

                pool.apply_async(myrun, (
                    ArgsObject('purchase_100', target_model='nn', target_l2_ratio=1e-8, target_privacy='grad_pert', target_dp=DP, target_epsilon=eps,
                               run=run),),ARGS)
                '''key = ('purchase_100',run,eps,DP,'nn')
                if key not in done:
                    done.add(key)
                    setDone(done)
                    print('Start',key)
                    run_experiment(ArgsObject('purchase_100',target_model='nn', target_l2_ratio=1e-8,target_privacy='grad_pert', target_dp=DP,target_epsilon=eps, run = run))
                    done = getDone()'''
    time.sleep(2)
    print('Wait for finish....')
    pool.wait()
