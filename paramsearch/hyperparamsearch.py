import optuna
if __name__=='__main__':
    import sys,os
    print('Adding')
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from DPMLadapter.ArgsObject import ArgsObject


def getBest(model='nn',dataset_name='purchase_100', n_trials=100):
    print('-'*100)
    print(f'Running Hyperparametersearch for {dataset_name} ({model}) with {n_trials} trials')
    print('-'*100)
    print()
    from evaluatingDPML.core.attack import load_data
    from DPMLadapter.TrainingContext import TrainingContext
    def objective(trial: optuna.Trial):
        from evaluatingDPML.core.attack import train_target_model
        with TrainingContext(keepCWD=False):
            args = ArgsObject(dataset_name,
                              target_model=model,
                              target_epochs=trial.suggest_int('target_epochs',5,100),
                              target_n_hidden=trial.suggest_int('target_n_hidden',2,400),
                              target_l2_ratio=trial.suggest_float('target_l2_ratio',1e-10,1e-4,log=True),
                              target_learning_rate=trial.suggest_float('target_learning_rate',0.0001,0.1,log=True),
                              target_batch_size=trial.suggest_int('target_batch_size',20,100),
                              save_model=False)
            dataset = load_data('target_data.npz', args)
            pred_y, membership, test_classes, classifier, aux = train_target_model(
                args=args,
                dataset=dataset,
                epochs=args.target_epochs,
                batch_size=args.target_batch_size,
                learning_rate=args.target_learning_rate,
                clipping_threshold=args.target_clipping_threshold,
                n_hidden=args.target_n_hidden,
                l2_ratio=args.target_l2_ratio,
                model=args.target_model,
                privacy=args.target_privacy,
                dp=args.target_dp,
                epsilon=args.target_epsilon,
                delta=args.target_delta,
                save=args.save_model)
        train_loss, train_acc, test_loss, test_acc = aux
        return train_acc

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=n_trials)

    print()
    print(f'Found best params: ({dataset_name=}, {model=}, {n_trials=})')
    print(study.best_params)
    print()

    return study.best_params


if __name__ == '__main__':
    getBest('nn','purchase_100')
