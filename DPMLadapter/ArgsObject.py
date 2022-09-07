

class ArgsObject(dict):
    DEFAULT = {
        'run': 1,
        'use_cpu': 0,
        'save_model': 0,
        'save_data': 0,
        'n_shadow': 5,
        'target_data_size': int(1e4),
        'target_test_train_ratio': 1,
        'target_model': 'nn',
        'target_learning_rate': 0.01,
        'target_batch_size': 200,
        'target_n_hidden': 256,
        'target_epochs': 100,
        'target_l2_ratio': 1e-8,
        'target_clipping_threshold': 1,
        'target_privacy': 'no_privacy',
        'target_dp': 'dp',
        'target_epsilon': 0.5,
        'target_delta': 1e-5,
        'attack_model': 'nn',
        'attack_learning_rate': 0.01,
        'attack_batch_size': 100,
        'attack_n_hidden': 64,
        'attack_epochs': 100,
        'attack_l2_ratio': 1e-6,
    }

    def __init__(self, train_dataset, **kwargs): #=None
        #assert train_dataset is not None, 'Cannot instantiate '
        param = self.DEFAULT.copy()
        param['train_dataset'] = train_dataset
        param.update(kwargs)
        super().__init__(param)

    def __getattr__(self, item):
        try:
            return self[item]
        except KeyError:
            raise AttributeError
    def copy(self):
        return ArgsObject(**self)
    def withChange(self, **kwargs):
        params = dict(self)
        params.update(kwargs)
        return ArgsObject(**params)
