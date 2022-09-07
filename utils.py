import pandas as pd
import numpy as np
import json

class FullPandasPrinter:
    def __init__(self,cntRows=None,cntColumn=None, reset = False,**kwargs):
        print('__init__', kwargs)
        self.options = {}
        if cntRows != False:
            self.options['display.max_rows']  = {'value':cntRows,'old_value':None}
            #self.options['display.height']  = {'value':None,'old_value':None}
        if cntColumn !=False:
            self.options['display.max_columns']  = {'value':cntColumn,'old_value':None}
            self.options['display.width']  = {'value':None,'old_value':None}

        self.reset = False
    def __enter__(self):
        for key, setting in self.options.items():
            setting['old_value'] = pd.get_option(key)
            pd.set_option(key,setting['value'])
        return self
    def __exit__(self, exc_type, exc_val, exc_tb):
        for key, setting in self.options.items():
            if self.reset:
                pd.reset_option(key)
            else:
                pd.set_option(key,setting['old_value'])

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)