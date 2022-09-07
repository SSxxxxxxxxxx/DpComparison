import functools

import pandas as pd
import os, sys
from matplotlib import pyplot as plt
import matplotlib.colors
import numpy as np
from typing import Tuple

FILEDIR = os.path.dirname(__file__)
if __name__ == '__main__':
    print('Adding to path')
    sys.path.append(os.path.join(FILEDIR, '..'))

from evaluation.attackOnResults import reduceToMostSuccessfull
from evaluation import graphics
from utils import FullPandasPrinter

# Generate data
ATTACKER_RESULTS_FILE_PURCHASE = os.path.join(FILEDIR, '..', 'savedResultsPurchase_100.pcl.gz')
ATTACKER_RESULTS_FILE_HOUSING = os.path.join(FILEDIR, '..', 'savedResultsHousing.pcl.gz')
ATTACKER_RESULTS_FILE_ADULT = os.path.join(FILEDIR, '..', 'savedResultsAdult.pcl.gz')

SIGMA_RESULTS_FILE = os.path.join(FILEDIR, '..', 'savedSigmas.pcl.gz')

FINAL_EXP_FILE = os.path.join(FILEDIR, '..', 'savedFinalExp.pcl.gz')


'''data_purchase_100 = pd.read_pickle(ATTACKER_RESULTS_FILE_PURCHASE)

data_housing = pd.read_pickle(ATTACKER_RESULTS_FILE_HOUSING)
data_housing = data_housing.drop(columns=['model_dirname'])

data_adult = pd.read_pickle(ATTACKER_RESULTS_FILE_ADULT)
data_adult = data_adult.drop(columns=['model_dirname'])

data = pd.concat([data_purchase_100, data_housing, data_adult], ignore_index=True)
'''
data = pd.read_pickle(FINAL_EXP_FILE)

# Output :D
def lesc(s: str):
    return s.replace('_', '-')


def addPic(fig: plt.Figure, name=None, show=False,folders = None):
    if name is not None:
        folders = ['output'] if folders is None else folders
        print(os.path.join(FILEDIR, *folders, f'{name}.svg'))
        os.makedirs(os.path.join(FILEDIR,*folders),exist_ok=True)
        fig.savefig(os.path.join(FILEDIR, *folders, f'{name}.svg'),bbox_inches='tight')
    if show:
        fig.show()
    plt.close(fig)


VARS = {}


def flushVars():
    with open(os.path.join(FILEDIR, 'output', 'results.dat'), 'w') as f:
        for key, val in VARS.items():
            f.write(f'{key.strip()}|@->{val.strip()}\n')


def addVars(value, name, flush=False, print_out=False, **kwargs):
    global VARS
    name = lesc(name)
    if name in VARS:
        raise ValueError(f'Variable with name \'{name}\' already exists')
    VARS[name] = str(value)
    if print_out:
        print(name, '=>', value)
    if flush:
        flushVars()


def addFloatVar(value, name, **kwargs):
    addVars(f'${value:0.4f}$', name, **kwargs)
    addVars(f'{value:0.4f}', name + '|raw', **kwargs)


def addIntVar(value, name, **kwargs):
    addVars(f'${value}$', name, **kwargs)
    addVars(str(value), name + '|raw', **kwargs)


def addDfVars(data: pd.DataFrame, names=None, values=None, keys=None, **kwargs):
    assert values is not None, 'One column must contain the values to be included!'
    if not isinstance(values, list):
        values = [values]
        names = [names]

    if keys is None:
        keys = list(sorted(c for c in list(data.columns) if c not in values))
    for i, value in enumerate(values):
        if isinstance(value, str):
            k_name = value
        elif isinstance(value, tuple):
            k_name = '|'.join(value)
        else:
            raise ValueError(f'Could not parse value {value}')

        if names is not None and i < len(names) and names[i] is not None:
            name = names[i]
        else:
            name = f'{k_name}>{"/".join(f"%({key})s" for key in keys)}'
        assert isinstance(name, str)

        for _, series in data.iterrows():
            addFloatVar(series[value], name % series[keys].to_dict())


# Prepare aggregated data
def colmap(col):
    if not col[1]:
        return col[0]
    return col

def colGrad(col,data = None,mx=None,mn=None, nan='g'):
    if mx is None:
        mx = data[col].max()
    if mn is None:
        mn = data[col].min()
    def ret(*args,data:pd.DataFrame=None,**kwargs):
        assert len(data[col].unique())==1
        val = data[col].iloc[0]
        if np.isnan(val):
            return nan
        color = matplotlib.colors.LinearSegmentedColormap.from_list("", ["red","violet","blue"])(np.log(val/mn)/np.log(mx/mn))
        return color
    return ret

def plotAxesSigmaWithEps(data,*args,ax:plt.Axes=None,**kwargs):

    ret = graphics.plotOverLabels(data,*args,ax=ax,**kwargs)
    #sigma = epochs * np.sqrt(2 * np.log(1.25 * epochs / delta)) / epsilon

    def funcs(data:pd.DataFrame):
        #Get num Epochs
        epochs = data['target_epochs'].unique()
        assert len(epochs)==1
        epochs = epochs[0]

        #Get num delta
        delta = data['target_delta'].unique()
        assert len(delta)==1
        delta = delta[0]



        def foward(x):
            return epochs * np.sqrt(2 * np.log(1.25 * epochs / delta))/x
        def invers(x):
            return epochs * np.sqrt(2 * np.log(1.25 * epochs / delta))/x
        return (foward, invers)


    secax = ax.secondary_xaxis('top', functions=funcs(data))
    secax.set_xlabel('naive composition $\epsilon$')

    return ret

data_red = reduceToMostSuccessfull(data, key='real_test_acc')
data_red_agg: pd.DataFrame = (data_red
                              # .dropna(subset=['result_folder', 'results_foldername', 'results_file',])
                              .drop(columns=['train_dataset', 'log_fields', 'label_dataset', 'source_dataset', 'results_file', 'results_foldername', 'result_folder', 'model_dirname'],errors='ignore')
                              .groupby(['base_dataset', 'method', 'target_epsilon', 'target_model', 'target_privacy', 'target_dp'], dropna=False)
                              .aggregate(['min', 'mean', 'max', 'std'])
                              # .rename(columns=colmap, inplace=True)
                              .reset_index()
                              )
data_red_agg.columns = np.array(list(map(colmap, list(data_red_agg.columns))), dtype=object)

# reduce sigma:
for field in ['sigma','target_epochs','target_delta']:
    assert data_red_agg[(field, 'min')].equals(data_red_agg[(field, 'max')]), f'Could not collapse {field} due to different values'
    data_red_agg[field] = data_red_agg[(field, 'min')]
    data_red_agg = data_red_agg.drop(columns=[f for f in list(data_red_agg.columns) if isinstance(f, tuple) and f[0] == field])

def getOptK():
    housing = [(
    [2, 5, 10, 20, 40, 60, 80, 100, 150, 200, 250, 300, 400, 500],[0.20120664875327302, 0.23661318943430512, 0.19301234450949714, 0.1795426457837274, 0.1676716189298121, 0.15871419320270233, 0.15677075472343138, 0.1513664455336867, 0.14975813149684664, 0.14605359120400901, 0.1441115445598599, 0.14254043827338583, 0.1435753107496716, 0.14258100904718862]
    ),(
    [2, 3, 4, 5, 6, 7, 8, 9, 10],[0.20120664875327302, 0.23224719086545703, 0.23098501727212012, 0.23661318943430512, 0.22605910960410366, 0.19678163780896105, 0.20407511677842224, 0.19315107277984064, 0.19301234450949714]
    )]
    adult=[(
    [2, 5, 10, 20, 40, 60, 80, 100, 150, 200, 250, 300, 400, 500], [0.21941534743745936, 0.2494358579432236, 0.275866992131926, 0.27621801256723744, 0.26562481552044515, 0.2621114940147862, 0.26530622168316126, 0.25516602910710234, 0.2541497256881973, 0.2439703317103089, 0.24019147903144009, 0.24284687383891465, 0.24236233041428695, 0.24428818664884355]
    ),(
    [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100], [0.2494358579432236, 0.275866992131926, 0.26089827774738233, 0.27621801256723744, 0.265657224329223, 0.26363818913961273, 0.26116031555009656, 0.26562481552044515, 0.2641439006341046, 0.26553564935407054, 0.2643960297692998, 0.2621114940147862, 0.26712445428843534, 0.26173298568302517, 0.25567166534173474, 0.26530622168316126, 0.26326931475597626, 0.26086100913262134, 0.25211935009921793, 0.25516602910710234]
    ),(
    [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30], [0.2494358579432236, 0.2419151520757351, 0.2629759059727314, 0.26147237908429255, 0.2729684534771858, 0.275866992131926, 0.25381252502130863, 0.27850741103607285, 0.2481627603484711, 0.2590989703731599, 0.26089827774738233, 0.2722245784114261, 0.26753664663976107, 0.2744970217136446, 0.26949054920960636, 0.27621801256723744, 0.25982716933452293, 0.26432625520789915, 0.26644082197967045, 0.2634750826775752, 0.265657224329223, 0.26443544247230893, 0.26143015401250524, 0.26438631679866725, 0.2630699236432354, 0.26363818913961273]
    )]
    results = {}
    for dataset, data in {'housing':housing, 'adult':adult}.items():
        data_agg = {}
        for d_k, d_v in data:
            data_agg.update({k:v for k,v in zip(d_k,d_v)})
        results[dataset] = data_agg

    def plotKs(ax:plt.Axes, data:dict, mark=None,name:str=None):
        df = pd.DataFrame(data.items(),columns=['n','score'])
        df = df.sort_values('n')
        ax.plot(df['n'], df['score'], 'bx-')
        ax.set_xlabel('number of clusters $n$')
        ax.set_ylabel('Silhouette score')
        ax.set_xscale('log')
        if name:
            ax.set_title(graphics.dName(name))
        if mark is not None:
            if isinstance(mark,tuple):
                x, y = mark
            else:
                x = mark
                y = data[x]
            ax.annotate(str(x), xy=(x, y), xytext=(1, 4), color='red', textcoords='offset points')
            ax.plot(x, y, 'rx')

    LABLE_COUNTCACHE = {'adult': 20, 'irishn': 22, 'diabetes': 10, 'housing': 5, 'purchase_100': 100}  # 12 is another option for adult

    fig: plt.Figure = plt.figure(figsize=(len(results)*4, 4), constrained_layout=True)
    subplots = fig.subplots(1,len(results))
    fig.suptitle('Silhouette analysis For optimal cluster number $n$')
    for ax, (datset, data) in zip(subplots, results.items()):
        plotKs(ax,data,mark=LABLE_COUNTCACHE[datset],name=datset)
    return fig





def sec1Accuracy():
    # Calc no-privacy
    with FullPandasPrinter():
        print('No-Privacy performance:')
        data_no_privacy = data_red_agg[data_red_agg['method'].isna() & (data_red_agg['target_privacy'] == 'no_privacy')]
        print(data_no_privacy[['base_dataset', 'target_model', ('real_train_acc', 'mean'), ('real_test_acc', 'mean')]])
        addDfVars(data_no_privacy[['target_privacy', 'base_dataset', 'target_model', ('real_train_acc', 'mean'), ('real_test_acc', 'mean')]], values=[('real_train_acc', 'mean'), ('real_test_acc', 'mean')])

    '''
    # Graph over pert_al
    graphics.figOverPLotsX(data_red_agg[(~data_red_agg['method'].isna()) & (data_red_agg['target_privacy']=='no_privacy') & (data_red_agg['target_model']=='nn')],
                           x_axes_key='base_dataset',
                           label_key='method',
                           x_axis='target_epsilon',
                           y_axis=('real_test_acc','mean')).show()'''

    addPic(graphics.figOverSubfigsXY(data_red_agg[(~data_red_agg['method'].isna()) & (data_red_agg['target_privacy'] == 'no_privacy')],
                                     y_axes_key='target_model',
                                     x_axes_key='base_dataset',
                                     label_values_key='method',
                                     x_axis='target_epsilon',
                                     y_axis=('real_test_acc', 'mean')),
           name='pertAlgAccOverEps',
           show=False)

    # sigma = epochs * np.sqrt(2 * np.log(1.25 * epochs / delta))
    addPic(graphics.figOverSubfigsXY(data_red_agg[(data_red_agg['method'].isna()) & (data_red_agg['target_privacy'] == 'grad_pert')],
                                     y_axes_key='target_model',
                                     x_axes_key='base_dataset',
                                     label_values_key='target_dp',
                                     x_axis='sigma',
                                     y_axis=('real_test_acc', 'mean'),
                                     plotAxes_function=plotAxesSigmaWithEps,),
           name='privLearnAccOverSigma',
           show=False)

    data_fusion_to_soria = (data_red_agg[(data_red_agg['method'] == 'soria') | (data_red_agg['method'] == 'fusion')]
                            .groupby(['base_dataset', 'target_epsilon', 'target_model', 'target_privacy', 'target_dp'], dropna=False)
                            .apply(lambda x: pd.Series({'soriaPerFusionOverEps': (x[x['method'] == 'soria'].iloc[-1][('real_test_acc', 'mean')] / x[x['method'] == 'fusion'].iloc[-1][('real_test_acc', 'mean')])}))
                            .groupby(['base_dataset', 'target_model', 'target_privacy', 'target_dp'], dropna=False)
                            .agg(soriaPerFusion=pd.NamedAgg(column='soriaPerFusionOverEps', aggfunc='mean'))
                            .reset_index())
    print('For context: Soria outperforming Fusion')
    with FullPandasPrinter():
        print(data_fusion_to_soria)
    print_out = False
    print('Smallest outperform')
    smallest_soria_fusion_outperform = data_fusion_to_soria.loc[data_fusion_to_soria['soriaPerFusion'].idxmin()]
    addVars(smallest_soria_fusion_outperform['target_model'], name='soriaPerFusion|min>target_model', print_out=print_out)
    addVars(smallest_soria_fusion_outperform['base_dataset'], name='soriaPerFusion|min>base_dataset', print_out=print_out)
    addFloatVar(1 - smallest_soria_fusion_outperform['soriaPerFusion'], name='soriaPerFusion|min>value', print_out=print_out)
    del smallest_soria_fusion_outperform
    print('Largest outperform')
    largest_soria_fusion_outperform = data_fusion_to_soria.loc[data_fusion_to_soria['soriaPerFusion'].idxmax()]
    addVars(largest_soria_fusion_outperform['target_model'], name='soriaPerFusion|max>target_model', print_out=print_out)
    addVars(largest_soria_fusion_outperform['base_dataset'], name='soriaPerFusion|max>base_dataset', print_out=print_out)
    addFloatVar(1 - largest_soria_fusion_outperform['soriaPerFusion'], name='soriaPerFusion|max>value', print_out=print_out)
    del largest_soria_fusion_outperform
    del data_fusion_to_soria

    data_better_softmax: pd.DataFrame = (data_red_agg[(data_red_agg['method'].isna()) & (data_red_agg['target_privacy'] == 'grad_pert')]
                                         .groupby(['base_dataset', 'method', 'target_epsilon', 'target_privacy', 'target_dp'], dropna=False)
                                         .apply(lambda x: pd.Series({'softmaxPerNnOverSigma': (x[x['target_model'] == 'softmax'].iloc[0][('real_test_acc', 'mean')] / x[x['target_model'] == 'nn'].iloc[0][('real_test_acc', 'mean')])}))
                                         .groupby(['base_dataset', 'method', 'target_privacy'], dropna=False)
                                         .agg(softmaxPerNnAccuracy=pd.NamedAgg(column='softmaxPerNnOverSigma', aggfunc='mean'))
                                         .reset_index())
    print('For context: Softmax better than nn')
    with FullPandasPrinter():
        print(data_better_softmax)
        addDfVars(data_better_softmax[['base_dataset', 'softmaxPerNnAccuracy']], values=['softmaxPerNnAccuracy'])


def sec2Attackers():
    # Calc no-privacy
    with FullPandasPrinter():
        print('No-Privacy performance:')
        data_no_privacy = data_red_agg[data_red_agg['method'].isna() & (data_red_agg['target_privacy'] == 'no_privacy')]
        print(data_no_privacy[['base_dataset', 'target_model', ('naive_yeom_mi', 'mean'), ('naive_yeom_mi', 'std'), ('naive_yeom_mi_max', 'mean'), ('naive_yeom_mi_max', 'std')]])
        addDfVars(data_no_privacy[['target_privacy', 'base_dataset', 'target_model', ('naive_yeom_mi', 'mean'), ('naive_yeom_mi', 'std'), ('naive_yeom_mi_max', 'mean'), ('naive_yeom_mi_max', 'std')]],
                  values=[('naive_yeom_mi', 'mean'), ('naive_yeom_mi', 'std'), ('naive_yeom_mi_max', 'mean'), ('naive_yeom_mi_max', 'std')])
    addFloatVar(data[data['method'].isna() & (data['target_privacy'] == 'no_privacy') & (data['base_dataset'] != 'purchase_100')]['naive_yeom_mi_max'].max(),
                name='naive-yeom-mi-max|max>purchase-100/softmax/no-privacy')

    # private Learning on purchase
    fig: plt.Figure = plt.figure(figsize=(4 * 4, 4), constrained_layout=True)
    subfigs = fig.subfigures(nrows=1, ncols=2)
    graphics.figOverPotsX(data_red_agg[(data_red_agg['target_privacy'] == 'grad_pert') & (data_red_agg['base_dataset'] == 'purchase_100')],
                          x_axes_key='target_model',
                          label_values_key='target_dp',
                          x_axis='target_epsilon',
                          y_axis='naive_yeom_mi',
                          plot_function=graphics.plt_errbar_agg,
                          fig=subfigs[0],
                          )
    graphics.figOverPotsX(data_red_agg[(data_red_agg['target_privacy'] == 'grad_pert') & (data_red_agg['base_dataset'] == 'purchase_100')],
                          x_axes_key='target_model',
                          label_values_key='target_dp',
                          x_axis='sigma',
                          y_axis='naive_yeom_mi',
                          plot_function=graphics.plt_errbar_agg,
                          fig=subfigs[1],
                          plotAxes_function=plotAxesSigmaWithEps,
                          )
    addPic(fig,
           name='naiveYeomPurchase100OverviewPrivateLearning',
           show=False)
    del fig,subfigs

    #For Appendix: Compare purchase_100 opt. vs normal yMIA
    fig: plt.Figure = plt.figure(figsize=(4 * 4, 4), constrained_layout=True)
    subfigs = fig.subfigures(nrows=1, ncols=2)
    graphics.figOverPotsX(data_red_agg[(data_red_agg['target_privacy'] == 'grad_pert') & (data_red_agg['base_dataset'] == 'purchase_100')],
                          x_axes_key='target_model',
                          label_values_key='target_dp',
                          x_axis='sigma',
                          y_axis='naive_yeom_mi',
                          plot_function=graphics.plt_errbar_agg,
                          fig=subfigs[0],
                          plotAxes_function=plotAxesSigmaWithEps,
                          )
    graphics.figOverPotsX(data_red_agg[(data_red_agg['target_privacy'] == 'grad_pert') & (data_red_agg['base_dataset'] == 'purchase_100')],
                          x_axes_key='target_model',
                          label_values_key='target_dp',
                          x_axis='sigma',
                          y_axis='naive_yeom_mi_max',
                          plot_function=graphics.plt_errbar_agg,
                          fig=subfigs[1],
                          plotAxes_function=plotAxesSigmaWithEps,
                          )
    addPic(fig,
           name='naiveOptVsNormYeomPurchase100PrivateLearning',
           show=False)
    del fig, subfigs
    '''def myMultiPlot(*args,y_axis=None,linestyle=None,linewidth=None,**kwargs):
        graphics.plt_errbar_agg(*args,y_axis='naive_yeom_mi',linestyle='-',linewidth=0.5,**kwargs)
        graphics.plt_errbar_agg(*args,y_axis='naive_yeom_mi_max',linestyle='--',linewidth=0.5,**kwargs)
    addPic(graphics.figOverPotsX(data_red_agg[(data_red_agg['target_privacy'] == 'grad_pert') & (data_red_agg['base_dataset'] == 'purchase_100')],
                          x_axes_key='target_model',
                          label_values_key='target_dp',
                          x_axis='sigma',
                          y_axis='naive_yeom_mi_max',
                          plot_function=myMultiPlot,
                          ),
           name=None,
           show=True)'''

    # For Appendix lack of coherence for housing and adult
    #fig: plt.Figure = plt.figure(figsize=(2 * 4, 4*4), constrained_layout=True)
    #subfigs = fig.subfigures(nrows=2, ncols=1)
    for i, fld in enumerate(['naive_yeom_mi', 'naive_yeom_mi_max']):
        addPic(graphics.figOverSubfigsXY(data_red_agg[(data_red_agg['target_privacy'] == 'grad_pert') & (data_red_agg['base_dataset'] != 'purchase_100')],
                                  x_axes_key='base_dataset',
                                  y_axes_key='target_model',
                                  label_values_key='target_dp',
                                  x_axis='sigma',
                                  y_axis=fld,
                                  plot_function=graphics.plt_errbar_agg,
                                  title=graphics.dName(fld),
                                         plotAxes_function=plotAxesSigmaWithEps,
                                  ),
               name=f'{"".join(p.capitalize() for p in fld.split("_"))}AdultHousingPrivateLearning',#None,#
               show=False)

    #del fig, subfigs
    with FullPandasPrinter():
        print(data_red_agg[(data_red_agg['target_privacy'] == 'grad_pert') & (data_red_agg['base_dataset'] == 'purchase_100') & (data_red_agg['target_model'] == 'nn') & (data_red_agg['target_dp'] == 'dp')])

    # Plot all naive private learnings

    '''addPic(graphics.figOverSubfigsXY(data_red_agg[data_red_agg['target_privacy']=='grad_pert'],
                                     y_axes_key='target_model',
                                     x_axes_key='base_dataset',
                                     label_values_key='target_dp',
                                     x_axis='sigma',
                                     y_axis=('yeom_mi', 'mean'),#'yeom_mi',#
                                     #y_err = ['min','max'],
                                     #logy=True,
                                     #plot_function=graphics.plt_errbar_agg,
                                     ),
           name=None,
           show=True)'''
    # addPic(graphics.figOverPotsX())
    for ds in ['adult', 'housing', 'purchase_100']:
        for md in ['nn', 'softmax']:
            tmp = data_red[(data_red['base_dataset'] == ds) & (data_red['target_model'] == md) & (data_red['target_privacy'] == 'grad_pert')]

            addFloatVar(tmp['sigma'].apply(np.log).corr(tmp['naive_yeom_mi_max'].apply(lambda x: max(0.00001, x)).apply(np.log)),
                        name=f'cor(sigma.log,naive_yeom_mi_max.log)>{ds}/{md}')
            print('naive_yeom_mi_max', ds, md, tmp['sigma'].apply(np.log).corr(tmp['naive_yeom_mi_max'].apply(lambda x: max(0.00001, x)).apply(np.log)), sep='\t')

            addFloatVar(tmp['sigma'].apply(np.log).corr(tmp['naive_yeom_mi']),
                        name=f'cor(sigma.log,naive_yeom_mi)>{ds}/{md}')
            print('naive_yeom_mi', ds, md, tmp['sigma'].apply(np.log).corr(tmp['naive_yeom_mi']), (tmp['naive_yeom_mi'] <= 0).sum() / tmp['naive_yeom_mi'].shape[0], sep='\t')
            print('min sub0 multiplier', ds, tmp[tmp['naive_yeom_mi'] <= 0]['sigma'].min())
            if ds == 'purchase_100':
                addFloatVar(tmp[tmp['naive_yeom_mi'] <= 0]['sigma'].min(), name=f'minSigmaSub0>{ds}/{md}')
            if ds == 'housing':
                addFloatVar((tmp['naive_yeom_mi'] <= 0).sum() / tmp['naive_yeom_mi'].shape[0], name=f'rationPLSub0>{ds}/{md}')
    del tmp

    # Pertrubation
    # yMIA_max
    addPic(graphics.figOverSubfigsXY(data_red_agg[(~data_red_agg['method'].isna()) & (data_red_agg['method'] != 'silentK')],
                                     y_axes_key='target_model',
                                     x_axes_key='base_dataset',
                                     label_values_key='method',
                                     x_axis='target_epsilon',
                                     y_axis='yeom_mi_max',
                                     plot_function=graphics.plt_errbar_agg
                                     ),
           name='OptyMIAOverEps',
           show=False)

    '''addPic(graphics.figOverSubfigsXY(data_red_agg[(~data_red_agg['method'].isna()) & (data_red_agg['method'] != 'silentK')],
                                     y_axes_key='target_model',
                                     x_axes_key='base_dataset',
                                     label_values_key='method',
                                     x_axis='target_epsilon',
                                     y_axis='yeom_mi',
                                     plot_function=graphics.plt_errbar_agg
                                     ),
           name=None,
           show=True) # TODO'''

    print()
    print('Perturbed yMIA')
    data_yeom_mi_agg: pd.DataFrame = (data_red[(~data_red['method'].isna()) & (data_red['method'] != 'silentK')]
                                      .drop(columns=['train_dataset', 'log_fields', 'label_dataset', 'source_dataset', 'results_file', 'results_foldername', 'result_folder', 'model_dirname'],errors='ignore')

                                      .groupby(['base_dataset',  'target_model','method'], dropna=False)
                                      .apply(lambda x:pd.Series({'yeom_mi_max_maxval':x['yeom_mi_max'].max(),
                                                                 'yeom_mi_max_sub0_ratio':(x['yeom_mi_max'] <= 0).sum() / x['yeom_mi'].shape[0],
                                                                 'corr(eps.log,yeom_mi_max.log)':x['target_epsilon'].apply(np.log).corr(x['yeom_mi_max'].apply(np.log))}))
                                      .reset_index()
                                      )
    with FullPandasPrinter():
        print(data_yeom_mi_agg)
    addFloatVar(data_yeom_mi_agg[(data_yeom_mi_agg['base_dataset']=='adult') & ((data_yeom_mi_agg['target_model']!='softmax') | (data_yeom_mi_agg['method']!='soria'))]['corr(eps.log,yeom_mi_max.log)'].min(), name='corr(eps.log,yeom_mi_max.log)|min>adult')
    addDfVars(data_yeom_mi_agg[data_yeom_mi_agg['base_dataset']!='purchase_100'][['base_dataset','target_model','method','corr(eps.log,yeom_mi_max.log)']],values=['corr(eps.log,yeom_mi_max.log)'])
    '''for ds in ['adult', 'housing', 'purchase_100']:
        for md in ['nn', 'softmax']:
            tmp = data_red[(data_red['base_dataset'] == ds) & (data_red['target_model'] == md) & (~data_red['method'].isna()) & (data_red['method'] != 'silentK')]
            print(ds, md, tmp['yeom_mi'].max(), (tmp['yeom_mi'] <= 0).sum() / tmp['yeom_mi'].shape[0])'''

    # Backlinked
    for dtst in ['adult','housing','purchase_100']:
        fig: plt.Figure = plt.figure(figsize=(4 * 4, 4), constrained_layout=True)
        subfigs = fig.subfigures(nrows=1, ncols=2)
        graphics.figOverPotsX(data_red_agg[(~data_red_agg['method'].isna()) & (data_red_agg['method'] != 'silentK') & (data_red_agg['base_dataset'] == dtst)],
                              x_axes_key='target_model',
                              label_values_key='method',
                              x_axis='target_epsilon',
                              y_axis='back_linked_yeom_mi',
                              plot_function=graphics.plt_errbar_agg,
                              fig=subfigs[0],
                              )
        graphics.figOverPotsX(data_red_agg[(~data_red_agg['method'].isna()) & (data_red_agg['method'] != 'silentK') & (data_red_agg['base_dataset'] == dtst)],
                              x_axes_key='target_model',
                              label_values_key='method',
                              x_axis='target_epsilon',
                              y_axis='opt_back_linked_yeom_mi',
                              plot_function=graphics.plt_errbar_agg,
                              fig=subfigs[1],
                              )
        if dtst in {'adult','housing','purchase_100'}:
            addPic(fig,
                   name=f'CyMIAOverEpsFor{dtst.capitalize().replace("_","")}',
                   show=False)
        else:
            pass#fig.show()
    del fig, subfigs


    # AyMIA
    # Plot all versions for soria (and others for apendix)
    for method in ['soria','fusion','doca']:
        addPic(graphics.figOverSubfigsXY(data_red_agg[(data_red_agg['method'] == method)],
                                         y_axes_key='target_model',
                                         x_axes_key='base_dataset',
                                         label_y_column_key=['yeom_mi_max', 'adapted_yeom_mi_median', 'opt_adapted_yeom_mi_median', 'adapted_yeom_mi_max', 'opt_adapted_yeom_mi_max', ],
                                         ylabel='attacker score',
                                         x_axis='target_epsilon',
                                         title='soria',
                                         # y_err = ['min','max'],
                                         # logy=True,
                                         plot_function=lambda *args, y_axis=None, **kwargs: graphics.plt_errbar_agg(*args, y_axis=y_axis, **kwargs) if y_axis != 'yeom_mi_max' else graphics.plt_plot(*args, y_axis=(y_axis, 'mean'), **kwargs)
                                         ),
               name=f'AyMIAoverview{method.capitalize()}',
               show=False)

    # Look at max values
    data_yeom_mi_agg: pd.DataFrame = (data_red[(~data_red['method'].isna()) & (data_red['method'] != 'silentK')]
                                      .drop(columns=['train_dataset', 'log_fields', 'label_dataset', 'source_dataset', 'results_file', 'results_foldername', 'result_folder', 'model_dirname'],errors='ignore')

                                      .groupby(['base_dataset', 'target_model', 'method'], dropna=False)
                                      .apply(lambda x: pd.Series({'yeom_mi_max_maxval': x['yeom_mi_max'].max(),
                                                                  'yeom_mi_max_sub0_ratio': (x['yeom_mi_max'] <= 0).sum() / x['yeom_mi'].shape[0],
                                                                  'corr(eps.log,yeom_mi_max.log)': x['target_epsilon'].apply(np.log).corr(x['yeom_mi_max'].apply(np.log))}))
                                      .reset_index()
                                      )



    del data_yeom_mi_agg
    dr = data_red[(~data_red['method'].isna()) & (data_red['method'] != 'silentK')]
    print('mean-diff -',((dr['adapted_yeom_mi_median']-dr['opt_adapted_yeom_mi_median']).mean()+(dr['adapted_yeom_mi_max']-dr['opt_adapted_yeom_mi_max']).mean())/2)
    addFloatVar(((dr['adapted_yeom_mi_median']-dr['opt_adapted_yeom_mi_median']).mean()+(dr['adapted_yeom_mi_max']-dr['opt_adapted_yeom_mi_max']).mean())/2,
                name='adapted_yeom_mi_medianDiffOpt|mean')
    '''print('means-diff',(dr['adapted_yeom_mi_median']/dr['opt_adapted_yeom_mi_median']).mean(),(dr['adapted_yeom_mi_max']/dr['opt_adapted_yeom_mi_max']).mean())
    print('means-diff',(dr['opt_adapted_yeom_mi_median']/dr['adapted_yeom_mi_median']).mean(),(dr['opt_adapted_yeom_mi_max']/dr['adapted_yeom_mi_max']).mean())
    print('mean-diff *',((dr['adapted_yeom_mi_median']/dr['opt_adapted_yeom_mi_median']).mean()+(dr['adapted_yeom_mi_max']/dr['opt_adapted_yeom_mi_max']).mean())/2)
    print('mean-diff *',((dr['opt_adapted_yeom_mi_median']/dr['adapted_yeom_mi_median']).mean()+(dr['opt_adapted_yeom_mi_max']/dr['adapted_yeom_mi_max']).mean())/2)'''
    del dr
    '''for method in ['fusion','doca']:
        addPic(graphics.figOverSubfigsXY(data_red_agg[(data_red_agg['method'] == method)],
                                         y_axes_key='target_model',
                                         x_axes_key='base_dataset',
                                         label_y_column_key=['yeom_mi_max','adapted_yeom_mi_median','opt_adapted_yeom_mi_median','adapted_yeom_mi_max','opt_adapted_yeom_mi_max',],
                                         ylabel='attacker score',
                                         x_axis='target_epsilon',
                                         title=method,
                                         # y_err = ['min','max'],
                                         # logy=True,
                                         plot_function=lambda *args,y_axis=None,**kwargs:graphics.plt_errbar_agg(*args,y_axis=y_axis,**kwargs) if y_axis!='yeom_mi_max' else graphics.plt_plot(*args,y_axis=(y_axis,'mean'),**kwargs)
                                         ),
               name=None,
               show=False)
    del method

    addPic(graphics.figOverSubfigsXY(data_red_agg[(~data_red_agg['method'].isna()) & (data_red_agg['method'] != 'silentK')],
                                     y_axes_key='target_model',
                                     x_axes_key='base_dataset',
                                     label_values_key='method',
                                     x_axis='target_epsilon',
                                     y_axis='opt_adapted_yeom_mi_max',  # ('naive_yeom_mi_max', 'mean')
                                     # y_err = ['min','max'],
                                     # logy=True,
                                     plot_function=graphics.plt_errbar_agg
                                     ),
           name= None,
           show=False)'''

def sec3K():
    data_agg:pd.DataFrame = (data[(data['method']=='soria')|(data['method']=='fusion')|(data['method']=='silentK')]
                              # .dropna(subset=['result_folder', 'results_foldername', 'results_file',])
                              .drop(columns=['train_dataset', 'log_fields', 'label_dataset', 'source_dataset', 'results_file', 'results_foldername', 'result_folder', 'model_dirname'],errors='ignore')
                              .groupby(['base_dataset', 'method', 'target_epsilon', 'target_model', 'target_privacy', 'target_dp','k'], dropna=False)
                              .aggregate(['min', 'mean', 'max', 'std'])
                              .reset_index()
                              )
    addPic(graphics.figOverSubfigsXY(data_red_agg[(data_red_agg['method'] == 'soria') | (data_red_agg['method'] == 'fusion')],
                                     y_axes_key='target_model',
                                     x_axes_key='base_dataset',
                                     label_values_key='method',
                                     y_axis=('k','mean'),
                                     x_axis='target_epsilon',
                                     # y_err = ['min','max'],
                                     logy=True,
                                     plot_function=graphics.plt_plot
                                     ),
           name='bestKOverEps',
           show=False)

    # Calculate loss due to micro-agg
    sm_k1,smk2 ,*_ =sorted(data_agg[data_agg['base_dataset']=='purchase_100']['k'].unique())
    accLoss = (data_agg[(data_agg['base_dataset'] == 'purchase_100') & (data_agg['k'] <= smk2) & (data_agg['method'] == 'silentK')]
               .groupby(['target_model'], dropna=False)
               .apply(lambda x: pd.Series({'acc_test_realLossFromMicroAgg': 1 - (x[x['k'] == smk2][('real_test_acc', 'mean')].iloc[0] / x[x['k'] == sm_k1][('real_test_acc', 'mean')].iloc[0])}))
               .reset_index())
    print('Accuracy loss due to micro-agg')
    with FullPandasPrinter():
        #addDfVars(accLoss,values=['acc_test_realLossFromMicroAgg'])
        print(accLoss)
    addFloatVar(accLoss['acc_test_realLossFromMicroAgg'].max(),name='acc_test_realLossFromMicroAgg|max>purchase_100')
    del accLoss,smk2,sm_k1


    for method in ['soria', 'fusion']:
        for field in ['real_test_acc','naive_yeom_mi','naive_yeom_mi_max']:#['yeom_mi_max']:#
            data_agg.loc[(data_agg['method'] == 'silentK'),['target_epsilon']] = np.nan
            fig:plt.Figure = graphics.figOverSubfigsXY(data_agg[(data_agg['method'] == method) | (data_agg['method'] == 'silentK')],
                                             y_axes_key='target_model',
                                             x_axes_key='base_dataset',
                                             label_values_key='target_epsilon',
                                             y_axis=field,
                                             x_axis='k',
                                             title=method,
                                             color=colGrad('target_epsilon',data_agg),
                                             zorder=lambda k,data =None,**kwargs:10 if np.isnan(k[1]) else None,
                                             label_name=lambda x,**kwargs:'silentK' if np.isnan(x) else f'$\\epsilon={str(x)}$' if isinstance(x,(int,float)) else graphics.dName_short(x,**kwargs),
                                             linewidth=lambda k,**kwargs: 0.75 if np.isnan(k[1]) else None,
                                             linestyle= lambda k,**kwargs: '--' if np.isnan(k[1]) else None,
                                             # y_err = ['min','max'],
                                             # logy=True,
                                             plot_function=graphics.plt_errbar_agg
                                             )
            for i,ax in enumerate(fig.axes):
                assert isinstance(ax,plt.Axes)
                ax.get_legend().remove()

            if (method, field) == ('fusion', 'real_test_acc'):
                fig.axes[2].legend()
            elif (method, field) == ('soria','naive_yeom_mi_max'):
                fig.axes[3].legend()
            '''
            handles, labels = fig.axes[0].get_legend_handles_labels()
            from matplotlib.gridspec import GridSpec

            for i,sf in enumerate(fig.subfigs):
                sf.subplotspec = gs[i,0]
            fig2 = fig.add_subfigure(gs[:,-1])
            #fig2.set_size_inches(2,2)
            leg = fig2.legend(handles, labels, loc='center')
            leg.set_in_layout(False)
            # trigger a draw so that constrained_layout is executed once
            # before we turn it off when printing....
            fig.canvas.draw()
            # we want the legend included in the bbox_inches='tight' calcs.
            leg.set_in_layout(True)
            # we don't want the layout to change at this point.
            fig.set_constrained_layout(False)
            w,h = fig.get_size_inches()
            fig.set_size_inches(w+2,h)'''
            if (method,field) in [('fusion','real_test_acc'),('soria','naive_yeom_mi_max')]:#
                addPic(fig,
                       name=f'{"".join(f.capitalize() for f in field.split("_"))}OverKFor{method.capitalize()}',
                       show=False)
    #graphics.plotKVsColumn(data[(data['method']=='soria') & (data['target_model']=='nn')& (data['base_dataset']=='adult')])

    addPic(getOptK(),name='optimalLabelK',show=False)

    pass

def secDiscussion():
    for dataset in ['purchase_100','housing','adult']:
        fig :plt.Figure = graphics.figOverSubfigsXY(data_red_agg[((~data_red_agg['method'].isna()) & (data_red_agg['method'] != 'silentK')) & (data_red_agg['base_dataset'] == dataset)],
                                         y_axes_key='target_model',
                                         x_axes_key='method',
                                         label_x_column_key=list(zip(['yeom_mi_max', 'opt_adapted_yeom_mi_median', 'adapted_yeom_mi_max', 'opt_adapted_yeom_mi_max'],graphics.constGen('mean'))),
                                         xlabel='attacker score',
                                         y_axis=('real_test_acc','mean'),
                                         title=dataset,
                                        logx=False,
                                         # y_err = ['min','max'],
                                         # logy=True,
                                         )

        for i, ax in enumerate(fig.axes):
            ax.get_legend().remove()
            graphics.addSomething(data_red_agg[data_red_agg['method'].isna() & (data_red_agg['target_privacy'] == 'grad_pert') & (data_red_agg['base_dataset'] == dataset)& (data_red_agg['target_model'] == ('nn' if i<3 else 'softmax'))].sort_values(('yeom_mi_max','mean')),
                                  x_axis=('yeom_mi_max','mean'),
                                  y_axis=('real_test_acc','mean'),
                                  label='private learning (opt. yMIA)',
                                  xlabel='attacker score',
                                  ax=ax,
                                        logx=False,)
            if i == 0:
                ax.legend()

        if dataset in ['purchase_100','housing','adult']:
            addPic(fig,
                   name=f'realTestAccOverAttackerScoreFor{dataset.replace("_","").capitalize()}',
                   show=False)
    pass

def saveLibaries(cols=1):
    import subprocess
    libs = subprocess.run(['pip', 'freeze'], stdout=subprocess.PIPE, text=True).stdout
    libs = libs.strip().replace("_","\\_").split('\n')
    libs = [l.split('==') for l in libs]
    libs = [f'{l} & ${v}$' for l,v in libs]
    if len(libs)%cols:
        libs += ['&']*(cols-(len(libs)%cols))
    tab = ' \\\\\n    \\hline\n    '.join(' & '.join(libs[i+ii*int(len(libs)/cols)] for ii in range(cols)) for i in range(int(len(libs)/cols)))
    tab = '''
\\begin{longtable}{|%s|}
    \\hline
    %s \\\\
    \\hline
    \\hline
    %s \\\\
    \\hline
    \\caption{Python libraries installed, when running the experiment.}
    \\label{tab:libraries}
\\end{longtable}
    ''' % ('||'.join('c|c' for _ in range(cols)),' & '.join('Name & Version' for i in range(cols)),tab)
    tab = tab.strip()
    print(tab)

def presentation():
    addPic(graphics.figPlotCollection(
        [[
            functools.partial(graphics.plotOverLabels,
                              data_red_agg[(~data_red_agg['method'].isna()) & (data_red_agg['target_privacy'] == 'no_privacy') & (data_red_agg['target_model'] == 'softmax') & (data_red_agg['base_dataset'] == 'purchase_100')],
                              label_values_key='method',
                              x_axis='target_epsilon',
                              y_axis=('real_test_acc', 'mean')
                              ),
            functools.partial(plotAxesSigmaWithEps,
                              data_red_agg[(data_red_agg['method'].isna()) & (data_red_agg['target_privacy'] == 'grad_pert') & (data_red_agg['target_model'] == 'nn') & (data_red_agg['base_dataset'] == 'housing')],
                              label_values_key='target_dp',
                              x_axis='sigma',
                              y_axis=('real_test_acc', 'mean'),
                              ),
        ]],titles=[['softmax/purchase_100','neural network/housing']]),
        name='exemplatoryAccuracy for Privacy',
        folders=['output','presentation'],
        show=False)
    addPic(graphics.figPlotCollection(
        [[
            functools.partial(plotAxesSigmaWithEps,
                              data_red_agg[(data_red_agg['target_privacy'] == 'grad_pert') & (data_red_agg['base_dataset'] == 'housing') & (data_red_agg['target_model'] == 'nn')],
                              label_values_key='target_dp',
                              x_axis='sigma',
                              y_axis='naive_yeom_mi',
                              plot_function=graphics.plt_errbar_agg,
                              ),
            functools.partial(plotAxesSigmaWithEps,
                              data_red_agg[(data_red_agg['target_privacy'] == 'grad_pert') & (data_red_agg['base_dataset'] == 'adult') & (data_red_agg['target_model'] == 'softmax')],
                              label_values_key='target_dp',
                              x_axis='sigma',
                              y_axis='naive_yeom_mi_max',
                              plot_function=graphics.plt_errbar_agg,
                              ),
        ]],
        titles=[['housing/neural network', 'adult/softmax']]
    ),
           name='exemplatoryYMIAprivateLearningUnsuccessful',
           folders=['output','presentation'],
           show=False)
    addPic(graphics.figOverPotsX(data_red_agg[(data_red_agg['target_privacy'] == 'grad_pert') & (data_red_agg['base_dataset'] == 'purchase_100')],
                          x_axes_key='target_model',
                          label_values_key='target_dp',
                          x_axis='sigma',
                          y_axis='naive_yeom_mi_max',
                          plot_function=graphics.plt_errbar_agg,
                          plotAxes_function=plotAxesSigmaWithEps,
                          ),
           name='exemplatoryYMIAprivateLearningSuccessful',
           folders=['output','presentation'],
           show=False)
    addPic(graphics.figPlotCollection(
        [[
            functools.partial(graphics.plotOverLabels,
                              data_red_agg[(~data_red_agg['method'].isna()) & (data_red_agg['method'] != 'silentK') & (data_red_agg['base_dataset'] == 'purchase_100') & (data_red_agg['target_model'] == 'softmax')],
                              label_values_key='method',
                              x_axis='target_epsilon',
                              y_axis='yeom_mi_max',
                              plot_function=graphics.plt_errbar_agg
                              ),
            functools.partial(graphics.plotOverLabels,
                              data_red_agg[(~data_red_agg['method'].isna()) & (data_red_agg['method'] != 'silentK') & (data_red_agg['base_dataset'] == 'adult') & (data_red_agg['target_model'] == 'nn')],
                              label_values_key='method',
                              x_axis='target_epsilon',
                              y_axis='yeom_mi_max',
                              plot_function=graphics.plt_errbar_agg
                              ),
        ]],
        titles=[['purchase_100/softmax', 'adult/neural network']]
    ),
    name='exemplatoryYMIAperturbedSenario',
    folders=['output','presentation'],
    show=False)



#sec1Accuracy()
#sec2Attackers()
#sec3K()
secDiscussion()
#saveLibaries(cols=2)
#presentation()

#flushVars()

# graphics.figAggByEps(data[(~data['method'].isna()) & (data['target_privacy']=='no_privacy') & (data['target_model']=='nn')],COLUMN='real_test_acc',fig_over_column='base_dataset',ax_over_column='method').show()

# graphics.figAggByEps(data[(~data['method'].isna()) & (data['target_privacy']=='no_privacy') & (data['target_model']=='nn')],COLUMN='naive_yeom_mi',fig_over_column='base_dataset',ax_over_column='method').show()

# graphics.figAggByEps(data[(~data['method'].isna()) & (data['target_privacy']=='no_privacy') & (data['target_model']=='softmax')],COLUMN='real_test_acc',fig_over_column='base_dataset',ax_over_column='method').show()
# graphics.figAggByEps(data[(data['method'].isna()) & (data['target_privacy']=='no_privacy')],COLUMN='real_test_acc',fig_over_column='base_dataset',ax_over_column='target_model').show()
