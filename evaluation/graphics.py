import os, sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.figure
from typing import Union,List,Dict,Tuple,Optional

from evaluation.attackOnResults import reduceToMostSuccessfull,KEY as EXPERIMENT_KEYS
from utils import FullPandasPrinter

np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


EXPLANATION = {
    'naive_yeom_mi':'Score of the yeom MIA attempting to retrieve perturbed training data',
    'naive_yeom_mi_max':'Score of the yeom MIA attempting to retrieve perturbed training data knowing optimal threshold',
    'yeom_mi':'Score of the yeom MIA attempting to retrieve original training data',
    'yeom_mi_max':'Score of the yeom MIA attempting to retrieve original training data knowing optimal threshold'
}

OTHER_WORD = {
    'naive_yeom_mi':'naive yMIA score',
    'naive_yeom_mi_max':'optimized naive yMIA score',
    'yeom_mi':'yMIA score',
    'yeom_mi_max':'optimized yMIA score',
    'back_linked_yeom_mi':'CyMIA score',
    'opt_back_linked_yeom_mi':'optimized CyMIA score',
    'adapted_yeom_mi_max':'maximized AyMIA score',
    'adapted_yeom_mi_median':'AyMIA score',
    'opt_adapted_yeom_mi_max':'optimized maximized AyMIA score',
    'opt_adapted_yeom_mi_median':'optimized AyMIA score',

    'target_epsilon':'privacy budget $\epsilon$',
    'k':'cluster size $k$',
    'real_train_acc':'$Acc_{train}^{real}$',
    'real_test_acc':'$Acc_{test}^{real}$',
    'train_acc':'$Acc_{train}^{training}$',
    'test_acc':'$Acc_{test}^{training}$',

    'nn':'neural network',
    'softmax':'softmax regression',

    'purchase_100':'purchase_100',
    'housing':'housing',
    'adult':'adult',

    'doca':'DOCA',
    'fusion':'fusion',
    'soria':'soria',
    'silentK':'silentK',

    'adv_cmp': 'Advanced composition',
    'rdp': 'Rényi DP',
    'zcdp': 'Zero-Concentrated DP',
    'dp': 'naive composition',

    'sigma':'noise multiplier',
}

SHORT_OTHER_WORD = {
    'naive_yeom_mi':'naive yMIA',
    'naive_yeom_mi_max':'opt. naive yMIA',
    'yeom_mi':'yMIA',
    'yeom_mi_max':'opt. yMIA',
    'back_linked_yeom_mi':'CyMIA',
    'opt_back_linked_yeom_mi':'opt. CyMIA',
    'adapted_yeom_mi_max':'max. AyMIA',
    'adapted_yeom_mi_median':'AyMIA',
    'opt_adapted_yeom_mi_max':'opt. max. AyMIA',
    'opt_adapted_yeom_mi_median':'opt. AyMIA',
}

AGG_WORDS = {
    'mean': 'Average %s'
}

SCENARIO_KEYS = ['base_dataset', 'method', 'k', 'target_epsilon', 'target_model', 'target_dp','target_privacy']

class ColorWheel:
    def __init__(self):
        pass

def const(val):
    return lambda *args,**kwargs:val
def constGen(val):
    def gen():
        while True:
            yield val
    return gen()

def update(d:dict,**kwargs):
    dd = d.copy()
    dd.update(kwargs)
    return dd

def dName(name, **kwargs):
    if name in OTHER_WORD:
        return OTHER_WORD[name]
    if isinstance(name, tuple) and len(name)==2:
        if name[1] in AGG_WORDS:
            return AGG_WORDS[name[1]] % dName(name[0])
    raise ValueError(f'Missong other word for \'{name}\'')

def dName_short(name, **kwargs):
    if name in SHORT_OTHER_WORD:
        return SHORT_OTHER_WORD[name]
    return dName(name,**kwargs)

def adoptedKwargs(kw:dict,kw_fields = None,kw_remove=None, key=None,**kwargs):
    kw_fields = ['color','zorder','linewidth','lw','linestyle','ls'] if kw_fields is None else kw_fields
    kw_remove = [] if kw_remove is None else kw_remove
    if not isinstance(kw_remove,list):
        kw_remove = [kw_remove]

    ret = kw.copy()
    for field in kw_remove:
        del ret[field]
    for field in kw_fields:
        if field not in ret:
            continue
        if callable(kw[field]):
            ret[field] = kw[field](key,**kwargs)
    return ret


'''def genPlotUnPerturbed(data, column='naive_yeom_mi_max', agg='mean', axs:plt.Axes=None):
    if axs is None:
        axs = plt
    data = data[data['method'].isna()]
    data = data[data['target_privacy']=='grad_pert']
    for dp in data['target_dp'].unique():
        dt = data[data['target_dp'] == dp]
        dt: pd.DataFrame = (dt
                            .groupby(['base_dataset', 'method', 'target_epsilon', 'target_model', 'target_dp'],
                                     dropna=False)
                            .aggregate(['max', 'mean', 'std'])
                            .reset_index())

        dt = dt.set_index('target_epsilon')
        dt[(column, agg)].plot(  # logy=True,
            logx=True,
            xlabel=dName('target_epsilon'),
            ylabel=dName(column),
            label=dp,
            ax=axs)
    axs.title(f'{dName(column)} evolving over epsilon for different notions of privacy')
    axs.legend()
    axs.show()
'''
'''def plotKVsColumn(data,COLUMN='naive_yeom_mi',agg='mean', axs:plt.Axes = None, log = False):
    if axs is None:
        axs = plt
    data = (data.groupby(['base_dataset', 'target_epsilon', 'method', 'target_model', 'target_privacy', 'target_dp','k'], dropna=False)
            .aggregate(['max','mean','std'])
            .reset_index())
    #data = data[data['target_epsilon']<=1]
    for d in data.groupby(['base_dataset', 'target_epsilon', 'method', 'target_model', 'target_privacy', 'target_dp'], dropna=False):
        axs.plot(d[1][['k']].to_numpy(),
                 d[1][[(COLUMN,agg)]].to_numpy(),
                 label = (d[0][2], d[0][1]),
                 linewidth = 3 if d[0][2]=='silentK' else 1)

    axs.legend()
    if log:
        axs.yscale('log')
    axs.xlabel(dName('k'))
    axs.ylabel(dName(COLUMN))
    axs.title(f'{dName(COLUMN)} evolving over different values of k for different pert-algo and epsilon')
    axs.show()
'''


def plt_plot(data,
             ax: plt.Axes = None,
             y_axis='yeom_mi',
             x_axis='target_epsilon',
             label=None,
             allow_constant=True,
             key=None,
             **kwargs):
    if key is None:
        key = y_axis

    if data.shape[0] == 1 and allow_constant:
        ax.axhline(y=data[y_axis].to_numpy(),
                   label=label if label is None or isinstance(label, str) else label(data),
                   color='magenta',
                   **adoptedKwargs(kwargs,kw_remove='color',key=key,data=data,y_axis=y_axis,x_axis=x_axis))
    else:
        ax.plot(data[x_axis].to_numpy(),
                data[y_axis].to_numpy(),
                label=label if label is None or isinstance(label, str) else label(data),
                **adoptedKwargs(kwargs,key=key,data=data,y_axis=y_axis,x_axis=x_axis))


def plt_errbar(data,
               ax: plt.Axes = None,
               y_axis='yeom_mi',
               x_axis='target_epsilon',
               y_err=None,
               label=None,
               allow_constant=True,
               key=None,
               **kwargs):
    if key is None:
        key = y_axis

    ax.errorbar(data[x_axis].to_numpy(),
                data[y_axis].to_numpy(),
                yerr=data[y_err].to_numpy() if not isinstance(y_err,list) else np.vstack([data[y_e].to_numpy() for y_e in y_err]),
                label=label if label is None or isinstance(label, str) else label(data),
                capsize=2,
                **adoptedKwargs(kwargs,key=key,data=data,y_axis=y_axis,x_axis=x_axis))
def plt_errbar_agg(*args,
                   y_axis='yeom_mi',
                   y_agg = 'mean',
                   y_err = 'std',
                   **kwargs):
    plt_errbar(*args,
               y_axis = (y_axis,y_agg),
               y_err = (y_axis, y_err) if not isinstance(y_err,list) else [(y_axis, y_e) for y_e in y_err],
               **kwargs)


def addSomething(data: pd.DataFrame,
                 y_axis='yeom_mi',
                 x_axis='target_epsilon',
                 label=None,
                 ax: plt.Axes = None,
                 logy=False,
                 logx=True,
                 xlabel=None,
                 ylabel=None,
                 color=None,
                 plot_function=None,
                 zorder=None,
                 data_filter=None,
                 **kwargs):
    # Bring params in tight format
    if ax is None:
        ax = plt

    data_filter = data_filter if data_filter is not None else lambda x,**kwargs:x
    plot_function = plt_plot if plot_function is None or not callable(plot_function) else plot_function

    if isinstance(xlabel,str):
        xlabel = const(xlabel)
    if isinstance(ylabel,str):
        ylabel = const(ylabel)

    if not callable(color):
        color = const(color)
    if not callable(zorder):
        zorder = const(zorder)

    # Prepare data and params
    params = update(kwargs,
                    ax=ax,
                    y_axis=y_axis,
                    x_axis=x_axis,
                    label=label,
                    color=color,
                    zorder=zorder)
    data = data_filter(data,**params)

    # Actual plotting
    plot_function(data,**params)

    #
    if logy:
        ax.set_yscale('log')
    if logx:
        ax.set_xscale('log')

    if xlabel is None:
        ax.set_xlabel(dName(x_axis))
    elif xlabel:
        ax.set_xlabel(xlabel(x_axis, data=data,y_axis=y_axis,x_axis=x_axis))

    if ylabel is None:
        ax.set_ylabel(dName(y_axis))
    elif ylabel:
        ax.set_ylabel(ylabel(y_axis, data=data,y_axis=y_axis,x_axis=x_axis))

def plotOverValueLabels(data:pd.DataFrame,
                        label_key:Union[str,dict,list,tuple] = 'method',
                        title=None,
                        ax:plt.Axes = None,
                        label_name= None,
                        show_legend =None,
                        **kwargs):
    ax = plt if ax is None else ax
    label_name = dName if label_name is None else label_name

    columns = list(data.columns)

    if isinstance(label_key,str):
        label_key =[label_key]
    if isinstance(label_key,(list,tuple)):
        temp = label_key
        label_key = {}
        for item in temp:
            if isinstance(item,dict):
                for k,v in item.items():
                    if k in label_key:
                        label_key[k].extend(v)
                    elif v is None:
                        label_key[k] = list(data[k].unique())
                    else:
                        label_key[k]=v
            else:
                assert item not in label_key, 'Cannot go over label twice'
                label_key[item] = list(data[item].unique())
        del temp

    plt_count = 0
    for key,values in label_key.items():
        for value in values:
            addSomething(data[(data[key]==value) if  not (isinstance(value,float) and np.isnan(value)) else data[key].isna()],
                         label=label_name(value,data= data,filter_column=key,filter_value=value),
                         key=(key,value),
                         ax=ax,
                         **kwargs)
            plt_count+=1

    if (show_legend is None and plt_count) or show_legend:
        ax.legend()

    if title is not None:
        ax.set_title(title)
    return ax

def plotOverColumnLabels(data:pd.DataFrame,
                        label_x_column_key = None,
                        label_y_column_key = None,
                        label_xy_column_key = None,
                        y_axis=None,
                        x_axis=None,
                        title=None,
                        ax:plt.Axes = None,
                        label_name= None,
                        show_legend =None,
                        **kwargs):
    ax = plt if ax is None else ax
    label_name = dName_short if label_name is None else label_name

    if label_x_column_key is not None and label_y_column_key is not None:
        assert label_xy_column_key is None, 'To much info given TODO'
        label_xy_column_key = list(zip(label_x_column_key,label_y_column_key))

    if label_xy_column_key is not None:
        label_xy_column_key=list(label_xy_column_key)
        key_gen =lambda *args:tuple(args)
    elif label_x_column_key is not None:
        assert y_axis is not None, 'No y axis given'
        label_xy_column_key= list(zip(label_x_column_key,constGen(y_axis)))
        key_gen =lambda *args:args[0]
    elif label_y_column_key is not None:
        assert x_axis is not None, 'No x axis given'
        label_xy_column_key= list(zip(constGen(x_axis),label_y_column_key))
        key_gen =lambda *args:args[1]
    else:
        label_xy_column_key = [(x_axis,y_axis)]
        key_gen =lambda *args:args


    for x_axis, y_axis in label_xy_column_key:
        key = key_gen(x_axis, y_axis)
        addSomething(data,
                     x_axis=x_axis,
                     y_axis=y_axis,
                     label=label_name(key,data= data),
                     ax=ax,
                     key=key,
                     **kwargs)

    if (show_legend is None and len(label_xy_column_key)>1) or show_legend:
        ax.legend()

    if title is not None:
        ax.set_title(title)
    return ax


def plotOverLabels(data: pd.DataFrame,
                         label_values_key=None,
                         title=None,
                         ax: plt.Axes = None,
                         show_legend=None,
                         **kwargs):
    ax = plt if ax is None else ax

    #label_column_key = [] if label_column_key is None else label_column_key
    #label_values_key = [] if label_values_key is None else label_values_key
    label_column_key_given = any(k in kwargs for k in ['label_x_column_key','label_y_column_key','label_xy_column_key'])
    if label_column_key_given:
        plotOverColumnLabels(data,ax=ax,show_legend=show_legend,**kwargs)
    if label_values_key:
        plotOverValueLabels(data, label_key=label_values_key,ax=ax,show_legend=show_legend,**kwargs)

    if (show_legend is None and label_column_key_given and label_values_key) or show_legend:
        ax.legend()

    if title is not None:
        ax.set_title(title)
    return ax


def figOverPotsX(data:pd.DataFrame, x_axes_key=None, figbasesize=None, x_fig_values = None, fig:plt.Figure = None, plotAxes_function=None, **kwargs):
    figbasesize = (4,4) if figbasesize is None else figbasesize
    plotAxes_function = plotOverLabels if plotAxes_function is None else plotAxes_function

    values = list(data[x_axes_key].unique()) if x_fig_values is None else x_fig_values
    if fig is None:
        fig, axes = plt.subplots(nrows=1, ncols=len(values), figsize=(figbasesize[0] * len(values), figbasesize[1]), constrained_layout=True)
    else:
        axes = fig.subplots(nrows=1,ncols=len(values))
    assert isinstance(fig, (plt.Figure, matplotlib.figure.SubFigure)), f'fig should be Figure, go {type(fig)} instead'

    for dataset, ax in zip(values, axes):
        plotAxes_function(data[data[x_axes_key] == dataset], ax=ax, **kwargs)
        ax.set_title(dName(dataset), fontsize=11)
    return fig

def figOverSubfigsXY(data:pd.DataFrame,y_axes_key=None,x_axes_key=None,figbasesize=None,x_fig_values=None,y_fig_values=None, title=None, fig:plt.Figure=None, **kwargs):
    figbasesize = (4,4) if figbasesize is None else figbasesize

    values_y = list(data[y_axes_key].unique()) if y_fig_values is None else y_fig_values
    values_x = list(data[x_axes_key].unique()) if x_fig_values is None else x_fig_values

    if fig is None:
        fig:plt.Figure = plt.figure(figsize=(figbasesize[0] * len(values_x), figbasesize[1]*len(values_y)),constrained_layout=True)
    subfigs = fig.subfigures(nrows=len(values_y), ncols=1)
    #fig, axes = plt.subplots(, figsize=(figbasesize[0] * len(values_y), figbasesize[1]), constrained_layout=True)
    #assert isinstance(fig, plt.Figure)

    for dataset, f in zip(values_y, subfigs):
        figOverPotsX(data[data[y_axes_key] == dataset], fig = f, x_axes_key=x_axes_key, x_fig_values=values_x, **kwargs)
        f.suptitle(dName(dataset), fontsize=14)
    if title is not None:
        fig.suptitle(title,fontsize=16)
    return fig

def figPlotCollection(plts:list, titles=None, figbasesize=None,fig:plt.Figure=None,):
    figbasesize = (4, 4) if figbasesize is None else figbasesize

    assert isinstance(plts,list)
    if not isinstance(plts[0],list):
        plts = [plts]
    n_rows = len(plts)
    n_cols= len(plts[0])
    if fig is None:
        fig = plt.figure(figsize=(figbasesize[0] * n_cols, figbasesize[1]*n_rows),constrained_layout=True)
    axes = fig.subplots(n_rows,n_cols,squeeze=False)
    for i_row, ax_row in enumerate(axes):
        for i_col, ax in enumerate(ax_row):
            plts[i_row][i_col](ax=ax)
            if titles is not None:
                ax.set_title(titles[i_row][i_col])
    return fig



#Work in progress: Between
'''
def plotOverLabels_Between(data: pd.DataFrame,
                          label_key=None,
                          label_column_key=None,
                          label_values_key=None,
                          title=None,
                          ax: plt.Axes = None,
                          label_name=None,
                          show_legend=None,
                          **kwargs):
    ax = plt if ax is None else ax
    label_name = dName if label_name is None else label_name
    columns = list(data.columns)
    
    def isSingleColumn(c):
        if isinstance(c, dict):
            return len(c) == 1 and list(c.keys())[0] in columns
        return c in columns

    def istuple(t):
        if not isinstance(t, tuple) or len(t) != 2:
            return False
        return isSingleColumn(t[0]) and isSingleColumn(t[0])

    if not isinstance(label_key, (list, tuple)) and not any(
            ((isinstance(t[0], dict) and len(t[0]) == 1) or t[0] in columns) and
            ((isinstance(t[1], dict) and len(t[1]) == 1) or t[1] in columns) for t in label_key if
            isinstance(t, tuple)):
        return plotOverValueLabels(data, label_key=label_key, title=title, ax=ax, label_name=label_name)


    label_column_key = [] if label_column_key is None else label_column_key
    label_values_key = [] if label_values_key is None else label_values_key
    
    colormapping = {}
    for item in label_key:
        if not istuple(item):
            new_label_key.append(item)
            colormapping[]

    

    if isinstance(label_key,str):
        label_key =[label_key]
    if isinstance(label_key,(list,tuple)):
        temp = label_key
        label_key = {}
        for item in temp:
            if isinstance(item,dict):
                for k,v in item.items():
                    assert k not in label_key, 'Cannot use column as blank label and be specific.'
                    if v is None:
                        label_key[k] = list(data[k].unique())
                    else:
                        label_key[k]=v
            else:
                assert item not in label_key, 'Cannot go over label twice'
                label_key[item] = list(data[item].unique())
        del temp

    plt_count = 0
    for key,values in label_key.items():
        for value in values:
            addPlot(data[data[key]==value],label=label_name(value,data= data,key=key),ax=ax, **kwargs)
            plt_count+=1
    

    if (show_legend is None and plt_count) or show_legend:
        ax.legend()

    if title is not None:
        ax.set_title(title)
    return ax
'''

# Old addPlot
'''
def addPlot(data:pd.DataFrame,
            y_axis = 'yeom_mi',
            x_axis ='target_epsilon',
            label = None,
            ax:plt.Axes = None,
            logy=False,
            logx=True,
            xlabel=None,
            ylabel=None,
            color = None,
            allow_constant=True,
            **kwargs):
    if ax is None:
        ax = plt

    if isinstance(xlabel,str):
        xlabel = const(xlabel)
    if isinstance(ylabel,str):
        ylabel = const(ylabel)

    if not callable(color):
        color = const(color)

    if data.shape[0]==1 and allow_constant:
        ax.axhline(y=data[y_axis].to_numpy(),
                   label=label if label is None or isinstance(label, str) else label(data),
                   c='magenta')
    else:
        ax.plot(data[x_axis].to_numpy(),
                data[y_axis].to_numpy(),
                label=label if label is None or isinstance(label, str) else label(data),
                c=color(y_axis,data = data))

    if logy:
        ax.set_yscale('log')
    if logx:
        ax.set_xscale('log')

    if xlabel is None:
        ax.set_xlabel(dName(x_axis))
    elif xlabel:
        ax.set_xlabel(xlabel(x_axis, data=data))

    if ylabel is None:
        ax.set_ylabel(dName(y_axis))
    elif ylabel:
        ax.set_ylabel(ylabel(y_axis, data=data))
'''


'''def addplotOverEpsilon(data, column = 'yeom_mi', print_out=False, label = None, axs:plt.Axes = None, logy=False, agg='mean'):
    if axs is None:
        axs:plt.Axes = plt
    data: pd.DataFrame = (data
            .groupby(['base_dataset', 'method', 'target_epsilon', 'target_model', 'target_dp'],dropna=False)
            .aggregate(['max','mean','std'])
            .reset_index())
    axs.plot(data['target_epsilon'].to_numpy(),data[(column,agg)].to_numpy(), label=label)
    if logy:
        axs.set_yscale('log')
    axs.set_xscale('log')
    axs.set_xlabel(dName('target_epsilon'))
    axs.set_ylabel(dName(column))
    if print_out:
        with FullPandasPrinter(cntRows=False):
            print(data)

def addplotOverEpsilon_old(data, column = 'yeom_mi', print_out=False, label = None, axs:plt.Axes = None, logy=False, agg='mean'):
    if axs is None:
        axs = plt
    data: pd.DataFrame = (data
            .groupby(['base_dataset', 'method', 'target_epsilon', 'target_model', 'target_dp'],dropna=False)
            .aggregate(['max','mean','std'])
            .reset_index())


    data = data.set_index('target_epsilon')
    data[(column,agg)].plot(logy=logy,
                               logx=True,
                               xlabel=dName('target_epsilon'),
                               ylabel=dName(column),
                               label=label,
                            ax=axs)
    if print_out:
        with FullPandasPrinter(cntRows=False):
            print(data)

def plotAggByEpsilonOverPert(data, COLUMN='adapted_yeom_mi_max', success_key='real_test_acc', agg='mean', axs: plt.Axes = None):
    axs = plt if axs is None else axs
    data = reduceToMostSuccessfull(data, key=success_key)

    addplotOverEpsilon(data=data[(data['method'] == 'fusion')], label='fusion', column=COLUMN, logy=True,
                                print_out=True, agg=agg, axs=axs)
    addplotOverEpsilon(data=data[(data['method'] == 'soria')], label='soria', column=COLUMN, logy=True,
                                print_out=True, agg=agg, axs=axs)
    # addplot(data = data[(data['method'] == 'silentK')],label='silentK',column='opt_adapted_yeom_mi_max',print_out=True)
    axs.legend()
    axs.set_title(f'{COLUMN} evolving over epsilon for different pert. algorithms')



def figAggByEpsBothML(data, COLUMN='adapted_yeom_mi_max', **kwargs):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize = (8, 4),constrained_layout=True)#
    assert isinstance(fig,plt.Figure)
    plotAggByEpsilonOverPert(data[data['target_model'] == 'nn'], COLUMN=COLUMN, axs = ax1, **kwargs)
    ax1.set_title('neural network',fontsize=11)
    plotAggByEpsilonOverPert(data[data['target_model'] == 'softmax'], COLUMN=COLUMN, axs = ax2, **kwargs)
    ax2.set_title('softmax',fontsize=11)
    #fig.suptitle(f'{graphics.dName(COLUMN)} evolving over epsilon for different pert. algorithms',fontsize=16)
    return fig

def plotAggByEpsilonOver(data, COLUMN='adapted_yeom_mi_max', success_key='real_test_acc', ax_over_column='method', agg='mean', axs: plt.Axes = None,**kwargs):
    axs = plt if axs is None else axs
    data = reduceToMostSuccessfull(data, key=success_key)
    values = list(data[ax_over_column].unique())
    for value in values:
        addplotOverEpsilon(data=data[(data[ax_over_column] == value)], label=dName(value), column=COLUMN, agg=agg, axs=axs,**kwargs)
    if len(data)>1:
        axs.legend()
    axs.set_title(f'{dName(COLUMN)} evolving over epsilon for different pert. algorithms')

def figAggByEps(data, COLUMN='adapted_yeom_mi_max', fig_over_column='base_dataset', **kwargs):
    datasets = list(data[fig_over_column].unique())
    fig, axes = plt.subplots(nrows=1, ncols=len(datasets), figsize = (4*len(datasets), 4),constrained_layout=True)#
    assert isinstance(fig,plt.Figure)

    for dataset,axs in zip(datasets,axes):
        plotAggByEpsilonOver(data[data[fig_over_column] == dataset], COLUMN=COLUMN, axs = axs, **kwargs)
        axs.set_title(dName(dataset),fontsize=11)
    #fig.suptitle(f'{graphics.dName(COLUMN)} evolving over epsilon for different pert. algorithms',fontsize=16)
    return fig'''