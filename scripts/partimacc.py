# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

# +
import matplotlib.pyplot as plt
import pandas as pd
import torch
import jkbc.utils.matrix_plot as mp
import sys
import zipfile
import io
import os

PERCENTILES = [0.1*v for v in range(11)]
PP_PERCENTILES = ["{0:0.1f}".format(x) for x in PERCENTILES]
PLOTS = []


# +
def zip_figures(name):
    s = list(set(PLOTS))
    with zipfile.ZipFile(f'{name}.zip', mode="w") as zf:
        for plot in s:
            fn = f'{plot}.png'
            zf.write(fn)
            os.remove(fn)
    PLOTS.clear()
    
def save_figure(name):
    PLOTS.append(name)
    plt.savefig(name)


# -

def average(identifiers, x, y):
    def _average(lst):
        return sum(lst)/len(lst)

    combined = zip(identifier, x, y)

    identifier_dict = {}
    for i, x_, y_ in combined:
        if i in identifier_dict:
            identifier_dict[i][0].append(x_)
            identifier_dict[i][1].append(y_)
        else:
            identifier_dict[i] = [[x_], [y_]]

    average_combined = []
    for i, lst in identifier_dict.items():
        avg_x = _average(lst[0])
        avg_y = _average(lst[1])
        average_combined.append((i,avg_x,avg_y))

    average_combined.sort(key=lambda tup: tup[0])
    identifiers = [tup[0] for tup in average_combined]
    avg_x = [tup[1] for tup in average_combined]
    avg_y = [tup[2] for tup in average_combined]
    
    return identifiers, avg_x, avg_y


def select_tags(data, tags):
    indexNames = []
    for index, row in data.iterrows():
        if row['Tags'] not in tags:
            indexNames.append(index)
    return data.drop(indexNames)


def get_matrix(identifiers, time, accuracy):
    rows = len(identifiers)
    cols = len(PERCENTILES)
    matrix = torch.zeros(rows, cols)

    # val = alp*time+(1-alp)acc
    for row in range(rows):
        for col in range(cols):
            alpha = PERCENTILES[col]
            
            matrix[row,col] = int((time[row]*alpha + (1-alpha)*accuracy[row])*100)
            
    return matrix
features = [
    {
        'Name':'Kernel size (Dilation 1)',
        'Tags':['DILATION_1_KERNEL'],
        'Params':['model_params.b5_dilation', 'model_params.b5_kernel']}, 
    {
        'Name':'Kernel size (Dilation 2)',
        'Tags':['DILATION_2_KERNEL'],
        'Params':['model_params.b5_dilation', 'model_params.b5_kernel']}, 
    {
        'Name':'Kernel size (Dilation 3)',
        'Tags':['DILATION_3_KERNEL'],
        'Params':['model_params.b5_dilation', 'model_params.b5_kernel']}, 
    {
        'Name':'Shuffle and Grouping',
        'Tags':['GROUPING'],
        'Params': ['model_params.b5_shuffle', 'model_params.b5_groups']},
    {
        'Name':'Number of Kernels',
        'Tags':['FILTERS'],
        'Params':['model_params.b5_filters']},
    {
        'Name':'Repeat',
        'Tags':['REPEAT'],
        'Params':['model_params.b5_repeat']}
]
data = pd.read_csv("../nbs/experiments/wandb/export-2020-05-25.csv")
data['time_predict'] = mp.normalise(data['time_predict'])
data['valid_loss'] = mp.normalise(data['valid_loss'])

best_values = torch.zeros(len(features), len(PERCENTILES))
for index in range(len(features)):
    run = features[index]
    # Selecting data
    run_data = select_tags(data, run['Tags']).sort_values(by=run['Params'])
    identifiers = list(zip(*(map(lambda x: list(run_data[x]), run['Params']))))
    accuracy = list(run_data['valid_loss'])
    time = list(run_data['time_predict'])
    matrix = get_matrix(identifiers, time, accuracy)
    
    # Get plot
    mp.get_matrix_plot(matrix, identifiers, PP_PERCENTILES, run['Name'])
    save_figure(run['Name'])
    # Save best of each percentile
    best_values[index] = torch.min(matrix, dim=0).values

mp.get_matrix_plot(best_values, [x['Name'] for x in features], PP_PERCENTILES, 'Hyper parameters')
save_figure(run['Name'])

# +
random_sweep = select_tags(data, ['RANDOM_SWEEP']).sort_values(by='valid_loss')
accuracy = list(random_sweep['valid_loss'])
time = list(random_sweep['time_predict'])

def split_pareto_set(x, y):
    current_best = 1000
    pareto_set = []
    others = []
    zipped = list(zip(x, y))
    zipped.sort(key=lambda tup: tup[0])
    for x_, y_ in zipped:
        if y_ < current_best:
            current_best = y_
            pareto_set.append((x_, y_))
        else:
            others.append((x_, y_))
            
    return list(zip(*pareto_set)), list(zip(*others))

pareto_set, others = split_pareto_set(time, accuracy)
fig, ax = plt.subplots()

ax.set_xlabel('Time')
ax.set_ylabel('Loss')
ax.set_title('Pareto Frontier', fontsize=12)

ax.set_ylim(0, 1)
ax.set_xlim(0, 1)
ax.grid(True)

ax.scatter(*others)
ax.scatter(*pareto_set)
ax.plot(*pareto_set)

save_figure('Pareto-frontier')
# -

zip_figures('plots')


