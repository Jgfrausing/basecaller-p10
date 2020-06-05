# %reload_ext autoreload
# %autoreload 2
# %matplotlib inline

# +
import matplotlib.pyplot as plt
import pandas as pd
import torch
import sys
import zipfile
import io
import os
import math
from cycler import cycler

PERCENTILES = [0,1]
PP_PERCENTILES = ["{0:0.1f}".format(x) for x in PERCENTILES]
PLOTS = []
BONITOS = [    
    '2j9fzbx4', #swept-durian-82 - Bonito
    '117mrxzu', #sparkling-terrain-405 - Bonito_KD
]
JKBC = [
    '2eiadj4y', #eternal-deluge-448 - JKBC1
    '1ywu3vo9', #breezy-cosmos-408 - JKBC2
    '2d84exku', #scarlet-sound-417 - JKBC3
    'j6f2sn3v', #vibrant-puddle-433 - JKBC4
    '1c2vr2my', #playful-oath-434 - JKBC5
]
JK_Blue   = '#5975A4'
JK_Red    = '#B55D60'
JK_Green  = '#5F9E6E'
JK_Orange = '#CC8963'
JK_Purple = '#867AAB'
JK_Brown  = '#8D7967'
JK_Pink   = '#D195C0'

JK_COLORS = [JK_Blue, JK_Red, JK_Green, JK_Orange, JK_Purple, JK_Brown, JK_Pink]
COLOR_CYCLER = cycler(color=JK_COLORS)

TIME_AXIS_LABEL = 'Prediction Time'
LOSS_AXIS_LABEL = 'CTC Loss'


# +
def zip_figures(name):
    s = list(set(PLOTS))
    with zipfile.ZipFile(f'{name}.zip', mode="w") as zf:
        for plot in s:
            fn = f'{plot}.png'
            zf.write(fn)
            #os.remove(fn)
    PLOTS.clear()
    
def save_figure(name, transparent=True):
    PLOTS.append(name)
    plt.savefig(name, transparent=transparent, bbox_inches='tight', pad_inches=0)


# +
def select_data(data, value, by_column='Tags'):
    indexNames = []
    for index, row in data.iterrows():
        if type(value) is bool:
            if value != row[by_column]:
                indexNames.append(index)
            else:
                continue
        elif value not in row[by_column]:
            indexNames.append(index)
    return data.copy().drop(indexNames)

def get_truple(data, x='time_predict',y='valid_loss', z='Name'):
    copied = data.sort_values(by=y).copy()
    time = list(copied[x])
    loss =  list(copied[y])
    identifier =  list(copied[z])
    return time, loss, identifier

def split_pareto_set(x, y, identifier):
    current_best = 1000
    pareto_set = []
    others = []
    zipped = list(zip(x, y, identifier))
    zipped.sort(key=lambda tup: tup[0])
    for x_, y_, id_ in zipped:
        if y_ < current_best:
            current_best = y_
            pareto_set.append((x_, y_, id_))
        else:
            others.append((x_, y_, id_))
            
    return list(zip(*pareto_set)), list(zip(*others))

def normalise(lst, max_val):
    return [(i/max_val) for i in lst]

def add_labels(lst):
    for time, loss, label in zip(*lst):
        ax.annotate(label, xy=(time, loss), ha='center', xytext=(0, 10), textcoords='offset pixels')


# -

# # Get Data

# +
data = pd.read_csv("./wandb/random_grid_bonito_kd_temperature.csv")

# Use only finished runs with an valid loss and time_predict
data = data[data.State == 'finished']
data = data[data.valid_loss.notnull()]
data = data[data.time_predict.notnull()]
data['valid_loss'] = [valid if math.isnan(ctc) else ctc for valid, ctc in zip(data['valid_loss'],data['ctc_loss'])]
#data = data[data.valid_loss < 0.3]

max_time = 6.697
max_loss = 0.3864
data['time_predict'] = normalise(data['time_predict'], max_time)
data['valid_loss'] = normalise(data['valid_loss'], max_loss)
# -

# # Experiment 1

# ## Pareto set 1

# +
bonito = get_truple(select_data(data, 'BONITO'))
random_sweep = get_truple(select_data(data, 'RANDOM_SWEEP'))
pareto_set, others = split_pareto_set(*random_sweep)

fig, ax = plt.subplots()
ax.set_prop_cycle(COLOR_CYCLER)
ax.set_xlabel(TIME_AXIS_LABEL)
ax.set_ylabel(LOSS_AXIS_LABEL)

ax.set_ylim(0.3, .85)
ax.set_xlim(0.1, 1)
ax.grid(True)

ax.plot(*pareto_set[:2], c=JK_COLORS[0], linestyle=':')
ax.scatter(*random_sweep[:2], s=100, c=JK_COLORS[0], edgecolors='black')
ax.scatter(*bonito[:2], s=100, c=JK_COLORS[1], edgecolors='black')
print(pareto_set[2])
save_figure('Pareto-set-1')
# -

# ## Importance

importance = pd.read_csv('wandb/importance.csv')
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.barh(importance['hp'], importance['importance'], color=JK_Blue, edgecolor='black')
ax1.set_title("Importance")
correlation_colors = [JK_Green if x < 0 else JK_Red for x in importance['correlation']]
ax2.barh(importance['hp'], importance['correlation'], tick_label="", color=correlation_colors, edgecolor='black')
ax2.set_title("Correlation")
fig.tight_layout()
plt.savefig('importance.png')

# # Experiment 2

# ## Dilation

# +
d1 = get_truple(select_data(data, 'DILATION_1_KERNEL'))
d2 = get_truple(select_data(data, 'DILATION_2_KERNEL'))
d3 = get_truple(select_data(data, 'DILATION_3_KERNEL'))

bonito = get_truple(select_data(data, 'BONITO'))
fig, ax = plt.subplots()

ax.set_xlabel('Kernel size')
ax.set_ylabel(LOSS_AXIS_LABEL)

ax.grid(True)

ax.scatter(*d1[:2], s=100, label='D=1', c=JK_COLORS[0], edgecolors='black')
ax.scatter(*d2[:2], s=100, label='D=2', c=JK_COLORS[2], edgecolors='black')
ax.scatter(*d3[:2], s=100, label='D=3', c=JK_COLORS[3], edgecolors='black')
ax.scatter(*bonito[:2], s=100, label='Bonito', c=JK_Red, edgecolors='black')
ax.legend()
save_figure('dilation-kernel')
# -

# ## Grouping

# +
grouping_data = select_data(data, 'GROUPING')
grouping_data['model_params.b1_shuffle']
shuffle = get_truple(select_data(grouping_data, True, 'model_params.b1_shuffle'), z='model_params.b1_groups')
noshuffle = get_truple(select_data(grouping_data, False, 'model_params.b1_shuffle'), z='model_params.b1_groups')
bonito = get_truple(select_data(grouping_data, 'BONITO'), z='model_params.b1_groups')

fig, ax = plt.subplots()

ax.set_xlabel(TIME_AXIS_LABEL)
ax.set_ylabel(LOSS_AXIS_LABEL)

ax.grid(True)

ax.scatter(*shuffle[:2], s=100, label='Shuffle', c=JK_COLORS[0], edgecolors='black')
add_labels(shuffle)

ax.scatter(*noshuffle[:2], s=100, label='No shuffle', c=JK_COLORS[2], edgecolors='black')
add_labels(noshuffle)
    
ax.scatter(*bonito[:2], s=100, label='Bonito', c=JK_Red, edgecolors='black')
add_labels(bonito)

ax.legend()
save_figure('grouping')
# -

# ## Pareto set 2

# +
bonito = get_truple(select_data(data, 'BONITO'))
random_sweep = get_truple(select_data(data, 'RANDOM_SWEEP'))
grid = get_truple(select_data(data, 'GRID'))
pareto_set, _ = split_pareto_set(*random_sweep)
combined = get_truple(pd.concat([select_data(data, 'RANDOM_SWEEP'), select_data(data, 'GRID')]))
pareto_set_new, _ = split_pareto_set(*combined)


fig, ax = plt.subplots()
ax.set_prop_cycle(COLOR_CYCLER)
ax.set_xlabel(TIME_AXIS_LABEL)
ax.set_ylabel(LOSS_AXIS_LABEL)

ax.set_ylim(0.3, .85)
ax.set_xlim(0.1, 1)
ax.grid(True)

ax.plot(*pareto_set_new[:2], c='black', linestyle='-')
ax.plot(*pareto_set[:2], c=JK_COLORS[0], linestyle=':')
ax.scatter(*random_sweep[:2], s=30, c=JK_COLORS[0], edgecolors='black')
ax.scatter(*grid[:2], s=100, c=JK_COLORS[2], edgecolors='black')
ax.scatter(*bonito[:2], s=100, c=JK_COLORS[1], edgecolors='black')
print(pareto_set_new[2])
save_figure('Pareto-set-2')
# -

zip_figures('plots')
