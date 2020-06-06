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

plt.tight_layout()

PERCENTILES = [0,1]
PP_PERCENTILES = ["{0:0.1f}".format(x) for x in PERCENTILES]
PLOTS = []
BONITO_KD = '117mrxzu' #sparkling-terrain-405
JKBC = [
    '2eiadj4y', #eternal-deluge-448 - JKBC1
    '1ywu3vo9', #breezy-cosmos-408 - JKBC2
    '2d84exku', #scarlet-sound-417 - JKBC3
    'j6f2sn3v', #vibrant-puddle-433 - JKBC4
    '1c2vr2my', #playful-oath-434 - JKBC5
]

JKBC_STARTED_FROM = [
    '5916pnqr',
    '2zgf3hhp',
    '142cxqz9',
    '18lbe8gh',
    '2v1c1nsk',
]

JK_Blue   = '#5975A4'
JK_Red    = '#B55D60'
JK_Green  = '#5F9E6E'
JK_Orange = '#CC8963'
JK_Purple = '#867AAB'
JK_Brown  = '#8D7967'
JK_Pink   = '#D195C0'

## COLORBLIND FRIENDLY COLORS FROM https://zenodo.org/record/3381072#.Xtoy9kV_taQ
JK_COLORS = ['#BBCC33', '#77AADD', '#EE8866', '#44BB99', '#99DDFF','#FFAABB', '#EEDD88', '#99DDFF',  '#DDDDDD']
JKBC_MARKERS = ['P', 'v', 'D', 's', 'p']

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
def select_data(data, value, by_column='Tags', drop_bonito=True):
    drop_bonito = False if 'Tags'==by_column and 'BONITO'==value else drop_bonito
    
    indexNames = []
    for index, row in data.iterrows():
        if drop_bonito and 'BONITO' in row['Tags']:
            indexNames.append(index)
            continue
        try:
            if value not in row[by_column]:
                indexNames.append(index)
        except:
            if value != row[by_column]:
                indexNames.append(index)
        
    return data.copy().drop(indexNames)

def remove_ids(data, ids):
    indexNames = []
    for index, row in data.iterrows():
        if row['ID'] in ids:
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

def add_labels(lst, y_offset=10):
    for time, loss, label in zip(*lst):
        ax.annotate(label, xy=(time, loss), ha='center', xytext=(0, y_offset), textcoords='offset pixels')


# -

# # Get Data

# +
data = pd.read_csv("./wandb/random_grid_bonito_kd_temperature.csv")

# Use only finished runs with an valid loss and time_predict
data = data[data.State == 'finished']
data = data[data.valid_loss.notnull()]
data = data[data.time_predict.notnull()]
data['kd_loss'] = [math.isnan(ctc) if math.isnan(ctc) else valid for valid, ctc in zip(data['valid_loss'],data['ctc_loss'])]
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
ax.set_xlabel(TIME_AXIS_LABEL)
ax.set_ylabel(LOSS_AXIS_LABEL)

ax.set_ylim(0.3, .85)
ax.set_xlim(0.1, 1)
ax.grid(True)

ax.plot(*pareto_set[:2], c=JK_COLORS[1], linestyle=':')
ax.scatter(*random_sweep[:2], s=100, c=JK_COLORS[1], label='Mutations', edgecolors='black')
ax.scatter(*bonito[:2], s=100, c=JK_COLORS[0], label='Bonito', edgecolors='black', marker='X')
ax.legend()
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
save_figure('importance')

# # Experiment 2

# ## Dilation

# +
d1 = get_truple(select_data(data, 'DILATION_1_KERNEL'), z='ID')
d2 = get_truple(select_data(data, 'DILATION_2_KERNEL'), z='ID')
d3 = get_truple(select_data(data, 'DILATION_3_KERNEL'), z='ID')

bonito = get_truple(select_data(data, 'BONITO'))
fig, ax = plt.subplots()


ax.set_xlabel(TIME_AXIS_LABEL)
ax.set_ylabel(LOSS_AXIS_LABEL)

ax.grid(True)

ax.scatter(*d1[:2], s=100, label='D=1', c=JK_COLORS[1], edgecolors='black', marker='s')
ax.scatter(*d2[:2], s=100, label='D=2', c=JK_COLORS[2], edgecolors='black', marker='D')
ax.scatter(*d3[:2], s=100, label='D=3', c=JK_COLORS[3], edgecolors='black', marker='P')
ax.scatter(*bonito[:2], s=100, label='Bonito', c=JK_COLORS[0], edgecolors='black', marker='X')
ax.legend()
save_figure('dilation-kernel')
# -

# ## Grouping

# +
grouping_data = select_data(data, 'GROUPING')
grouping_data['model_params.b1_shuffle']
shuffle = get_truple(select_data(grouping_data, True, 'model_params.b1_shuffle'), z='model_params.b1_groups')
noshuffle = get_truple(select_data(grouping_data, False, 'model_params.b1_shuffle'), z='model_params.b1_groups')
bonito = get_truple(select_data(data, 'BONITO'), z='model_params.b1_groups')

fig, ax = plt.subplots()

ax.set_xlabel(TIME_AXIS_LABEL)
ax.set_ylabel(LOSS_AXIS_LABEL)

ax.grid(True)

ax.scatter(*shuffle[:2], s=100, label='Shuffle', c=JK_COLORS[1], edgecolors='black', marker='D')
add_labels(shuffle)

ax.scatter(*noshuffle[:2], s=100, label='No shuffle', c=JK_COLORS[2], edgecolors='black', marker='s')
add_labels(noshuffle)
    
ax.scatter(*bonito[:2], s=100, label='Bonito', c=JK_COLORS[0], edgecolors='black', marker='X')
add_labels(bonito)

ax.legend()
save_figure('grouping')
# -

# ## Kernels

# +
filters = get_truple(select_data(data, 'FILTERS'), z='model_params.b5_filters')
bonito = get_truple(select_data(data, 'BONITO'), z='model_params.b5_filters')

fig, ax = plt.subplots()

ax.set_xlabel(TIME_AXIS_LABEL)
ax.set_ylabel(LOSS_AXIS_LABEL)

ax.grid(True)

ax.scatter(*filters[:2], s=100, label='Shuffle', c=JK_COLORS[1], edgecolors='black')  
add_labels(filters)
ax.scatter(*bonito[:2], s=100, label='Bonito', c=JK_COLORS[0], edgecolors='black', marker='X')
add_labels(bonito)
save_figure('kernels')
# -

# ## Repeats

# +
filters = get_truple(select_data(data, 'REPEAT'), z='model_params.b1_repeat')
bonito = get_truple(select_data(data, 'BONITO'), z='model_params.b1_repeat')

fig, ax = plt.subplots()

ax.set_xlabel(TIME_AXIS_LABEL)
ax.set_ylabel(LOSS_AXIS_LABEL)

ax.grid(True)

ax.scatter(*filters[:2], s=100, c=JK_COLORS[1], edgecolors='black')  
add_labels(filters)
ax.scatter(*bonito[:2], s=100, label='Bonito', c=JK_COLORS[0], edgecolors='black', marker='X')
add_labels(bonito)
save_figure('repeat')
# -

# ## Pareto set final

# +
# Pareto data
#BONITO
bonito = get_truple(select_data(data, 'BONITO'))
bonitokd = get_truple(select_data(data, BONITO_KD, 'ID'))
bonito_labels = ['Bonito', 'Bonito-KD']
bonito_colors = [JK_COLORS[1], JK_COLORS[6]]

# RANDOM AND GRID
random_sweep = get_truple(select_data(data, 'RANDOM_SWEEP'))
grid = get_truple(select_data(data, 'GRID'))
combined_data = pd.concat([select_data(data, 'RANDOM_SWEEP'), select_data(data, 'GRID')])
combined = get_truple(combined_data)
combined_no_dilation_data = remove_ids(combined_data, d1[2]+d2[2]+d3[2])
combinde_no_dilation = get_truple(combined_no_dilation_data)
# KD

kd_data = select_data(select_data(data, 'KNOWLEDGE_DISTILLATION'), 4., 'kd_temperature')
kd_data_filtered = remove_ids(select_data(select_data(data, 'KNOWLEDGE_DISTILLATION'), 4., 'kd_temperature'), JKBC)

kd = get_truple(kd_data, z='kd_alpha')
kd_filtered = get_truple(kd_data_filtered, z='started_from')
kd_times_dict = {d['ID']:d['time_predict'] for _, d in data.iterrows()}
kd_times = [kd_times_dict[x] for x in kd_filtered[2]]

# JKBC
jkbc = list(map(lambda x: get_truple(select_data(data, x, 'ID')), JKBC))
jkbc_started_from = list(map(lambda x: get_truple(select_data(data, x, 'ID')), JKBC_STARTED_FROM))
jkbc_labels = ['JKBC-1', 'JKBC-2', 'JKBC-3', 'JKBC-4', 'JKBC-5']
jkbc_colors = [JK_COLORS[0], JK_COLORS[2], JK_COLORS[3], JK_COLORS[4], JK_COLORS[5]]

pareto_set, _ = split_pareto_set(*combined)

# +
fig, ax = plt.subplots()
ax.set_xlabel(TIME_AXIS_LABEL)
ax.set_ylabel(LOSS_AXIS_LABEL)

ax.set_ylim(0.3, .85)
ax.set_xlim(0.1, 1)
ax.grid(True)


ax.plot(*pareto_set[:2], c=JK_COLORS[1], linestyle=':')
ax.scatter(*combinde_no_dilation[:2], s=30, c='grey')
ax.scatter(*bonito[:2], s=100, c=JK_COLORS[0], edgecolors='black', marker='X', label='Bonito')
ax.scatter(*d1[:2], s=60, label='Dilation', c='black', edgecolors='black', marker='$D$')
ax.scatter(*d2[:2], s=60, c='black', edgecolors='black', marker='$D$')
ax.scatter(*d3[:2], s=60, c='black', edgecolors='black', marker='$D$')

for x, label, m in zip(jkbc_started_from, jkbc_labels, JKBC_MARKERS):
    ax.scatter(*x[:2], s=100, c=JK_COLORS[1], label=label, edgecolors='black', marker=m)

ax.legend()
save_figure('Pareto-set-2')

# +
fig, ax = plt.subplots()
ax.set_xlabel(TIME_AXIS_LABEL)
ax.set_ylabel(LOSS_AXIS_LABEL)

ax.set_ylim(0.2, 1)
ax.set_xlim(0.1, 1)
ax.grid(True)

ax.scatter(*combined[:2], s=30, c='grey')


ax.scatter(*bonito[:2], s=100, c=JK_COLORS[0], edgecolors='black', marker='X', label='Bonito')
ax.scatter(*bonitokd[:2], s=100, c=JK_COLORS[3], edgecolors='black', marker='X')

for x, label, m in zip(jkbc_started_from, jkbc_labels, JKBC_MARKERS):
    ax.scatter(*x[:2], s=100, c=JK_COLORS[1], label=label, edgecolors='black', marker=m)
    
for x, time, label, m in zip(jkbc, jkbc_started_from, jkbc_labels, JKBC_MARKERS):
    ax.scatter(time[0], x[1], s=100, c=JK_COLORS[2], edgecolors='black', marker=m)

ax.scatter(kd_times, kd_filtered[1], s=30, c='black', marker='x', label='KD')

ax.legend()
save_figure('Pareto-set-final')

# +
ax = kd_data.boxplot(column='valid_loss', by='kd_alpha')
fig = ax.get_figure()
ax.set_title('')
fig.suptitle('')

ax.set_ylim(0.2, 1)
ax.set_xlabel('Alpha')
ax.set_ylabel(LOSS_AXIS_LABEL)

save_figure('alpha-boxplot')
# -

plt.show()

zip_figures('plots')


