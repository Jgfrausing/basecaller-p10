import argparse
import os

# +
import matplotlib.pyplot as plt
import pandas as pd
import torch

np.set_printoptions(precision=2)


# +
def plot_figure(data, row_labels, col_labels, feature='', normalise_column=False):
    def _normalise_column(data):
        col_normalised = data.clone().t()
        for col in range(len(col_normalised)):
            values = normalise(col_normalised[col])
            for row in range(len(col_normalised[col])):
                col_normalised[col,row] = values[row]
        return col_normalised.t()
    
    colours = _normalise_column(data) if normalise_column else data
    
    # Create figure
    fig = plt.figure()
    ax = fig.add_subplot()
    cax = ax.matshow(colours, vmin=0, vmax=1, cmap='summer')
    fig.colorbar(cax)

    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_ylabel(feature)
    
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_xlabel('Accuracy vs Speed\n>>-->>-->>')
    
    for (i, j), z in np.ndenumerate(data):
        ax.text(j, i, '{:0.1f}'.format(z), ha='center', va='center')

    #plt.savefig(output)
    plt.show()
    
def normalise(lst):
    min_, max_ = min(lst), max(lst)
    return [(i-min_)/(max_-min_) for i in lst]

# +
percentiles = [0.1*v for v in range(11)]
start = 10
rows = 10
cols =len(percentiles)

data = pd.read_csv("../nbs/experiments/wandb/export-full-clean.csv")
accuracy = [1-x for x in normalise(data['read_identity'][start:start+rows])]
time_predict = normalise(data['Runtime'][start:start+rows])
identifier = data['model_params.c3_kernel'][start:start+rows]

matrix = torch.zeros(rows, cols)

# val = alp*time+(1-alp)acc
for row in range(rows):
    for col in range(cols):
        alpha = percentiles[col]
        matrix[row,col] = time_predict[row]*alpha + (1-alpha)*read_identity[row]
plot_figure(matrix, identifier, ["{0:0.1f}".format(x) for x in percentiles], "Kernel")
# -

time_predict, accuracy
