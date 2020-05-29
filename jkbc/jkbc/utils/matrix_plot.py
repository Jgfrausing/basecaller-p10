import matplotlib.pyplot as plt
import torch
import numpy as np


# +
def get_matrix_plot(data, row_labels, col_labels, feature='', normalise_column=False, vmin=0, vmax=100, round_ints=True, legend=False):
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
    fig, ax = plt.subplots(figsize=(10,10))
    cax = ax.matshow(colours, vmin=vmin, vmax=vmax, cmap='rainbow', interpolation='lanczos')
    
    if legend:
        fig.colorbar(cax)

    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_ylabel(feature)
    
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_xlabel('Accuracy vs Speed\n>>-->>-->>')

    for (i, j), z in np.ndenumerate(data):
        val = int(z) if round_ints else "{0:0.1f}".format(z)            
        ax.text(j, i, val, ha='center', va='center')

    plt.tight_layout()
    return plt
    
def normalise(lst):
    min_, max_ = min(lst), max(lst)
    return [(i-min_)/(max_-min_) for i in lst]
