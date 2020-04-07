import os

import torch

import jkbc.utils.preprocessing as prep

def save_training_data(data, folder):
    __make_dir(folder)
    torch.save(torch.Tensor(data.x), f'{folder}/x.pt')
    torch.save(torch.Tensor(data.x_lengths), f'{folder}/x_lengths.pt')
    torch.save(torch.Tensor(data.y), f'{folder}/y.pt')
    torch.save(torch.Tensor(data.y_lengths), f'{folder}/y_lengths.pt')
        
def load_training_data(folder): 
    x = torch.load(f'{folder}/x.pt')
    x_lengths = torch.load(f'{folder}/x_lengths.pt')
    y = torch.load(f'{folder}/y.pt')
    y_lengths = torch.load(f'{folder}/y_lengths.pt')
    
    return prep.ReadObject(None, x, x_lengths, y, y_lengths, None)

def __make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)