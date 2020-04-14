import os

import torch

import jkbc.utils.preprocessing as prep

def save_training_data(data, folder):
    __make_dir(folder)
    torch.save(torch.Tensor(data.x), f'{folder}/x.pt')
    torch.save(torch.Tensor(data.x_lengths), f'{folder}/x_lengths.pt')
    torch.save(torch.Tensor(data.y), f'{folder}/y.pt')
    torch.save(torch.Tensor(data.y_lengths), f'{folder}/y_lengths.pt')

def save_teacher_data(teacher, data_path, name):
    teacher_folder = f"{data_path}/teachers";
    __make_dir(teacher_folder)
    torch.save(torch.Tensor(teacher), f'{teacher_folder}/{name}.pt')

def load_training_data(folder): 
    x = torch.load(f'{folder}/x.pt')
    x_lengths = torch.load(f'{folder}/x_lengths.pt')
    y = torch.load(f'{folder}/y.pt')
    y_lengths = torch.load(f'{folder}/y_lengths.pt')
    
    return prep.ReadObject(None, x, x_lengths, y, y_lengths, None)

def load_training_data_with_teacher(folder, teacher_name):
    data = load_training_data(folder)
    teacher = load_teacher_data(folder, teacher_name)
    return data, teacher

def load_teacher_data(folder, teacher_name):
    return torch.load(f'{folder}/teachers/{teacher_name}.pt')


def __make_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)
