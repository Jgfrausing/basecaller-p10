import torch
import torch.nn as nn

import jkbc.types as t
import jkbc.utils.torch_files as f

def generate_and_save_y_teacher(folder_path: t.PathLike, teacher_name: str, x, teacher_model: nn.Module, bs: int) -> None:
    '''Uses a teacher_model to create and save y_teacher to feather files.'''
    
    window_count = x.shape[0]
    windows_processed = 0
    
    y_teacher = []
    
    while windows_processed < window_count:
        batch = x[windows_processed:windows_processed+bs]
        y_teacher.append(teacher_model(batch).detach().cpu())
        windows_processed += bs
    
    y_teacher = torch.cat(y_teacher).numpy()
    
    f.save_teacher_data(y_teacher, folder_path, teacher_name)