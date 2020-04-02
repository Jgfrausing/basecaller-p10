import torch
import torch.nn as nn

import jkbc.types as t
import jkbc.utils.files as f

def generate_and_save_y_teacher(folder_path: t.PathLike, teacher_name: str, x, teacher_model: nn.Module, bs: int, ws:int = 300) -> None:
    '''Uses a teacher_model to create and save y_teacher to feather files.'''
    
    window_count = x.shape[0]
    windows_processed = 0
    
    y_teacher = []
    
    while windows_processed < window_count:
        batch = x[windows_processed:windows_processed+bs]
        y_teacher.append(teacher_model(batch).detach().cpu())
        windows_processed += bs
    
    x = x.view((window_count, ws)).cpu().numpy()
    y_teacher = torch.cat(y_teacher).numpy()
    
    f.write_kd_y_teacher_to_feather_file(folder_path, teacher_name, y_teacher)