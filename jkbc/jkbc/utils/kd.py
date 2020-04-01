import torch
import torch.nn as nn

import jkbc.types as t
import jkbc.utils.files as f

def generate_and_save_kd_data_from_teacher(folder_path: t.PathLike, x, y, y_lengths, teacher_model: nn.Module, bs: int, ws:int = 300):
    '''Uses a teacher_model to create and save (x, y, y_lengths, y_teacher) to feather files.
     Note: y_teacher.shape == (window_count, dim_out * alphabet_size)
    '''
    assert x.shape[0] == y.shape[0] == len(y_lengths), "The x, y, and y_lengths should match in size"
    
    window_count = x.shape[0]
    windows_processed = 0
    
    y_teacher = []
    
    while windows_processed < window_count:
        batch = x[windows_processed:windows_processed+bs]
        y_teacher.append(teacher_model(batch).detach().cpu())
        windows_processed += bs
    
    # Basically: dim_out * alphabet_size
    teacher_logit_size = y_teacher[0].shape[1] * y_teacher[0].shape[2]
    
    # Convert to 2d numpy arrays on cpu
    x = x.view((window_count, ws)).cpu().numpy()
    y = y.numpy()
    y_teacher = torch.cat(y_teacher).view((window_count, teacher_logit_size)).numpy()
    
    f.write_kd_data_to_feather_file(folder_path, (x, y, y_lengths, y_teacher))
    
    return x, y, y_lengths, y_teacher