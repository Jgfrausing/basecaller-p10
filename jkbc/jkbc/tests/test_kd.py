import shutil
import torch
import torch.nn as nn

import jkbc.utils.kd as kd
import jkbc.utils.files as f

WS = 300
N = 10
ALPHABET_SIZE = 5
OUT_SIZE = 20
FEATURES = 1
MAX_LABEL_LEN = 15

FOLDER_PATH = "teacher_feather_test_2"


def teardown_function(function):
    '''This function is called after each test function in this file.'''
    shutil.rmtree(FOLDER_PATH)


def test_generate_and_save_kd_data_from_teacher_with_load():
  # Arrange
  teacher_model = TeacherModel()
  
  x = torch.rand((N, FEATURES, WS))
  y = torch.randint(10, (N, MAX_LABEL_LEN))
  y_lengths = [3 for _ in range(N)]
  
  # Act
  x_kd, y_kd, y_lengths_kd, y_teacher_kd = kd.generate_and_save_kd_data_from_teacher(FOLDER_PATH, x, y, y_lengths, teacher_model, bs = 2, ws = WS)
  x_f, y_f, y_lengths_f, y_teacher_f = f.read_kd_data_from_feather_file(FOLDER_PATH)
  
  # Assert
  assert x_f.shape == (N, WS)
  assert y_f.shape == (N, MAX_LABEL_LEN)
  assert len(y_lengths_f) == N
  assert y_teacher_f.shape == (N, OUT_SIZE, ALPHABET_SIZE)
  ## Consider checking that y_teacher_f == y_teacher_kd
  
  
class TeacherModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.lin = nn.Linear(WS * FEATURES, OUT_SIZE * ALPHABET_SIZE)
    
  def forward(self, xb):
    bs = xb.size(0)
    xb = xb.view(bs, -1)
    xb = self.lin(xb)
    return xb.view((bs, OUT_SIZE, ALPHABET_SIZE))