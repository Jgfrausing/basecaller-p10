import shutil
import torch
import torch.nn as nn

import numpy as np

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


def test_generate_and_save_y_teacher_with_load():
  # Arrange
  teacher_model = TeacherModel()
  teacher_name = "teacher_kd"
  
  x = torch.rand((N, FEATURES, WS))
  
  # Act
  kd.generate_and_save_y_teacher(FOLDER_PATH, teacher_name, x, teacher_model, bs = 2, ws = WS)
  y_teacher_f = f.read_kd_y_teacher_from_feather_file(FOLDER_PATH, teacher_name, OUT_SIZE, ALPHABET_SIZE)
  
  # Assert
  assert y_teacher_f.shape == (N, OUT_SIZE, ALPHABET_SIZE)
  ## Consider checking that y_teacher_f == y_teacher_kd
  
  
def test_generate_and_save_y_teacher_correct_ordering():
  # Arrange
  teacher_model = TeacherModel()
  teacher_name = "teacher_ordering"
  x = torch.randint(10, (N, FEATURES, WS), dtype=torch.float)
  y_teacher = teacher_model(x).detach()
  
  # Act 
  kd.generate_and_save_y_teacher(FOLDER_PATH, teacher_name, x, teacher_model, bs = 2, ws = WS)
  y_teacher_f = torch.tensor(f.read_kd_y_teacher_from_feather_file(FOLDER_PATH, teacher_name, OUT_SIZE, ALPHABET_SIZE)).detach()
  
  # Assert
  DECIMAL_PRECISION = 5
  np.testing.assert_almost_equal(y_teacher.numpy(), y_teacher_f.numpy(), decimal=DECIMAL_PRECISION)
  
  
class TeacherModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.lin = nn.Linear(WS * FEATURES, OUT_SIZE * ALPHABET_SIZE)
    
  def forward(self, xb):
    bs = xb.size(0)
    xb = xb.view(bs, -1)
    xb = self.lin(xb)
    return xb.view((bs, OUT_SIZE, ALPHABET_SIZE))
  