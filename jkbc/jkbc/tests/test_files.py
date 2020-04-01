import shutil
import pathlib as pl

import pytest
import numpy as np

import jkbc.utils.files as f


path_base = pl.Path("test_files")

def teardown_function(function):
    '''This function is called after each test function in this file.'''
    shutil.rmtree(path_base)

    
def test_write_data_to_feather_file__no_error():
  # Arrange
  N = 10
  x = [[.1] for _ in range(N)]
  y = list(range(N))
  y_lengths = [1 for _ in range(N)]
  
  # Act + Assert
  try:
    f.write_data_to_feather_file(path_base, (x, y, y_lengths))
  except:
    pytest.fail("Failed during write_data_to_feather_file")
    

def test_write_and_read_data_to_feather_file():
  # Arrange
  N = 10
  x1 = [[.1] for _ in range(N)]
  y1 = list(range(N))
  y_lengths1 = [1 for _ in range(N)]
  
  # Act
  try:
    f.write_data_to_feather_file(path_base, (x1, y1, y_lengths1))
    (x2, y2, y_lengths2) = f.read_data_from_feather_file(path_base)
  except:
    pytest.fail("Failed during write or read")
    
  # Assert
  contains_floats = True
  assert_eq_lists(x1, x2, contains_floats)
  assert_eq_lists(y1, y2)
  assert_eq_lists(y_lengths1, y_lengths2)
  

 
def test_write_kd_data_to_feather_file__no_error():
  # Arrange
  N = 10
  x = [[.1, .2, .3] for _ in range(N)]
  y = list(range(N))
  y_lengths = [1 for _ in range(N)]
  y_teacher = np.random.rand(N, 20)
  
  # Act + Assert
  try:
    f.write_kd_data_to_feather_file(path_base, (x, y, y_lengths, y_teacher))
  except:
    pytest.fail("Failed during write_data_to_feather_file")


def test_kd_write_and_read_data_to_feather_file():
  # Arrange
  N = 10
  OUT_SIZE = 7
  ALPHABET_SIZE = 5
  
  x1 = [[.1, .2, .3] for _ in range(N)]
  y1 = list(range(N))
  y_lengths1 = [1 for _ in range(N)]
  y_teacher1 = np.random.rand(N, OUT_SIZE * ALPHABET_SIZE)
  
  # Act
  try:
    f.write_kd_data_to_feather_file(path_base, (x1, y1, y_lengths1, y_teacher1))
    (x2, y2, y_lengths2, y_teacher2) = f.read_kd_data_from_feather_file(path_base)
  except:
    pytest.fail("Failed during write or read")
    
  # Assert
  contains_floats = True
  assert_eq_lists(x1, x2, contains_floats)
  assert_eq_lists(y1, y2)
  assert_eq_lists(y_lengths1, y_lengths2)
  assert y_teacher2.shape == (N, OUT_SIZE, ALPHABET_SIZE)
  y_teacher1 = y_teacher1.reshape(N, OUT_SIZE, ALPHABET_SIZE)
  np.testing.assert_almost_equal(y_teacher1.reshape(N, OUT_SIZE, ALPHABET_SIZE), y_teacher2)
    
    
def assert_eq_lists(l1, l2, contains_floats: bool = False):
  assert len(l1) == len(l2)
  if contains_floats:
    assert all([e1 == pytest.approx(e2) for e1, e2 in zip(l1, l2)])
  else:
    assert all([e1 == e2 for e1, e2 in zip(l1, l2)])
  
