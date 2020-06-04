import jkbc.utils.preprocessing as pp
import numpy as np

def test_add_label_padding_simple():
  # Arrange
  labels = [[1, 2, 3], [2, 3]]
  fixed_label_len = 5

  # Act
  result = pp.add_label_padding(labels, fixed_label_len)

  # Assert
  np.testing.assert_array_equal(result, [[1, 2, 3, 0, 0], [2, 3, 0, 0, 0]])


''''
def test_succesful_read():
    elm = pp.SignalCollection("../mapped_umi16to9.hdf5")[264]
    assert len(elm.x) > 0
    assert len(elm.y) > 0
    assert len(elm.reference) > 0
'''