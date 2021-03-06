from typing import List, Dict

import jkbc.utils.postprocessing as pp
import pytest


def test_assembles_correctly():
    # Arrange
    reads: List[str] = ['AAGGCCTAGCT',
                        'AGGCCTAGCAA', 'GGCCTAGCTC', 'AAAGGCCTAGT']
    window_size: int = 100  # arbitrarily chosen
    stride: int = 1  # arbitrarily chosen
    alphabet: Dict[int, str] = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}

    # Act
    result: str = pp.assemble(reads, window_size, stride, alphabet)

    # Assert
    assert "AAGGCCTAGCTA" == result

def test_calc_accuracy_identical():
    # Arrange
    s: str = 'AAGGCCTTGGCC'

    # Act
    accuracy = pp.calc_accuracy(s, s)

    # Assert
    assert accuracy == pytest.approx(100)

'''
def test_calc_accuracy_bad():
    # Arrange
    s1: str = 'AAGGCC'
    s2: str = 'GTTTTT'

    # Act
    accuracy = pp.calc_accuracy(s1, s2)

    # Assert
    assert accuracy == pytest.approx(0)
'''


def test_convert_idx_to_base_sequence_simple():
    # Arrange
    idx = [0, 2, 1, 3, 0]
    alphabet = list("ABCD")

    # Act
    result =pp.convert_idx_to_base_sequence(idx, alphabet)

    # Assert
    assert result == "ACBDA"