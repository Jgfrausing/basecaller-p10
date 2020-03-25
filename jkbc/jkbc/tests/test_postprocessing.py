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

def test_calc_sequence_error_metrics_identical():
    # Arrange
    s: str = 'AAGGCCTTGGCC'

    # Act
    rates = pp.calc_sequence_error_metrics(s, s)

    # Assert
    assert rates.identity  == pytest.approx(1)
    assert rates.error     == pytest.approx(0)
    assert rates.mismatch  == pytest.approx(0)
    assert rates.deletion  == pytest.approx(0)
    assert rates.insertion == pytest.approx(0)

def test_calc_sequence_error_metrics_third_mismatch():
    # Arrange
    s1: str = 'AAGGCC'
    s2: str = 'AATTCC'

    # Act
    rates = pp.calc_sequence_error_metrics(s1, s2)

    # Assert
    assert rates.mismatch == pytest.approx(1/3)

def test_calc_sequence_error_metrics_different():
    # Arrange
    s1: str = 'AAGGCCTT'
    s2: str = 'AATTCC'

    # Act
    rates = pp.calc_sequence_error_metrics(s1, s2)

    # Assert
    assert rates.mismatch == pytest.approx(1/4)
    assert rates.insertion == pytest.approx(0)
    assert rates.deletion == pytest.approx(1/4)
    assert rates.identity == pytest.approx(1 - 1/4 - 1/4) # 1 - deletions - mismatch
    assert rates.error == pytest.approx(1/4 + 0 + 1/4) # mismatch + insertion + deletion


def test_convert_idx_to_base_sequence_simple():
    # Arrange
    idx = [0, 2, 1, 3, 0]
    alphabet = "ABCD"

    # Act
    result =pp.convert_idx_to_base_sequence(idx, alphabet)

    # Assert
    assert result == "ACBDA"