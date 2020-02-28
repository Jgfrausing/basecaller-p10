from typing import List, Dict

import jkbc.utils.postprocessing as pp


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

def test_calcs_correct_identity():
    # Arrange
    s1: str = 'AAGGCC'
    s2: str = 'AAAGGCCA'

    # Act
    rates = pp.calc_sequence_error_metrics(s1, s2)

    # Assert
    assert rates.identity == 1

def test_convert_idx_to_base_sequence_simple():
    # Arrange
    idx = [0, 2, 1, 3, 0]
    alphabet = "ABCD"

    # Act
    result =pp.convert_idx_to_base_sequence(idx, alphabet)

    # Assert
    assert result == "ACBDA"